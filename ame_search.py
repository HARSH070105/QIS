import itertools
import math
import random
import time
import torch

# ---------------------------------------------------------------------------
# 1. Precompute Vectorized Gather Tables & Edge-to-Cut Incidence Map
# ---------------------------------------------------------------------------
def build_gather_tables(device: torch.device):
    reps = [(0,) + comb for comb in itertools.combinations(range(1, 8), 3)]
    all_parties = set(range(8))
    base_idx = torch.arange(4**8, dtype=torch.int64, device=device).reshape((4,) * 8)

    g_list, inv_g_list, qubit_cuts = [], [], []
    for s in reps:
        sc = tuple(sorted(all_parties - set(s)))
        perm = s + sc
        g_flat = base_idx.permute(*perm).reshape(-1)
        inv_g_flat = torch.argsort(g_flat)
        g_list.append(g_flat.reshape(256, 256))
        inv_g_list.append(inv_g_flat)

        q_s = [2 * p + b for p in s for b in (0, 1)]
        q_sc = [2 * p + b for p in sc for b in (0, 1)]
        qubit_cuts.append((q_s, q_sc))

    gather_idx = torch.stack(g_list, dim=0)          # (35, 256, 256)
    inv_gather_idx = torch.stack(inv_g_list, dim=0)  # (35, 65536)
    return gather_idx, inv_gather_idx, qubit_cuts


# ---------------------------------------------------------------------------
# 2. Ultra-Fast Incremental Bitwise Tabu / Annealing Search over GF(2)^(16x16)
# ---------------------------------------------------------------------------
def rank8_gf2(rows_8: list) -> int:
    """Computes exact GF(2) rank of eight 8-bit integers via bitwise elimination."""
    r_list = rows_8[:]
    rank = 0
    for bit in range(8):
        mask = 1 << bit
        pivot = -1
        for i in range(rank, 8):
            if r_list[i] & mask:
                pivot = i
                break
        if pivot == -1:
            continue
        r_list[rank], r_list[pivot] = r_list[pivot], r_list[rank]
        p_val = r_list[rank]
        for i in range(rank + 1, 8):
            if r_list[i] & mask:
                r_list[i] ^= p_val
        rank += 1
    return rank


def optimize_16qubit_graph_incremental(qubit_cuts: list, max_time_sec: float = 1.4, seed: int = 42):
    """
    Incremental bitwise Tabu + Simulated Annealing over 16x16 binary graphs.
    Each edge flip only recomputes the <= 20 affected cuts (25,000+ flips/sec).
    """
    random.seed(seed)
    gf4_blocks = {
        0: [[0, 0], [0, 0]],
        1: [[1, 0], [0, 1]],
        2: [[0, 1], [1, 1]],
        3: [[1, 1], [1, 0]],
    }
    circ_row = [0, 1, 1, 2, 2, 1, 1]

    base_adj = [[0] * 16 for _ in range(16)]
    for i in range(7):
        for j in range(i + 1, 7):
            blk = gf4_blocks[circ_row[(j - i) % 7]]
            for bi in range(2):
                for bj in range(2):
                    base_adj[2 * i + bi][2 * j + bj] = blk[bi][bj]
                    base_adj[2 * j + bj][2 * i + bi] = blk[bj][bi]

    # Precompute for each of the 120 edges (u, v) which cuts it crosses: (cut_id, row_idx, bit_mask)
    edges = [(u, v) for u in range(16) for v in range(u + 1, 16)]
    edge_incidence = []
    for u, v in edges:
        inc = []
        for c_id, (q_s, q_sc) in enumerate(qubit_cuts):
            if u in q_s and v in q_sc:
                inc.append((c_id, q_s.index(u), 1 << q_sc.index(v)))
            elif v in q_s and u in q_sc:
                inc.append((c_id, q_s.index(v), 1 << q_sc.index(u)))
        edge_incidence.append(inc)

    best_adj = None
    best_total_delta = 999
    best_max_delta = 999
    best_unitary = 0

    t_start = time.time()
    restart = 0
    while time.time() - t_start < max_time_sec:
        restart += 1
        adj = [row[:] for row in base_adj]
        for q in range(14):
            for b in (14, 15):
                bit = random.randint(0, 1)
                adj[q][b] = adj[b][q] = bit
        if restart > 2:
            # Perturb 4 random internal edges to escape GF(4) parity locks
            for _ in range(4):
                u, v = random.choice(edges)
                adj[u][v] ^= 1
                adj[v][u] ^= 1

        # Build initial 35x8 bit-rows and per-cut deltas
        cut_rows = []
        cut_deltas = []
        for q_s, q_sc in qubit_cuts:
            rows = []
            for r in q_s:
                val = 0
                for c_idx, c in enumerate(q_sc):
                    if adj[r][c]:
                        val |= (1 << c_idx)
                rows.append(val)
            cut_rows.append(rows)
            rk = rank8_gf2(rows)
            cut_deltas.append((1 << (8 - rk)) - 1)

        total_delta = sum(cut_deltas)
        tabu_until = [0] * 120

        for step in range(1, 900):
            # Pick best non-tabu (or aspiration) move from a random tournament of 16 edges
            best_e = -1
            best_diff = 999
            best_new_deltas = None

            candidates = random.sample(range(120), 16)
            for e_idx in candidates:
                inc = edge_incidence[e_idx]
                if not inc:
                    continue
                diff = 0
                upd = []
                for c_id, r_i, mask in inc:
                    rows = cut_rows[c_id]
                    rows[r_i] ^= mask
                    rk = rank8_gf2(rows)
                    rows[r_i] ^= mask
                    new_d = (1 << (8 - rk)) - 1
                    diff += new_d - cut_deltas[c_id]
                    upd.append((c_id, r_i, mask, new_d))

                # Aspiration criterion: allow tabu edge if it beats global best
                if tabu_until[e_idx] > step and (total_delta + diff >= best_total_delta):
                    continue
                if diff < best_diff:
                    best_diff = diff
                    best_e = e_idx
                    best_new_deltas = upd

            if best_e == -1:
                continue

            # Accept improving, plateau (diff == 0), or mild thermal uphill moves
            temp = max(0.05, 0.45 * (1.0 - step / 900.0))
            if best_diff <= 0 or random.random() < math.exp(-best_diff / temp):
                u, v = edges[best_e]
                adj[u][v] ^= 1
                adj[v][u] ^= 1
                for c_id, r_i, mask, new_d in best_new_deltas:
                    cut_rows[c_id][r_i] ^= mask
                    cut_deltas[c_id] = new_d
                total_delta += best_diff
                tabu_until[best_e] = step + random.randint(7, 14)

                max_d = max(cut_deltas)
                u_cnt = sum(1 for d in cut_deltas if d == 0)
                if (total_delta < best_total_delta) or (
                    total_delta == best_total_delta and max_d < best_max_delta
                ):
                    best_total_delta = total_delta
                    best_max_delta = max_d
                    best_unitary = u_cnt
                    best_adj = [row[:] for row in adj]

    return best_adj, best_total_delta / 35.0, float(best_max_delta), best_unitary


# ---------------------------------------------------------------------------
# 3. Tangent-Kernel Subspace Saddle Breaker (Lifts Zero Singular Values Cleanly)
# ---------------------------------------------------------------------------
@torch.no_grad()
def build_tangent_kernel_seed(
    best_adj: list,
    gather_idx: torch.Tensor,
    inv_gather_idx: torch.Tensor,
    device: torch.device,
    proj_iters: int = 8,
    seed: int = 42,
):
    """
    Synthesizes the 16-qubit graph state and constructs a perturbation direction
    in the intersection of:
      1. The 128-D zero-singular-value kernel on the deficient cuts (Delta > 0)
      2. The skew-Hermitian unitary tangent space on the unitary cuts (Delta == 0)
    """
    torch.manual_seed(seed)
    idx = torch.arange(65536, dtype=torch.int64, device=device)
    bits = ((idx[:, None] >> torch.arange(15, -1, -1, device=device)) & 1).to(torch.float64)
    adj_upper = torch.triu(torch.tensor(best_adj, dtype=torch.float64, device=device), diagonal=1)
    quad_phase = torch.sum((bits @ adj_upper) * bits, dim=1)
    psi_exact = torch.exp(1j * torch.pi * quad_phase) / 256.0

    # SVD of the exact graph state across all 35 cuts
    mats0 = psi_exact[gather_idx]  # (35, 256, 256)
    U0, S0, Vh0 = torch.linalg.svd(mats0, full_matrices=False)
    rho0 = torch.bmm(mats0, mats0.mH)
    deltas0 = 256.0 * torch.sum(rho0.real**2 + rho0.imag**2, dim=(-2, -1)) - 1.0
    deficient_mask = deltas0 > 1e-5

    # Initialize perturbation in the exact nullspace (singular values < 1e-5) of deficient cuts
    delta_psi = torch.randn(65536, dtype=torch.complex128, device=device)
    delta_psi = delta_psi / torch.linalg.norm(delta_psi)

    for _ in range(proj_iters):
        d_mats = delta_psi[gather_idx]  # (35, 256, 256)
        proj_mats = torch.zeros_like(d_mats)

        for c in range(35):
            if deficient_mask[c]:
                # Project onto the 128-D zero-singular-value kernel (columns 128..255) + skew-Hermitian active block
                U_ker = U0[c, :, 128:]      # (256, 128)
                Vh_ker = Vh0[c, 128:, :]    # (128, 256)
                ker_part = U_ker @ (U_ker.mH @ d_mats[c] @ Vh_ker.mH) @ Vh_ker

                U_act = U0[c, :, :128]
                Vh_act = Vh0[c, :128, :]
                M_act = U_act.mH @ d_mats[c] @ Vh_act.mH
                skew_act = 0.5 * (M_act - M_act.mH)
                proj_mats[c] = 1.6 * ker_part + 0.4 * (U_act @ skew_act @ Vh_act)
            else:
                # Project onto skew-Hermitian tangent space of U(256) at M0[c]:
                # P_tangent(dM) = 0.5 * (dM - 256 * M0 @ dM^H @ M0)
                M_c = mats0[c]
                proj_mats[c] = 0.5 * (d_mats[c] - 256.0 * (M_c @ d_mats[c].mH @ M_c))

        delta_psi = proj_mats.reshape(35, 65536).gather(1, inv_gather_idx).mean(dim=0)
        # Orthogonalize against psi_exact
        overlap = torch.vdot(psi_exact, delta_psi)
        delta_psi = delta_psi - overlap * psi_exact
        delta_psi = delta_psi / torch.linalg.norm(delta_psi)

    # Line-search optimal saddle-breaking amplitude eta so all singular values lift > 0
    best_eta = 0.20
    best_combined = 999.0
    for eta_cand in [0.12, 0.16, 0.20, 0.24, 0.28]:
        cand = psi_exact + eta_cand * delta_psi
        cand = cand / torch.linalg.norm(cand)
        c_mats = cand[gather_idx]
        c_rho = torch.bmm(c_mats, c_mats.mH)
        c_deltas = torch.clamp(
            256.0 * torch.sum(c_rho.real**2 + c_rho.imag**2, dim=(-2, -1)) - 1.0,
            min=0.0,
        )
        score = torch.mean(c_deltas).item() + 0.15 * torch.max(c_deltas).item()
        if score < best_combined:
            best_combined = score
            best_eta = eta_cand

    psi_seed = psi_exact + best_eta * delta_psi
    psi_seed = psi_seed / torch.linalg.norm(psi_seed)
    return psi_exact, psi_seed


# ---------------------------------------------------------------------------
# 4. Exact Analytical Gradient with Orthogonal Radial Tether
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_loss_and_grad(
    psi_flat: torch.Tensor,
    gather_idx: torch.Tensor,
    inv_gather_idx: torch.Tensor,
    gamma: float = 0.0,
    mu_barrier: float = 0.0,
    eps_barrier: float = 0.10,
    norm_tether: float = 1.0,
):
    norm_sq = torch.sum(psi_flat.real**2 + psi_flat.imag**2)
    norm = torch.sqrt(norm_sq)
    u_flat = psi_flat / norm

    mats = u_flat[gather_idx]       # (35, 256, 256)
    rho = torch.bmm(mats, mats.mH)  # (35, 256, 256)

    purities = torch.sum(rho.real**2 + rho.imag**2, dim=(-2, -1))
    deltas = torch.clamp(256.0 * purities - 1.0, min=0.0)

    mean_loss = torch.mean(deltas).item()
    max_delta = torch.max(deltas).item()
    purity_obj = torch.mean(deltas + gamma * (deltas**2)).item()

    # Exact fourth-moment Wirtinger gradient
    w_purity = (1024.0 / 35.0) * (1.0 + 2.0 * gamma * deltas)
    grad_mats = w_purity[:, None, None] * torch.bmm(rho, mats)

    # Optional Cholesky Log-Det Spectral Barrier to lift small singular values
    barrier_obj = 0.0
    if mu_barrier > 0.0:
        eye = torch.eye(256, dtype=rho.dtype, device=rho.device).unsqueeze(0)
        reg_rho = 256.0 * rho + eps_barrier * eye
        L, info = torch.linalg.cholesky_ex(reg_rho)
        if torch.all(info == 0):
            diag_L = torch.diagonal(L, dim1=-2, dim2=-1).real
            logdet = 2.0 * torch.sum(torch.log(diag_L), dim=-1) / 256.0
            j_val = torch.clamp(math.log(1.0 + eps_barrier) - logdet, min=0.0)
            barrier_obj = mu_barrier * torch.mean(j_val).item()
            inv_m = torch.cholesky_solve(mats, L)
            grad_mats = grad_mats - (2.0 * mu_barrier / 35.0) * inv_m

    # Inverse gather + Riemannian projection onto sphere tangent space
    grad_u = grad_mats.reshape(35, 65536).gather(1, inv_gather_idx).sum(dim=0)
    proj = torch.sum(u_flat.real * grad_u.real + u_flat.imag * grad_u.imag)
    riemannian_grad = (grad_u - proj * u_flat) / norm

    # Orthogonal radial tether: keeps ||psi|| == 1 inside L-BFGS without external resets
    radial_diff = (norm_sq - 1.0).item()
    tether_obj = 0.25 * norm_tether * (radial_diff**2)
    total_grad = riemannian_grad + (norm_tether * radial_diff) * psi_flat

    return purity_obj + barrier_obj + tether_obj, mean_loss, max_delta, total_grad


# ---------------------------------------------------------------------------
# 5. Complete Monotonic Pipeline
# ---------------------------------------------------------------------------
def run_monotonic_ame84_search():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    gather_idx, inv_gather_idx, qubit_cuts = build_gather_tables(device)

    # --- Stage 0: Incremental Bitwise Tabu Search over GF(2)^(16x16) ---
    best_adj, _, _, n_unitary = optimize_16qubit_graph_incremental(
        qubit_cuts, max_time_sec=1.4, seed=42
    )
    psi_exact, psi_seed = build_tangent_kernel_seed(
        best_adj, gather_idx, inv_gather_idx, device, proj_iters=8, seed=42
    )

    _, exact_loss, exact_max, _ = evaluate_loss_and_grad(
        psi_exact, gather_idx, inv_gather_idx, gamma=0.0, mu_barrier=0.0
    )
    _, init_loss, init_max, _ = evaluate_loss_and_grad(
        psi_seed, gather_idx, inv_gather_idx, gamma=0.0, mu_barrier=0.0
    )

    print(f"[Device: {device} | Full 35 Bipartitions | Tangent-Kernel Saddle Breaker]")
    print(
        f"Incremental Tabu 16-Qubit Graph  -> Mean Loss: {exact_loss:.6e} "
        f"({n_unitary}/35 cuts exact 0) | Max Cut: {exact_max:.4f} ({time.time() - t0:.2f}s)"
    )
    print(
        f"Tangent-Kernel Lifted Seed       -> Mean Loss: {init_loss:.6e} "
        f"| Max Cut: {init_max:.4f} ({time.time() - t0:.2f}s)"
    )

    best_psi = psi_exact.clone() if exact_loss < init_loss else psi_seed.clone()
    best_loss = min(exact_loss, init_loss)

    # Parameter vector for L-BFGS (no external in-place mutation inside stages!)
    param_real = torch.view_as_real(psi_seed.clone()).reshape(-1).requires_grad_(True)

    # Three-stage monotonic schedule ending with exact unweighted Mean Loss (gamma=0, mu=0)
    schedules = [
        ("Stage 1: Kernel Lift ", 150, 0.20, 0.025, 0.10),
        ("Stage 2: Cut Balance ", 150, 0.05, 0.005, 0.08),
        ("Stage 3: Exact Polish", 250, 0.00, 0.000, 0.05),
    ]

    for stage_name, max_steps, gamma_val, mu_val, eps_val in schedules:
        lbfgs = torch.optim.LBFGS(
            [param_real],
            lr=1.0,
            max_iter=25,
            history_size=80,
            tolerance_grad=1e-13,
            tolerance_change=1e-15,
            line_search_fn="strong_wolfe",
        )

        def closure():
            psi_c = torch.view_as_complex(param_real.detach().reshape(65536, 2))
            obj_val, m_loss, _, total_grad = evaluate_loss_and_grad(
                psi_c,
                gather_idx,
                inv_gather_idx,
                gamma=gamma_val,
                mu_barrier=mu_val,
                eps_barrier=eps_val,
            )
            param_real.grad = torch.view_as_real(total_grad).reshape(-1).clone()
            return torch.tensor(obj_val, dtype=torch.float64, device=device)

        for outer in range(1, (max_steps // 25) + 1):
            lbfgs.step(closure)
            with torch.no_grad():
                psi_c = torch.view_as_complex(param_real.detach().reshape(65536, 2))
                psi_norm = psi_c / torch.linalg.norm(psi_c)
                _, m_loss, max_d, _ = evaluate_loss_and_grad(
                    psi_norm, gather_idx, inv_gather_idx, gamma=0.0, mu_barrier=0.0
                )
                if m_loss < best_loss:
                    best_loss = m_loss
                    best_psi = psi_norm.clone()
            print(
                f"[{stage_name} {outer * 25:3d}] Mean Loss = {m_loss:.6e} "
                f"| Best = {best_loss:.6e} | Max Cut = {max_d:.4f} ({time.time() - t0:.1f}s)"
            )

        # At stage boundary only, warm-start next L-BFGS instance from best_psi if stage drifted
        with torch.no_grad():
            psi_curr = torch.view_as_complex(param_real.detach().reshape(65536, 2))
            _, curr_m, _, _ = evaluate_loss_and_grad(
                psi_curr, gather_idx, inv_gather_idx, gamma=0.0, mu_barrier=0.0
            )
            if best_loss < curr_m and stage_name != "Stage 1: Kernel Lift ":
                param_real.copy_(torch.view_as_real(best_psi).reshape(-1))

    # --- Final Diagnostic Across All 35 Bipartitions ---
    _, final_loss, final_max, _ = evaluate_loss_and_grad(
        best_psi, gather_idx, inv_gather_idx, gamma=0.0, mu_barrier=0.0
    )
    mats = (best_psi / torch.linalg.norm(best_psi))[gather_idx]
    svals = torch.linalg.svdvals(mats) * 16.0
    min_sval = torch.min(svals).item()
    cut_losses = 256.0 * torch.sum((svals / 16.0) ** 4, dim=-1) - 1.0
    near_unitary = int(torch.sum(cut_losses < 1e-3).item())
    under_1e2 = int(torch.sum(cut_losses < 1e-2).item())

    print("-" * 78)
    print(
        f"Best Verified 35-Cut Mean Loss: {final_loss:.6e} | Worst Cut: {final_max:.6e}"
    )
    print(
        f"Cuts < 1e-3: {near_unitary}/35 | Cuts < 1e-2: {under_1e2}/35 | Min Singular Value (16*sigma_min): {min_sval:.4f}"
    )
    return best_psi.reshape((4,) * 8)


if __name__ == "__main__":
    run_monotonic_ame84_search()