import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. TENSOR REORDERINGS & POLAR PROJECTION
# ==========================================

def reshuffle(U, d):
    """Reshuffling: U^R_{ij, kl} = U_{ik, jl} (bipartition 13|24)."""
    return U.reshape(d, d, d, d).transpose(0, 2, 1, 3).reshape(d*d, d*d)

def partial_transpose(U, d):
    """Partial transpose: U^Gamma_{ij, kl} = U_{il, kj} (bipartition 14|23)."""
    return U.reshape(d, d, d, d).transpose(0, 3, 2, 1).reshape(d*d, d*d)

def polar_project(A):
    """Closest unitary in Frobenius norm via SVD: P(A) = V W^dagger."""
    V, _, Wh = np.linalg.svd(A, full_matrices=False)
    return V @ Wh

def rank1_row_project(U, d):
    """Projects each row state |psi_{ij}> onto the nearest separable state |a_{ij}>|b_{ij}>."""
    U_sep = np.zeros_like(U, dtype=complex)
    for row_idx in range(d * d):
        psi_mat = U[row_idx].reshape(d, d)
        u, s, vh = np.linalg.svd(psi_mat, full_matrices=False)
        # Keep only the largest singular value and normalize to unit norm
        psi_rank1 = np.outer(u[:, 0], vh[0, :])
        U_sep[row_idx] = psi_rank1.reshape(-1)
    return U_sep

# ==========================================
# 2. METRICS: 2-UNITARITY & ENTANGLEMENT
# ==========================================

def compute_metrics(U, d):
    """Computes residual error epsilon(U), average entropy S_bar(U), and purity gamma_bar(U)."""
    I = np.eye(d * d)
    UR = reshuffle(U, d)
    UG = partial_transpose(U, d)
    
    err_U  = np.linalg.norm(U.conj().T @ U - I, 'fro')
    err_UR = np.linalg.norm(UR.conj().T @ UR - I, 'fro')
    err_UG = np.linalg.norm(UG.conj().T @ UG - I, 'fro')
    eps = max(err_U, err_UR, err_UG) / d
    
    entropies = []
    purities = []
    for M in (U, UR, UG):
        # Singular values of M give eigenvalues of reduced density matrix: p_m = sigma_m^2 / d^2
        sigmas = np.linalg.svd(M, compute_uv=False)
        p = (sigmas ** 2) / (d ** 2)
        p = p / np.sum(p)  # Ensure strict normalization
        p_nz = p[p > 1e-15]
        entropies.append(-np.sum(p_nz * np.log2(p_nz)))
        purities.append(np.sum(p ** 2))
        
    return eps, np.mean(entropies), np.mean(purities)

# ==========================================
# 3. SEEDING STRATEGIES (MODES A, B, C)
# ==========================================

def haar_random_unitary(n, rng=None):
    """Draws a Haar-random unitary from U(n)."""
    if rng is None:
        rng = np.random.default_rng()
    Z = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / np.sqrt(2.0)
    Q, R = np.linalg.qr(Z)
    d_diag = np.diag(R)
    ph = d_diag / np.abs(d_diag)
    return Q * ph

def seed_mode_A(d, rng=None):
    """Mode A: Haar-random unitary in U(d^2)."""
    return haar_random_unitary(d * d, rng)

def seed_mode_B(d, eta=0.15, rng=None):
    """Mode B: Perturbed permutation matrix U_0 = P(Perm + eta * G)."""
    if rng is None:
        rng = np.random.default_rng()
    n = d * d
    # Construct a cyclic Latin-like permutation matrix P
    P = np.zeros((n, n), dtype=complex)
    perm_cols = []
    for i in range(d):
        for j in range(d):
            k = (i + j) % d
            l = (i + 2 * j) % d
            perm_cols.append(k * d + l)
    # Break ties to make P a true permutation matrix
    used = set()
    unassigned_rows = []
    for r, c in enumerate(perm_cols):
        if c not in used:
            P[r, c] = 1.0
            used.add(c)
        else:
            unassigned_rows.append(r)
    remaining_cols = [c for c in range(n) if c not in used]
    rng.shuffle(remaining_cols)
    for r, c in zip(unassigned_rows, remaining_cols):
        P[r, c] = 1.0
        
    G = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / np.sqrt(2.0)
    return polar_project(P + eta * G)

def seed_mode_C_d6(rng=None):
    """Mode C: Coarse-grained OLS block seed from Table 4 for d = 6."""
    if rng is None:
        rng = np.random.default_rng()
    # Table 4 coarse-grained pairs (Rank group in {0,1,2}, Regiment group in {0,1,2})
    table4 = [
        [(0,0), (0,1), (2,2), (2,0), (1,1), (1,2)],
        [(2,0), (2,1), (1,2), (1,0), (0,1), (0,2)],
        [(1,2), (1,0), (0,1), (0,2), (2,0), (2,1)],
        [(0,2), (0,0), (2,1), (2,2), (1,0), (1,1)],
        [(2,1), (2,2), (1,0), (1,1), (0,2), (0,0)],
        [(1,1), (1,2), (0,0), (0,1), (2,2), (2,0)]
    ]
    U0 = np.zeros((36, 36), dtype=complex)
    for r_grp in range(3):
        for c_grp in range(3):
            # Find the 4 cells (i, j) belonging to this coarse-grained group
            cells = [i * 6 + j for i in range(6) for j in range(6) if table4[i][j] == (r_grp, c_grp)]
            # Find the 4 basis states (k, l) belonging to this group
            ks = [2 * r_grp, 2 * r_grp + 1]
            ls = [2 * c_grp, 2 * c_grp + 1]
            kl_indices = [k * 6 + l for k in ks for l in ls]
            # Embed a random 4x4 unitary block
            U_block = haar_random_unitary(4, rng)
            U0[np.ix_(cells, kl_indices)] = U_block
    return U0

# ==========================================
# 4. ALTERNATING POLAR-PROJECTION SOLVER
# ==========================================

def run_alternating_projection(d, mode='A', max_iter=500, tol=1e-12, unentangled=False, eta=0.15, seed=None):
    rng = np.random.default_rng(seed)
    if mode == 'A':
        U = seed_mode_A(d, rng)
    elif mode == 'B':
        U = seed_mode_B(d, eta=eta, rng=rng)
    elif mode == 'C':
        if d != 6:
            raise ValueError("Mode C is specific to d=6 Table 4 block structure.")
        U = seed_mode_C_d6(rng)
        
    history = {'eps': [], 'S': [], 'gamma': []}
    
    for it in range(max_iter):
        eps, S_bar, gamma_bar = compute_metrics(U, d)
        history['eps'].append(eps)
        history['S'].append(S_bar)
        history['gamma'].append(gamma_bar)
        
        if eps < tol and not unentangled:
            break
            
        # Step 1: Reshuffle projection
        U = reshuffle(polar_project(reshuffle(U, d)), d)
        # Step 2: Partial-transpose projection
        U = partial_transpose(polar_project(partial_transpose(U, d)), d)
        # Optional Problem 3 step: Rank-1 separable projection on each officer state
        if unentangled:
            U = rank1_row_project(U, d)
        # Step 3: Unitary restoration
        U = polar_project(U)
        
    return U, history

# ==========================================
# 5. PROBLEM 1: OFFICER STATE INSPECTOR
# ==========================================

def analyze_36_officers(U):
    """Inspects Schmidt spectra and support sizes of the 36 officer states |psi_{ij}>."""
    schmidt_ranks = []
    entropies = []
    supports = []
    for r in range(36):
        psi = U[r].reshape(6, 6)
        sigmas = np.linalg.svd(psi, compute_uv=False)
        p = sigmas ** 2
        p_nz = p[p > 1e-10]
        schmidt_ranks.append(len(p_nz))
        entropies.append(-np.sum(p_nz * np.log2(p_nz)))
        supports.append(np.sum(np.abs(U[r]) > 1e-5))
    return np.array(schmidt_ranks), np.array(entropies), np.array(supports)

# ==========================================
# 6. PROBLEM 2: HIGUCHI-SUDBERY STATE (d=2)
# ==========================================

def verify_higuchi_sudbery():
    """Constructs the exact analytical 4-qubit Higuchi-Sudbery state and checks its metrics."""
    omega = np.exp(2j * np.pi / 3.0)
    psi = np.zeros((2, 2, 2, 2), dtype=complex)
    psi[0,0,1,1] = 1.0; psi[1,1,0,0] = 1.0
    psi[0,1,0,1] = omega; psi[1,0,1,0] = omega
    psi[0,1,1,0] = omega**2; psi[1,0,0,1] = omega**2
    psi /= np.sqrt(6.0)
    
    # Flatten to 4x4 matrix U where |Psi> = (1/2) sum U_{ij,kl} |ijkl>
    U_hs = 2.0 * psi.reshape(4, 4)
    eps, S_bar, gamma_bar = compute_metrics(U_hs, d=2)
    print(f"[Problem 2] Analytical Higuchi-Sudbery 4-qubit state:")
    print(f"  Average Purity gamma_bar = {gamma_bar:.6f} (Theoretical: 1/3 = 0.333333)")
    print(f"  Average Entropy S_bar    = {S_bar:.6f} bits (log2(3) = {np.log2(3):.6f} bits)")
    print(f"  1 + 0.5*log2(3) benchmark = {1 + 0.5*np.log2(3):.6f} bits\n")

if __name__ == "__main__":
    # 1. Problem 1: Compare Modes A, B, C for d=6
    print("Running Problem 1: AME(4,6) Seeding Mode Comparison...")
    U_A, hist_A = run_alternating_projection(6, mode='A', max_iter=400, seed=42)
    U_B, hist_B = run_alternating_projection(6, mode='B', max_iter=400, eta=0.2, seed=42)
    U_C, hist_C = run_alternating_projection(6, mode='C', max_iter=400, seed=7)
    
    print(f"  Mode A final eps: {hist_A['eps'][-1]:.3e} (iters: {len(hist_A['eps'])})")
    print(f"  Mode B final eps: {hist_B['eps'][-1]:.3e} (iters: {len(hist_B['eps'])})")
    print(f"  Mode C final eps: {hist_C['eps'][-1]:.3e} (iters: {len(hist_C['eps'])})")
    
    ranks, ents, supps = analyze_36_officers(U_C)
    print(f"  Converged Mode C Officers -> Mean Officer Entropy: {np.mean(ents):.4f} bits, Schmidt ranks: {np.unique(ranks)}\n")
    
    # 2. Problem 2: d=2 Obstruction
    verify_higuchi_sudbery()
    _, hist_d2 = run_alternating_projection(2, mode='A', max_iter=200, seed=10)
    print(f"[Problem 2] Alternating Projection d=2 Plateau:")
    print(f"  Plateau eps = {hist_d2['eps'][-1]:.4f}, S_bar = {hist_d2['S'][-1]:.4f} bits, gamma_bar = {hist_d2['gamma'][-1]:.4f}\n")
    
    # 3. Problem 3: Entangled vs Unentangled QLS across d=2,3,4,5,6
    print("Running Problem 3: Entangled vs. Unentangled QLS (d=2..6)...")
    for d_val in [2, 3, 4, 5, 6]:
        _, h_ent = run_alternating_projection(d_val, mode='A', max_iter=300, unentangled=False, seed=12)
        _, h_sep = run_alternating_projection(d_val, mode='A', max_iter=300, unentangled=True, seed=12)
        print(f"  d={d_val}: Entangled final eps = {h_ent['eps'][-1]:.3e} | Unentangled (Rank-1) final eps = {h_sep['eps'][-1]:.3e}")