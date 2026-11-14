import numpy as np

# ==========================================
# PROBLEM 4: MERMIN-PERES MAGIC SQUARE GAME
# ==========================================

def verify_mermin_peres():
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    kron = np.kron
    # 3x3 grid of two-qubit Pauli observables
    grid = [
        [kron(I, Z),  kron(Z, I),  kron(Z, Z)],
        [kron(X, I),  kron(I, X),  kron(X, X)],
        [-kron(X, Z), -kron(Z, X), kron(Y, Y)]
    ]
    
    I4 = np.eye(4, dtype=complex)
    print("=== Problem 4: Mermin-Peres Magic Square Verification ===")
    
    # Verify row products == +I4 and mutual commutation
    for r in range(3):
        prod = grid[r][0] @ grid[r][1] @ grid[r][2]
        comm = all(np.allclose(grid[r][a] @ grid[r][b], grid[r][b] @ grid[r][a]) 
                   for a in range(3) for b in range(a+1, 3))
        print(f"  Row {r+1}: Product == +I4 is {np.allclose(prod, I4)}, Commuting = {comm}")
        
    # Verify column products == -I4 and mutual commutation
    for c in range(3):
        prod = grid[0][c] @ grid[1][c] @ grid[2][c]
        comm = all(np.allclose(grid[a][c] @ grid[b][c], grid[b][c] @ grid[a][c]) 
                   for a in range(3) for b in range(a+1, 3))
        print(f"  Col {c+1}: Product == -I4 is {np.allclose(prod, -I4)}, Commuting = {comm}")

def test_z3_pauli_obstruction():
    """Tests generalized Z_3 Weyl-Heisenberg clock (Z) and shift (X) operators."""
    omega = np.exp(2j * np.pi / 3.0)
    Z3 = np.diag([1.0, omega, omega**2])
    X3 = np.roll(np.eye(3, dtype=complex), 1, axis=0)
    # Check commutation relation X3 * Z3 = omega * Z3 * X3
    comm_phase = np.allclose(X3 @ Z3, omega * (Z3 @ X3))
    print(f"  Z_3 Pauli Weyl-Heisenberg relation (X Z = omega Z X) holds: {comm_phase}\n")

# ==========================================
# PROBLEM 5: MUB ALTERNATING PROJECTION
# ==========================================

def polar_project(A):
    V, _, Wh = np.linalg.svd(A, full_matrices=False)
    return V @ Wh

def flat_modulus_project(A, d):
    """Projects matrix entries to flat magnitude 1/sqrt(d) while preserving phases."""
    phases = np.exp(1j * np.angle(A))
    return phases / np.sqrt(d)

def search_mubs(d, num_extra_bases=3, max_iter=1000, seed=42):
    """
    Searches for `num_extra_bases` unitaries U_1, ..., U_m in U(d) that are
    mutually unbiased with the computational basis I_d and with each other.
    Total bases k = 1 + num_extra_bases.
    """
    rng = np.random.default_rng(seed)
    m = num_extra_bases
    # Initialize m random unitaries
    bases = []
    for _ in range(m):
        Z = (rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))) / np.sqrt(2.0)
        bases.append(polar_project(Z))
        
    residuals = []
    for it in range(max_iter):
        # Measure max deviation from |<u|v>| = 1/sqrt(d) across all pairs
        max_dev = 0.0
        for a in range(m):
            dev_I = np.max(np.abs(np.abs(bases[a]) - 1.0 / np.sqrt(d)))
            max_dev = max(max_dev, dev_I)
            for b in range(a + 1, m):
                overlap = bases[a].conj().T @ bases[b]
                dev_ab = np.max(np.abs(np.abs(overlap) - 1.0 / np.sqrt(d)))
                max_dev = max(max_dev, dev_ab)
        residuals.append(max_dev)
        if max_dev < 1e-10:
            break
            
        # Alternating projection update for each basis
        new_bases = []
        for a in range(m):
            target = flat_modulus_project(bases[a], d)
            for b in range(m):
                if b != a:
                    # Enforce flat overlap with basis b: bases[a] ~ bases[b] @ flat(bases[b]^dagger @ bases[a])
                    overlap_flat = flat_modulus_project(bases[b].conj().T @ bases[a], d)
                    target += bases[b] @ overlap_flat
            new_bases.append(polar_project(target / m))
        bases = new_bases
        
    return bases, residuals

if __name__ == "__main__":
    verify_mermin_peres()
    test_z3_pauli_obstruction()
    
    print("=== Problem 5: Searching for k=4 MUBs (I_d + 3 Unitary Bases) ===")
    for d_val in [3, 4, 5, 6]:
        # Run best of 5 random restarts
        best_res = min(search_mubs(d_val, num_extra_bases=3, max_iter=800, seed=s)[1][-1] for s in range(5))
        print(f"  Dimension d={d_val} (k=4 MUBs): Min Unbiasedness Residual = {best_res:.3e}")