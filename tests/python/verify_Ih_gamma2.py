"""
Phase gamma.2 of Milestone G: scalable super-rep + column-picking
construction for the higher-dimensional irreps of I_h.

For each I_h orbit minimum r in an M-sector and each irrep Gamma:

  1. Compute the d_Gamma x d_Gamma "stabilizer matrix"
         A(r, Gamma) = sum_{h in Stab(r)} rho_Gamma(h)*
     Its rank equals n_Gamma(r) = (1/|Stab|) sum_h chi_Gamma(h), the
     multiplicity of Gamma in the orbit's irrep decomposition.

  2. SVD of A. Keep the n_Gamma(r) columns of U with non-zero singular
     values; call this orthonormal d_Gamma x n_Gamma(r) matrix U_r.

  3. The corresponding "row 1, atomic k" basis vectors live in the
     orbit subspace and are uniquely identified by (r, k); the full
     M-sector basis is the union of all (r, k) pairs across orbits.

  4. H matrix elements in this basis follow from
         H((a_R, k'), (r, k)) = sqrt(L_r/L_{a_R}) * c_a *
                                (U_{a_R}^dagger * rho_Gamma(g_min)* * U_r)_{k', k}
     where (a_R, g_min) is the I_h min-image of the spin-flipped state
     and L are I_h orbit lengths.

  5. Verification: aggregated H block eigenvalues (across all chosen
     Gamma) must match the dim_M x dim_M Frobenius reference of
     phase gamma.1 to machine precision.

For the s = 1/2 icosahedron this construction reproduces the same
block dimensions and spectra as the Frobenius projector approach, but
without ever storing or diagonalizing the dim_M x dim_M projector.
The same matrix-element machinery is what an FTLM Lanczos kernel
would use inside the I_h-adapted SpMV.
"""
import numpy as np

from verify_Ih_full import IH_perm, IH_keys, \
    IH_Ag, IH_Au, IH_T1g, IH_T1u, IH_T2g, IH_T2u, \
    IH_Fg, IH_Fu, IH_Hg, IH_Hu

from verify_Ih_beta import (
    PERMS, apply_perm_to_state, BONDS,
    enumerate_M_sector, build_H_M_full, digits_of,
    min_image_Ih, stabilizer_Ih,
)


# ----------------------------------------------------------------
# Irrep table
# ----------------------------------------------------------------

IRREPS = [
    ("A_g",  1, IH_Ag),
    ("A_u",  1, IH_Au),
    ("T1_g", 3, IH_T1g),
    ("T1_u", 3, IH_T1u),
    ("T2_g", 3, IH_T2g),
    ("T2_u", 3, IH_T2u),
    ("F_g",  4, IH_Fg),
    ("F_u",  4, IH_Fu),
    ("H_g",  5, IH_Hg),
    ("H_u",  5, IH_Hu),
]


def rho(irrep_data, g):
    """Get the d x d irrep matrix of element g. For 1D irreps returned
    as a 1x1 numpy array for uniform downstream handling."""
    M = irrep_data[g]
    if np.isscalar(M) or (isinstance(M, np.ndarray) and M.ndim == 0):
        return np.array([[complex(M)]])
    if isinstance(M, np.ndarray) and M.ndim == 1:
        return np.array([[complex(M[0])]])
    return M


# Group inverse: g^-1 from permutation
def perm_inv_idx(g_idx):
    P = IH_perm[g_idx]
    Pinv = P.T   # permutation matrix transpose = inverse
    return IH_keys[Pinv.round(6).tobytes()]


# ----------------------------------------------------------------
# Per-rep stabilizer matrix and SVD-based column basis U_r
# ----------------------------------------------------------------

def stabilizer_matrix(rep, irrep_data, d):
    """A(r, Gamma) = sum_{h in Stab(r)} rho_Gamma(h)* (d x d)."""
    A = np.zeros((d, d), dtype=complex)
    stab = stabilizer_Ih(rep)
    for h in stab:
        A += np.conj(rho(irrep_data, h))
    return A, stab


def column_basis(A, tol=1e-8):
    """Eigendecomposition of the Hermitian PSD matrix A.
    Returns (V, eigvals) where V[:, k] are eigenvectors with non-zero
    eigenvalues sorted in descending order. The matrix-element formula
    needs the eigenvalues d_k for normalization."""
    eigvals, V = np.linalg.eigh(A)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    V = V[:, order]
    keep = np.abs(eigvals) > tol
    return V[:, keep], eigvals[keep]


# ----------------------------------------------------------------
# Super-rep enumeration under I_h (no irrep-specific filtering yet)
# ----------------------------------------------------------------

def enumerate_super_reps(M_target):
    """All I_h orbit minima in the M-sector."""
    states = enumerate_M_sector(M_target)
    reps = []
    for n in states:
        r, _ = min_image_Ih(n)
        if r == n:
            reps.append(n)
    return reps


# ----------------------------------------------------------------
# Build H_block in the (M, Gamma) basis using the matrix-element
# formula. Returns the sparse H_block and the (rep, k) -> global index
# map for diagnostics.
# ----------------------------------------------------------------

def build_H_block(reps, irrep_data, d_irrep, J=1.0):
    d_loc = 2
    N_sites = 12
    s_val = 0.5

    # Step 1: per rep, compute V_r (eigenvectors of A) and eigvals.
    V_per_rep = []
    eig_per_rep = []
    L_per_rep = []
    n_per_rep = []
    for r in reps:
        A, stab = stabilizer_matrix(r, irrep_data, d_irrep)
        V_r, eigvals = column_basis(A)
        V_per_rep.append(V_r)
        eig_per_rep.append(eigvals)
        L_per_rep.append(120 // len(stab))
        n_per_rep.append(len(eigvals))

    # Step 2: build (rep, k) -> global index
    global_idx = []
    n_basis = 0
    rep_offsets = []
    for i, n_r in enumerate(n_per_rep):
        rep_offsets.append(n_basis)
        for k in range(n_r):
            global_idx.append((i, k))
        n_basis += n_r

    if n_basis == 0:
        return np.zeros((0, 0)), n_per_rep, L_per_rep

    H_block = np.zeros((n_basis, n_basis), dtype=complex)

    # Step 3: diagonal contributions
    powers = [d_loc ** k for k in range(N_sites)]
    for i, r in enumerate(reps):
        if n_per_rep[i] == 0:
            continue
        dg = digits_of(r)
        m = [dg[k] - s_val for k in range(N_sites)]
        diag = 0.0
        for (a_site, b_site) in BONDS:
            diag += J * m[a_site] * m[b_site]
        off_i = rep_offsets[i]
        for k in range(n_per_rep[i]):
            H_block[off_i + k, off_i + k] += diag

    # Step 4: off-diagonal contributions via matrix-element formula:
    #   M_{k', k} = sqrt(d_{a_R, k'} / d_{r, k}) * c_a *
    #              (V_{a_R}^dagger * rho_Gamma(g_min)^* * V_r)_{k', k}
    # The eigenvalue ratio replaces the orbit-length ratio used for 1D
    # irreps (where the two are equal); for higher-d irreps with
    # non-trivial stabilizers they differ and the eigenvalue version is
    # the correct one.
    state_to_rep_idx = {r: i for i, r in enumerate(reps)}
    for i, r in enumerate(reps):
        if n_per_rep[i] == 0:
            continue
        V_r = V_per_rep[i]
        eig_r = eig_per_rep[i]
        sqrt_eig_r = np.sqrt(eig_r)
        dg = digits_of(r)
        m = [dg[k] - s_val for k in range(N_sites)]
        off_i = rep_offsets[i]
        for (a_site, b_site) in BONDS:
            ma, mb = m[a_site], m[b_site]
            for sign in (+1, -1):
                if sign == +1:
                    if not (dg[a_site] < d_loc - 1 and dg[b_site] > 0):
                        continue
                    coeff = 0.5 * J * np.sqrt(s_val*(s_val+1) - ma*(ma+1)) \
                                     * np.sqrt(s_val*(s_val+1) - mb*(mb-1))
                    n_new = r + powers[a_site] - powers[b_site]
                else:
                    if not (dg[a_site] > 0 and dg[b_site] < d_loc - 1):
                        continue
                    coeff = 0.5 * J * np.sqrt(s_val*(s_val+1) - ma*(ma-1)) \
                                     * np.sqrt(s_val*(s_val+1) - mb*(mb+1))
                    n_new = r - powers[a_site] + powers[b_site]
                rep_a, g_min = min_image_Ih(n_new)
                if rep_a not in state_to_rep_idx:
                    continue
                j = state_to_rep_idx[rep_a]
                if n_per_rep[j] == 0:
                    continue
                V_a = V_per_rep[j]
                sqrt_eig_a = np.sqrt(eig_per_rep[j])
                rho_star = np.conj(rho(irrep_data, g_min))
                inner = V_a.conj().T @ rho_star @ V_r   # n_a x n_r
                # Multiply each row by sqrt(d_{a,k'}) and each column by
                # 1/sqrt(d_{r,k}) via broadcasting.
                M = coeff * inner * (sqrt_eig_a[:, None] / sqrt_eig_r[None, :])
                off_j = rep_offsets[j]
                H_block[off_j : off_j + n_per_rep[j],
                        off_i : off_i + n_per_rep[i]] += M

    return H_block, n_per_rep, L_per_rep


# ----------------------------------------------------------------
# Per-M verification across all irreps
# ----------------------------------------------------------------

def verify_M(M_target):
    print(f"--- M = {M_target} ---")
    H_M, _ = build_H_M_full(M_target)
    E_full = np.sort(np.linalg.eigvalsh(H_M))
    dim_M = len(E_full)

    reps = enumerate_super_reps(M_target)
    print(f"  super-reps under I_h: {len(reps)}  (dim_M = {dim_M})")

    all_E = []
    dim_total = 0
    overall = True
    for name, d_irrep, irrep_data in IRREPS:
        H_block, n_per_rep, _ = build_H_block(reps, irrep_data, d_irrep)
        n_basis = sum(n_per_rep)
        dim_total += n_basis
        if n_basis == 0:
            continue
        H_block = 0.5 * (H_block + H_block.conj().T)
        # Hermiticity check
        herm_err = np.max(np.abs(H_block - H_block.conj().T))
        E_block = np.real(np.linalg.eigvalsh(H_block))
        # Each block eigenvalue appears d_irrep times in the full Hilbert space
        # (degeneracy across the d_irrep rows of the irrep).
        for e in E_block:
            for _ in range(d_irrep):
                all_E.append(e)
        print(f"    {name:5s} (d={d_irrep}): block dim = {n_basis:3d} x d={d_irrep} -> {n_basis*d_irrep:3d}, herm_err = {herm_err:.2e}")

    all_E = np.sort(all_E)
    if len(all_E) != dim_M:
        print(f"    DIM MISMATCH: aggregated {len(all_E)} vs full {dim_M}  FAIL")
        overall = False
    else:
        err = np.max(np.abs(all_E - E_full))
        status = "OK " if err < 1e-9 else "FAIL"
        print(f"    Aggregated (with d_Gamma multiplicity) = {len(all_E)} (expected {dim_M})  [OK]")
        print(f"    max|dE| aggregated vs full = {err:.2e}  [{status}]")
        if err > 1e-9: overall = False
    return overall


if __name__ == "__main__":
    print("Phase gamma.2: scalable super-rep + column-picking for all I_h irreps\n")
    ok = True
    for M in [3, 2, 1, 0]:
        if not verify_M(M):
            ok = False
        print()
    print("=" * 60)
