"""
Python-Referenz der matrixfreien I_h-SpMV. Spiegelt die Logik von
`spmv_pg_Ih_matlab.m` und verifiziert sie gegen den expliziten
sparse H_block aus `verify_Ih_gamma2.py`.

Für jedes (M, Gamma) auf dem s=1/2 Ikosaeder:
    1. enumerate super-reps + V_r/lambda_r (aus gamma2)
    2. baue H_block sparse (Referenz)
    3. wende H_block * X an  (Y_sparse)
    4. wende matrix-free SpMV an  (Y_spmv)
    5. vergleiche ||Y_sparse - Y_spmv|| / ||Y_sparse||

Beide Pfade müssen bis Maschinenpräzision übereinstimmen.
"""

import numpy as np

from verify_Ih_full import IH_T1g, IH_T1u, IH_T2g, IH_T2u, \
    IH_Fg, IH_Fu, IH_Hg, IH_Hu, IH_Ag, IH_Au

from verify_Ih_beta import (
    PERMS, apply_perm_to_state, BONDS,
    enumerate_M_sector, build_H_M_full,
    min_image_Ih, digits_of,
    D_LOC, N_SITES, S_VAL,
)

from verify_Ih_gamma2 import (
    rho, stabilizer_matrix, column_basis, build_H_block,
)


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


def enumerate_with_gamma2(M_target, irrep_data, d_irrep, tol=1e-8):
    """Returns reps (with n_Gamma > 0), V_per_rep, eig_per_rep, n_per_rep."""
    states = enumerate_M_sector(M_target)
    reps_all = [n for n in states if min_image_Ih(n)[0] == n]
    reps, V_per, eig_per, n_per = [], [], [], []
    for r in reps_all:
        A, _ = stabilizer_matrix(r, irrep_data, d_irrep)
        V, eigvals = column_basis(A, tol=tol)
        if len(eigvals) > 0:
            reps.append(r)
            V_per.append(V)
            eig_per.append(eigvals)
            n_per.append(len(eigvals))
    return reps, V_per, eig_per, n_per


def spmv_pg_Ih(super_reps, V_per_rep, eig_per_rep, n_per_rep,
               irrep_data, d_irrep, X, J=1.0):
    """Matrix-free I_h SpMV. X: (n_basis, B) array; returns Y of same shape."""
    s_val = 0.5
    d_loc = 2
    N = 12
    powers = [d_loc ** k for k in range(N)]

    n_reps = len(super_reps)
    rep_offsets = [0]
    for n in n_per_rep[:-1]:
        rep_offsets.append(rep_offsets[-1] + n)
    n_basis = sum(n_per_rep)
    assert X.shape[0] == n_basis

    state_to_rep = {r: i for i, r in enumerate(super_reps)}
    sqrt_eig = [np.sqrt(e) for e in eig_per_rep]

    Y = np.zeros_like(X, dtype=complex)

    # Diagonal Sz Sz
    for i, r in enumerate(super_reps):
        dg = digits_of(r)
        m = [dg[k] - s_val for k in range(N)]
        diag_i = sum(J * m[a] * m[b] for (a, b) in BONDS)
        off = rep_offsets[i]
        ni = n_per_rep[i]
        Y[off:off+ni, :] += diag_i * X[off:off+ni, :]

    # Off-diagonal: scatter from source reps
    for i, r in enumerate(super_reps):
        V_r = V_per_rep[i]
        sqrt_eig_r = sqrt_eig[i]
        off_i = rep_offsets[i]
        n_i = n_per_rep[i]
        X_r = X[off_i:off_i+n_i, :]
        dg = digits_of(r)
        m = [dg[k] - s_val for k in range(N)]

        for (a_site, b_site) in BONDS:
            ma, mb = m[a_site], m[b_site]
            for sign in (+1, -1):
                if sign == +1:
                    if not (dg[a_site] < d_loc - 1 and dg[b_site] > 0):
                        continue
                    c_a = 0.5 * J * np.sqrt(s_val*(s_val+1) - ma*(ma+1)) \
                                  * np.sqrt(s_val*(s_val+1) - mb*(mb-1))
                    n_new = r + powers[a_site] - powers[b_site]
                else:
                    if not (dg[a_site] > 0 and dg[b_site] < d_loc - 1):
                        continue
                    c_a = 0.5 * J * np.sqrt(s_val*(s_val+1) - ma*(ma-1)) \
                                  * np.sqrt(s_val*(s_val+1) - mb*(mb+1))
                    n_new = r - powers[a_site] + powers[b_site]
                rep_a, g_min = min_image_Ih(n_new)
                j = state_to_rep.get(rep_a)
                if j is None or n_per_rep[j] == 0:
                    continue
                V_a = V_per_rep[j]
                sqrt_eig_a = sqrt_eig[j]
                off_j = rep_offsets[j]
                n_j = n_per_rep[j]
                # rho(g_a)^T = conj(rho(g_min)) for unitary irreps.
                rho_T = np.conj(rho(irrep_data, g_min))
                inner = V_a.conj().T @ rho_T @ V_r              # n_j x n_i
                M_block = c_a * inner * (sqrt_eig_a[:, None] / sqrt_eig_r[None, :])
                Y[off_j:off_j+n_j, :] += M_block @ X_r
    return Y


def verify_M(M_target, B_block=4, tol_rel=1e-12, seed=31415):
    print(f"--- M = {M_target} ---")
    rng = np.random.default_rng(seed + M_target)
    ok_all = True
    for name, d, irrep_data in IRREPS:
        reps, V_per, eig_per, n_per = enumerate_with_gamma2(M_target, irrep_data, d)
        n_basis = sum(n_per)
        if n_basis == 0:
            print(f"    {name:5s} (d={d}): empty")
            continue

        # Reference: build sparse H via the gamma2 routine, multiply.
        H_block, _, _ = build_H_block(reps, irrep_data, d)
        H_block = 0.5 * (H_block + H_block.conj().T)

        X = rng.standard_normal((n_basis, B_block)) + \
            1j * rng.standard_normal((n_basis, B_block))
        Y_sparse = H_block @ X
        Y_spmv = spmv_pg_Ih(reps, V_per, eig_per, n_per, irrep_data, d, X)

        nrm = np.linalg.norm(Y_sparse)
        if nrm < 1e-15:
            err_rel = np.linalg.norm(Y_sparse - Y_spmv)
        else:
            err_rel = np.linalg.norm(Y_sparse - Y_spmv) / nrm
        ok = err_rel < tol_rel
        ok_all = ok_all and ok
        print(f"    {name:5s} (d={d}): n_basis = {n_basis:3d}, "
              f"||Y_sparse - Y_spmv||_rel = {err_rel:.2e}  "
              f"[{'OK' if ok else 'FAIL'}]")
    return ok_all


if __name__ == "__main__":
    print("Verify_spmv_pg_Ih: matrix-free I_h SpMV vs sparse H_block\n")
    all_ok = True
    for M in [0, 1, 2, 3]:
        if not verify_M(M):
            all_ok = False
        print()
    print("=" * 60)
    print(f"OVERALL: {'PASS' if all_ok else 'FAIL'}")
