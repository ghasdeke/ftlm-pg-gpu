"""
Python-Referenz fuer den CLT-basierten Gather-SpMV.

Spiegelt build_clt_pg_Ih.m und spmv_pg_Ih_clt_matlab.m, und verifiziert
den Gather-Pfad gegen die matrixfreie Scatter-SpMV (verify_spmv_pg_Ih.py),
die wiederum bereits bit-exakt gegen den expliziten sparse H_block ist.

Damit ist die CLT-Datenstruktur, die der CUDA-Kernel konsumieren wird,
in Python festgenagelt.
"""

import numpy as np

from verify_Ih_full import IH_T1g, IH_T1u, IH_T2g, IH_T2u, \
    IH_Fg, IH_Fu, IH_Hg, IH_Hu, IH_Ag, IH_Au

from verify_Ih_beta import (
    PERMS, apply_perm_to_state, BONDS,
    enumerate_M_sector,
    min_image_Ih, digits_of,
    D_LOC, N_SITES, S_VAL,
)

from verify_Ih_gamma2 import (
    rho, stabilizer_matrix, column_basis,
)

from verify_spmv_pg_Ih import enumerate_with_gamma2, spmv_pg_Ih


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


def build_clt(super_reps, V_per_rep, eig_per_rep, n_per_rep,
              irrep_data, d_irrep, J=1.0):
    """Precompute the gather CLT."""
    s_val = 0.5
    d_loc = 2
    N = 12
    powers = [d_loc ** k for k in range(N)]
    n_reps = len(super_reps)

    rep_offsets = [0]
    for n in n_per_rep[:-1]:
        rep_offsets.append(rep_offsets[-1] + n)
    n_basis = sum(n_per_rep)
    state_to_rep = {r: i for i, r in enumerate(super_reps)}
    sqrt_eig = [np.sqrt(e) for e in eig_per_rep]

    # Per-rep diagonal
    diag_vals = []
    for r in super_reps:
        dg = digits_of(r)
        m = [dg[k] - s_val for k in range(N)]
        diag_vals.append(sum(J * m[a] * m[b] for (a, b) in BONDS))

    # First pass: per-output-rep entry lists
    entries = [[] for _ in range(n_reps)]   # entries[t] = list of (src, M_e)

    for i, r in enumerate(super_reps):
        V_r = V_per_rep[i]
        sqrt_eig_r = sqrt_eig[i]
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
                t = state_to_rep.get(rep_a)
                if t is None or n_per_rep[t] == 0:
                    continue
                V_t = V_per_rep[t]
                sqrt_eig_t = sqrt_eig[t]
                rho_T = np.conj(rho(irrep_data, g_min))         # d x d
                inner = V_t.conj().T @ rho_T @ V_r              # n_t x n_i
                M_e = c_a * inner * (sqrt_eig_t[:, None] / sqrt_eig_r[None, :])
                entries[t].append((i, M_e))

    clt = {
        'n_basis': n_basis,
        'n_reps': n_reps,
        'd_irrep': d_irrep,
        'rep_offsets': rep_offsets,
        'n_per_rep': n_per_rep,
        'diag_vals': diag_vals,
        'entries': entries,
    }
    return clt


def spmv_clt(clt, X):
    """Gather-style SpMV from the CLT."""
    n_basis = clt['n_basis']
    assert X.shape[0] == n_basis
    Y = np.zeros_like(X, dtype=complex)
    for t in range(clt['n_reps']):
        off_t = clt['rep_offsets'][t]
        n_t = clt['n_per_rep'][t]
        # Diagonal
        Y[off_t:off_t+n_t, :] += clt['diag_vals'][t] * X[off_t:off_t+n_t, :]
        # Gather
        for (src, M_e) in clt['entries'][t]:
            off_s = clt['rep_offsets'][src]
            n_s = clt['n_per_rep'][src]
            Y[off_t:off_t+n_t, :] += M_e @ X[off_s:off_s+n_s, :]
    return Y


def verify_M(M_target, B_block=4, tol_rel=1e-12, seed=27182):
    print(f"--- M = {M_target} ---")
    rng = np.random.default_rng(seed + M_target)
    ok_all = True
    for name, d, irrep_data in IRREPS:
        reps, V_per, eig_per, n_per = enumerate_with_gamma2(M_target, irrep_data, d)
        n_basis = sum(n_per)
        if n_basis == 0:
            print(f"    {name:5s} (d={d}): empty")
            continue

        clt = build_clt(reps, V_per, eig_per, n_per, irrep_data, d)
        n_entries = sum(len(e) for e in clt['entries'])

        X = rng.standard_normal((n_basis, B_block)) + \
            1j * rng.standard_normal((n_basis, B_block))

        Y_mf = spmv_pg_Ih(reps, V_per, eig_per, n_per, irrep_data, d, X)
        Y_clt = spmv_clt(clt, X)

        nrm = np.linalg.norm(Y_mf)
        if nrm < 1e-15:
            err_rel = np.linalg.norm(Y_mf - Y_clt)
        else:
            err_rel = np.linalg.norm(Y_mf - Y_clt) / nrm
        ok = err_rel < tol_rel
        ok_all = ok_all and ok
        print(f"    {name:5s} (d={d}): n_basis = {n_basis:3d}, "
              f"n_clt_entries = {n_entries:4d}, "
              f"||Y_mf - Y_clt||_rel = {err_rel:.2e}  "
              f"[{'OK' if ok else 'FAIL'}]")
    return ok_all


if __name__ == "__main__":
    print("Verify_clt_pg_Ih: gather (CLT) SpMV vs matrix-free SpMV\n")
    all_ok = True
    for M in [0, 1, 2, 3]:
        if not verify_M(M):
            all_ok = False
        print()
    print("=" * 60)
    print(f"OVERALL: {'PASS' if all_ok else 'FAIL'}")
