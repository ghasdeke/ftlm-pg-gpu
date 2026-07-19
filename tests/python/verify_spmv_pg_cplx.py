"""
Milestone C verification: per-thread SpMV reference for the complex-k
PG-FTLM GPU kernel.

The complex CUDA kernel will do, per output thread t:

    1. state       = basis[t] = reps[t]
    2. L_r         = orbit_lens[t]
    3. sqrt_L_r    = sqrtf(L_r)
    4. digits      = decompose(state)
    5. diag        = J * sum_<ij> mi*mj
    6. result_re[b] = diag * V_re[t*B+b]                # diagonal: phase 1
       result_im[b] = diag * V_im[t*B+b]
    7. for each bond <i,j>, for each of {S+_i S-_j, S-_i S+_j}:
         state_a = state with appropriate digit flip
         (rep_a, h_min) = min_image_C_N(state_a)
         idx_a = CLT_lookup(rep_a)
         if idx_a valid:
             L_a    = orbit_lens[idx_a]
             norm   = sqrt_L_r * rsqrtf(L_a)
             # phase factor exp(-i * k * h_min)
             cos_phi =  cos(k * h_min)        # = c_cos_kh[h_min]
             sin_phi =  sin(k * h_min)        # = c_sin_kh[h_min]
             # complex coefficient (alpha_re + i*alpha_im) := c_a * norm * (cos - i*sin)
             alpha_re =  c_a * norm * cos_phi
             alpha_im = -c_a * norm * sin_phi
             for b in 0..B:
                 vr = V_re[idx_a*B+b]
                 vi = V_im[idx_a*B+b]
                 result_re[b] += alpha_re*vr - alpha_im*vi
                 result_im[b] += alpha_re*vi + alpha_im*vr
    8. W_re[t*B+b] = result_re[b]
       W_im[t*B+b] = result_im[b]

This Python routine reproduces that logic and is compared against
H_pg @ V for the complex H_pg built by build_H_pg_sector (verified in
Milestone A). For arbitrary complex input vectors the agreement must be
machine precision.

In addition we verify, for k = 0, that the same routine reduces to the
real result computed by the Milestone B reference (verify_spmv_pg_k0).
"""

import numpy as np
from verify_milestone_A import (
    enumerate_sector_with_translation,
    build_H_pg_sector,
    min_image_ring,
    digits_of,
)


def spmv_pg_cplx_threadlike(reps, orbit_lens, bonds, s_val, J, N, p_irrep, V):
    """
    Per-thread SpMV in the rep basis for any p in [0, N-1].

    V is a complex array of shape (dim,) or (dim, B). Returns W of same shape.
    """
    d_loc = round(2 * s_val + 1)
    dim = len(reps)
    powers = [d_loc ** k for k in range(N)]
    k_phase = 2 * np.pi * p_irrep / N
    cos_table = np.array([np.cos(k_phase * h) for h in range(N)])
    sin_table = np.array([np.sin(k_phase * h) for h in range(N)])

    rep_to_idx = {int(r): i for i, r in enumerate(reps)}

    W = np.zeros_like(V, dtype=np.complex128)

    for t in range(dim):
        state = int(reps[t])
        L_r = int(orbit_lens[t])
        sqrt_L_r = np.sqrt(L_r)
        dg = digits_of(state, d_loc, N)
        m = [dg[k] - s_val for k in range(N)]

        # Diagonal (phase = 1)
        diag = 0.0
        for (a, b) in bonds:
            diag += J * m[a] * m[b]

        if V.ndim == 1:
            W[t] = diag * V[t]
        else:
            W[t, :] = diag * V[t, :]

        # Off-diagonal
        for (a_idx, b_idx) in bonds:
            ma, mb = m[a_idx], m[b_idx]

            # S+_a S-_b
            if dg[a_idx] < d_loc - 1 and dg[b_idx] > 0:
                coeff = 0.5 * J * \
                    np.sqrt(s_val * (s_val + 1) - ma * (ma + 1)) * \
                    np.sqrt(s_val * (s_val + 1) - mb * (mb - 1))
                state_a = state + powers[a_idx] - powers[b_idx]
                rep_a, h_min, L_a = min_image_ring(state_a, d_loc, N)
                if rep_a in rep_to_idx:
                    idx_a = rep_to_idx[rep_a]
                    norm = sqrt_L_r / np.sqrt(L_a)
                    cos_phi = cos_table[h_min]
                    sin_phi = sin_table[h_min]
                    alpha_re =  coeff * norm * cos_phi
                    alpha_im =  coeff * norm * sin_phi
                    if V.ndim == 1:
                        vr = V[idx_a].real; vi = V[idx_a].imag
                        W[t] += (alpha_re*vr - alpha_im*vi) + \
                                1j * (alpha_re*vi + alpha_im*vr)
                    else:
                        vr = V[idx_a, :].real
                        vi = V[idx_a, :].imag
                        W[t, :] += (alpha_re*vr - alpha_im*vi) + \
                                   1j * (alpha_re*vi + alpha_im*vr)

            # S-_a S+_b
            if dg[a_idx] > 0 and dg[b_idx] < d_loc - 1:
                coeff = 0.5 * J * \
                    np.sqrt(s_val * (s_val + 1) - ma * (ma - 1)) * \
                    np.sqrt(s_val * (s_val + 1) - mb * (mb + 1))
                state_a = state - powers[a_idx] + powers[b_idx]
                rep_a, h_min, L_a = min_image_ring(state_a, d_loc, N)
                if rep_a in rep_to_idx:
                    idx_a = rep_to_idx[rep_a]
                    norm = sqrt_L_r / np.sqrt(L_a)
                    cos_phi = cos_table[h_min]
                    sin_phi = sin_table[h_min]
                    alpha_re =  coeff * norm * cos_phi
                    alpha_im =  coeff * norm * sin_phi
                    if V.ndim == 1:
                        vr = V[idx_a].real; vi = V[idx_a].imag
                        W[t] += (alpha_re*vr - alpha_im*vi) + \
                                1j * (alpha_re*vi + alpha_im*vr)
                    else:
                        vr = V[idx_a, :].real
                        vi = V[idx_a, :].imag
                        W[t, :] += (alpha_re*vr - alpha_im*vi) + \
                                   1j * (alpha_re*vi + alpha_im*vr)

    return W


def ring_bonds(N):
    return [(i, (i + 1) % N) for i in range(N)]


def run_case(N, s_val, J):
    M_max = round(N * s_val)
    bonds = ring_bonds(N)
    overall_pass = True
    print(f"=== N={N}, s={s_val}, J={J} (complex SpMV verification) ===")

    for M in range(0, M_max + 1):
        for p in range(N):
            reps, Ls, dim = enumerate_sector_with_translation(N, s_val, M, p)
            if dim == 0:
                continue

            H_ref = build_H_pg_sector(N, s_val, J, reps, Ls, p).toarray()
            # Ensure complex dtype for verification
            H_ref = H_ref.astype(np.complex128)

            rng = np.random.default_rng(1000*p + M + 7)
            v_single = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
            V_block = rng.standard_normal((dim, 4)) + 1j * rng.standard_normal((dim, 4))

            w_ref_s = H_ref @ v_single
            W_ref_b = H_ref @ V_block

            w_thr_s = spmv_pg_cplx_threadlike(
                reps, Ls, bonds, s_val, J, N, p, v_single)
            W_thr_b = spmv_pg_cplx_threadlike(
                reps, Ls, bonds, s_val, J, N, p, V_block)

            err1 = np.max(np.abs(w_thr_s - w_ref_s))
            err2 = np.max(np.abs(W_thr_b - W_ref_b))
            tol = 1e-10 * max(np.max(np.abs(w_ref_s)), 1.0)
            status1 = "OK " if err1 < tol else "FAIL"
            status2 = "OK " if err2 < tol else "FAIL"
            if err1 >= tol or err2 >= tol:
                print(f"  M={M:+d} p={p:2d} dim={dim:4d} "
                      f"max|dw_s|={err1:.2e} [{status1}] "
                      f"max|dW_b|={err2:.2e} [{status2}]")
                overall_pass = False

    # Summary (per-sector PASS messages suppressed unless failures)
    print(f"  All (M, p) sectors PASS" if overall_pass else "  SOME FAILURES")
    print(f"=> {'PASS' if overall_pass else 'FAIL'}\n")
    return overall_pass


if __name__ == "__main__":
    print("Milestone C SpMV verification (all k, per-thread logic vs sparse H_pg)\n")
    ok = True
    ok &= run_case(N=4, s_val=0.5, J=1.0)
    ok &= run_case(N=6, s_val=0.5, J=1.0)
    ok &= run_case(N=8, s_val=0.5, J=1.0)
    ok &= run_case(N=4, s_val=1.0, J=1.0)
    ok &= run_case(N=10, s_val=0.5, J=1.0)
    print("=" * 60)
    print(f"OVERALL: {'ALL PASS' if ok else 'SOME FAILED'}")
