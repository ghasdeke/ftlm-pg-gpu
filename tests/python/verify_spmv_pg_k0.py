"""
Milestone B verification: per-thread SpMV reference for the k=0
PG-FTLM GPU kernel.

The CUDA kernel will do the following per output thread t:
    1. state = basis[t] = reps[t]              # rep at row t
    2. L_r   = orbit_lens[t]
    3. decompose state into N digits (base d_loc)
    4. result[b] = J * sum_<ij> mi*mj * V[t*B+b]   # diagonal (unchanged)
    5. for each bond <i,j>:
         for each of {S+_i S-_j, S-_i S+_j}:
             state_a = state with appropriate digit flip
             rep_a   = min_image_C_N(state_a)      # NEW
             idx_a   = CLT_lookup(rep_a)
             if idx_a valid:
                 L_a   = orbit_lens[idx_a]         # NEW
                 norm  = sqrt(L_r / L_a)           # NEW
                 result[b] += c_a * norm * V[idx_a*B+b]
    6. W[t*B+b] = result[b]

We verify this thread-level logic in Python against the reference sparse
matrix H_pg from Milestone A. They must agree to machine precision for
arbitrary input vectors v in the rep basis.

This validates the SpMV kernel logic *before* we commit it to CUDA. The
CUDA file will then be a literal transcription of this logic in C with
all the existing block-Lanczos infrastructure unchanged.
"""

import numpy as np
from verify_milestone_A import (
    enumerate_sector_with_translation,
    build_H_pg_sector,
    min_image_ring,
    cyclic_shift,
    digits_of,
)


def spmv_pg_k0_threadlike(reps, orbit_lens, bonds, s_val, J, N, V):
    """
    Per-thread SpMV in the rep basis for k=0.

    V can be shape (dim,) or (dim, B). Returns W of the same shape.

    Faithfully mirrors what each GPU thread will compute:
       - digits of basis[t] = reps[t]
       - diagonal accumulation
       - off-diagonal: state_a -> min_image -> CLT lookup ->
         sqrt(L_r/L_a) factor -> accumulate
    """
    d_loc = round(2 * s_val + 1)
    dim = len(reps)
    powers = [d_loc ** k for k in range(N)]

    # rep -> idx_a (CLT-like lookup, here a simple dict)
    rep_to_idx = {int(r): i for i, r in enumerate(reps)}

    if V.ndim == 1:
        W = np.zeros_like(V)
    else:
        W = np.zeros_like(V)

    for t in range(dim):
        state = int(reps[t])
        L_r = int(orbit_lens[t])
        sqrt_L_r = np.sqrt(L_r)
        dg = digits_of(state, d_loc, N)
        m = [dg[k] - s_val for k in range(N)]

        # Diagonal
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
                rep_a, _h, _L = min_image_ring(state_a, d_loc, N)
                if rep_a in rep_to_idx:
                    idx_a = rep_to_idx[rep_a]
                    L_a = orbit_lens[idx_a]
                    norm = sqrt_L_r / np.sqrt(L_a)
                    total = coeff * norm
                    if V.ndim == 1:
                        W[t] += total * V[idx_a]
                    else:
                        W[t, :] += total * V[idx_a, :]

            # S-_a S+_b
            if dg[a_idx] > 0 and dg[b_idx] < d_loc - 1:
                coeff = 0.5 * J * \
                    np.sqrt(s_val * (s_val + 1) - ma * (ma - 1)) * \
                    np.sqrt(s_val * (s_val + 1) - mb * (mb + 1))
                state_a = state - powers[a_idx] + powers[b_idx]
                rep_a, _h, _L = min_image_ring(state_a, d_loc, N)
                if rep_a in rep_to_idx:
                    idx_a = rep_to_idx[rep_a]
                    L_a = orbit_lens[idx_a]
                    norm = sqrt_L_r / np.sqrt(L_a)
                    total = coeff * norm
                    if V.ndim == 1:
                        W[t] += total * V[idx_a]
                    else:
                        W[t, :] += total * V[idx_a, :]

    return W


def ring_bonds(N):
    return [(i, (i + 1) % N) for i in range(N)]


def run_case(N, s_val, J):
    print(f"=== N={N}, s={s_val}, J={J} (k=0 SpMV verification) ===")
    M_max = round(N * s_val)
    bonds = ring_bonds(N)
    overall_pass = True

    for M in range(0, M_max + 1):
        reps, Ls, dim = enumerate_sector_with_translation(N, s_val, M, p_irrep=0)
        if dim == 0:
            continue

        # Reference: explicit sparse H_pg, verified in Milestone A.
        H_ref = build_H_pg_sector(N, s_val, J, reps, Ls, p_irrep=0).toarray()

        # Random test vectors (single and block).
        rng = np.random.default_rng(42 + M)
        v_single = rng.standard_normal(dim)
        V_block = rng.standard_normal((dim, 4))

        # Reference SpMV results.
        w_ref_single = H_ref @ v_single
        W_ref_block = H_ref @ V_block

        # Thread-like SpMV.
        w_thread_single = spmv_pg_k0_threadlike(
            reps, Ls, bonds, s_val, J, N, v_single)
        W_thread_block = spmv_pg_k0_threadlike(
            reps, Ls, bonds, s_val, J, N, V_block)

        err1 = np.max(np.abs(w_thread_single - w_ref_single))
        err2 = np.max(np.abs(W_thread_block - W_ref_block))
        tol = 1e-10 * max(np.max(np.abs(w_ref_single)), 1.0)

        status1 = "OK " if err1 < tol else "FAIL"
        status2 = "OK " if err2 < tol else "FAIL"
        print(f"  M=+{M}: dim={dim}, max|dw_single|={err1:.2e} [{status1}], "
              f"max|dW_block|={err2:.2e} [{status2}]")
        if err1 > tol or err2 > tol:
            overall_pass = False

    print(f"=> {'PASS' if overall_pass else 'FAIL'}\n")
    return overall_pass


if __name__ == "__main__":
    print("Milestone B SpMV verification (k=0, per-thread logic vs sparse H_pg)\n")
    ok = True
    ok &= run_case(N=4, s_val=0.5, J=1.0)
    ok &= run_case(N=6, s_val=0.5, J=1.0)
    ok &= run_case(N=8, s_val=0.5, J=1.0)
    ok &= run_case(N=4, s_val=1.0, J=1.0)
    ok &= run_case(N=10, s_val=0.5, J=1.0)
    print("=" * 60)
    print(f"OVERALL: {'ALL PASS' if ok else 'SOME FAILED'}")
