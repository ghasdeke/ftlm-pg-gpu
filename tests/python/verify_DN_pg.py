"""
Milestone F verification: D_N (dihedral) basis restriction for real-momentum
PG-FTLM sectors of a Heisenberg ring.

The reflection sigma acts on a state n with N base-d_loc digits by
reversing the digit order. As a within-sector Z_2 symmetry sigma is only
meaningful at the real momenta p = 0 and p = N/2 (even N); at other p it
maps the sector to its (-k)-partner already handled by merge_kbar.

For each (M, p in {0, N/2}) we verify:

  1. The C_N rep basis decomposes under sigma into super-reps:
       - cross-paired: r and sigma_R(r) form a 2D pair; only the smaller
         is kept as a super-rep
       - self-stable: sigma_R(r) = r; the rep has sigma-eigenvalue
         lambda = exp(-i k h_sigma(r)) = +/-1; lives in sigma_par = lambda.
     The dim counts D_plus + D_minus = D_full match.

  2. The H matrix in the sigma-super-rep basis has matrix elements
        H[r', r; sigma_par] = <r'|H|r> + sigma_par * <r'|H|sigma_R(r)>
     for cross-paired r' and r (at p=0; analogous at p=N/2 with a phase
     factor exp(-i*k*h_sigma) per transition).
     Built explicitly and diagonalized, gives the sigma_par-subspace
     spectrum.

  3. Concatenating sigma_+ and sigma_- spectra reproduces the full
     C_N-only spectrum at the same (M, p).

If all three checks pass, the kernel/sparse-builder math is right and
we proceed to MATLAB and CUDA.
"""

import numpy as np
import scipy.sparse as sp

from verify_milestone_A import (
    enumerate_sector_with_translation,
    build_H_pg_sector,
    min_image_ring,
    digits_of,
)


# ----------------------------------------------------------------
# Sigma action on the state-integer encoding
# ----------------------------------------------------------------

def apply_sigma_ring(n, d_loc, N):
    """Digit-reverse n: a_k -> a_{N-1-k}."""
    digs = digits_of(n, d_loc, N)
    new_digs = list(reversed(digs))
    out = 0
    for k in range(N):
        out += new_digs[k] * (d_loc ** k)
    return out


def sigma_action_per_rep(reps, N, s_val, p_irrep):
    """For each rep r, compute sigma_R(r), h_sigma(r), and the
    sigma-eigenvalue if self-stable. Returns (sigma_R_idx, h_sigma_arr,
    lambda_arr, is_self_stable, sigma_partner_state)."""
    d_loc   = round(2*s_val + 1)
    k_phase = 2 * np.pi * p_irrep / N
    dim = len(reps)
    rep_to_idx = {int(r): i for i, r in enumerate(reps)}

    sigma_R_idx       = np.zeros(dim, dtype=np.int64)
    h_sigma_arr       = np.zeros(dim, dtype=np.int32)
    lambda_arr        = np.zeros(dim, dtype=np.complex128)
    is_self_stable    = np.zeros(dim, dtype=bool)
    sigma_partner     = np.zeros(dim, dtype=np.int64)

    for t, r in enumerate(reps):
        s_state = apply_sigma_ring(int(r), d_loc, N)
        s_rep, h_s, _ = min_image_ring(s_state, d_loc, N)
        # sigma|r, p> has phase exp(-i k h_s) and ends up in s_rep
        sigma_partner[t] = s_rep
        if int(s_rep) in rep_to_idx:
            sigma_R_idx[t] = rep_to_idx[int(s_rep)]
        else:
            sigma_R_idx[t] = -1   # shouldn't happen for closed sectors
        h_sigma_arr[t]   = h_s
        lambda_arr[t]    = np.exp(-1j * k_phase * h_s)
        is_self_stable[t] = (sigma_R_idx[t] == t)

    return sigma_R_idx, h_sigma_arr, lambda_arr, is_self_stable, sigma_partner


# ----------------------------------------------------------------
# Super-rep enumeration for sigma_par
# ----------------------------------------------------------------

def enumerate_DN(N, s_val, M_target, p_irrep, sigma_par):
    """Return (super_reps, orbit_lens, type_arr) where:
       super_reps : column of state integers in the sigma_par super-rep basis
       orbit_lens : C_N orbit length per super-rep (NOT D_N orbit length)
       type_arr   : 1 for self-stable, 0 for cross-paired
    """
    reps_C, L_C, dim_C = enumerate_sector_with_translation(N, s_val, M_target, p_irrep)
    if dim_C == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    sR_idx, h_s, lam, is_self, sigma_partner = sigma_action_per_rep(
        reps_C, N, s_val, p_irrep)

    # Super-rep: r is a super-rep if reps_C[r] < sigma_partner[r] OR
    # (reps_C[r] == sigma_partner[r], i.e., self-stable with right sigma_par).
    super_mask = np.zeros(dim_C, dtype=bool)
    type_arr_full = np.zeros(dim_C, dtype=np.int32)
    for t in range(dim_C):
        if is_self[t]:
            # self-stable; lambda must equal sigma_par (only real lambda allowed)
            lam_real = np.real(lam[t])
            # lam should be +/-1 in exact arithmetic
            if abs(lam_real - sigma_par) < 1e-8:
                super_mask[t] = True
                type_arr_full[t] = 1  # self-stable
        else:
            # cross-paired: keep r if r < sigma_partner (i.e., r is the smaller of the pair)
            if int(reps_C[t]) <= int(sigma_partner[t]):
                super_mask[t] = True
                type_arr_full[t] = 0  # cross-paired

    super_reps = reps_C[super_mask]
    orbit_lens = L_C[super_mask]
    type_arr   = type_arr_full[super_mask]
    return super_reps, orbit_lens, type_arr


# ----------------------------------------------------------------
# Sparse H in the sigma_par super-rep basis
# ----------------------------------------------------------------

def build_H_DN_sparse(N, s_val, J, super_reps, orbit_lens, type_arr,
                       sigma_par, p_irrep):
    """Build H in the (M, p, sigma_par) super-rep basis using the
    formula H[r', r] = <r'|H|r>_{C_N} + sigma_par * <r'|H|sigma_R(r)>_{C_N}
    with the sqrt(2) factors when self-stable/cross-paired mix.
    """
    d_loc = round(2 * s_val + 1)
    n_total = d_loc ** N
    bonds = ring_bonds(N)
    dim_sup = len(super_reps)
    if dim_sup == 0:
        return sp.csr_matrix((0, 0))

    # First we need the full C_N rep basis at (M, p) to build the underlying
    # H_C and the sigma map; then we transform to super-rep basis.
    M_target = 0  # will be inferred from total digit sum of any super_rep
    # Infer M_target from first super-rep
    if len(super_reps) > 0:
        m_first = sum(digits_of(int(super_reps[0]), d_loc, N)) - N * s_val
        M_target = int(round(m_first))
    reps_C, L_C, _ = enumerate_sector_with_translation(N, s_val, M_target, p_irrep)
    H_C = build_H_pg_sector(N, s_val, J, reps_C, L_C, p_irrep).toarray()

    # sigma map for C_N reps
    sR_idx, h_s, lam, is_self, _ = sigma_action_per_rep(reps_C, N, s_val, p_irrep)

    # Map from C_N rep index to (super_rep_idx_in_super_basis, sigma_sign)
    # super_rep_idx is the index in `super_reps`; sigma_sign is +1 if this
    # C_N rep IS the super-rep, sigma_par if it's the sigma-partner of one.
    super_rep_state_to_super_idx = {int(s): i for i, s in enumerate(super_reps)}
    map_to_super_idx = np.full(len(reps_C), -1, dtype=np.int64)
    map_sigma_sign   = np.zeros(len(reps_C), dtype=np.complex128)
    for t, rC in enumerate(reps_C):
        rC_int = int(rC)
        if rC_int in super_rep_state_to_super_idx:
            # rC IS the super-rep itself
            map_to_super_idx[t] = super_rep_state_to_super_idx[rC_int]
            map_sigma_sign[t]   = +1.0
        else:
            # rC's sigma-partner is the super-rep
            partner_idx = sR_idx[t]
            if partner_idx >= 0 and int(reps_C[partner_idx]) in super_rep_state_to_super_idx:
                map_to_super_idx[t] = super_rep_state_to_super_idx[int(reps_C[partner_idx])]
                # sigma maps rC to (phase * partner). In the SpMV gather
                # we accumulate sigma_par-weighted contribution.
                map_sigma_sign[t]   = sigma_par * 1.0
            # If neither rC nor its partner is in super-rep basis: this
            # means the C_N rep is "excluded" (e.g. self-stable with wrong
            # lambda); contributions to/from it are 0 in this sigma_par
            # sector, so we leave map_to_super_idx[t] = -1.

    # Norm factor per super-rep: 1/sqrt(2) for cross-paired, 1 for self-stable.
    # The formula H[r', r; sigma_par] = (1/(N_r' N_r)) * sum_4_terms reduces to
    # what we derived: <r'|H|r> + sigma_par <r'|H|sigma_R(r)> for both cross,
    # sqrt(2) <r'|H|r> for cross/self mix, <r'|H|r> for self/self.
    # Equivalent expression: factor = (norm_r'_super / norm_r_super) where
    # norm = 1 for self-stable and 1/sqrt(2) for cross-paired, after
    # combining with the 2-term sum. We implement directly via the matrix
    # element formula.

    norm_super = np.ones(dim_sup)
    norm_super[type_arr == 0] = 1.0 / np.sqrt(2.0)   # cross-paired
    # (self-stable kept norm = 1)

    # Build H_DN dense:
    H_DN = np.zeros((dim_sup, dim_sup), dtype=H_C.dtype)
    # H_DN[I, J] = sum over (i, j) such that map_to_super_idx[i]=I, map_to_super_idx[j]=J
    # of (sigma_sign[i] * sigma_sign[j]) * H_C[i, j] * scaling.
    # But "scaling" needs care. Let's instead use the explicit super-rep
    # construction:
    #   |I, sigma_par> = norm_super[I] * (|reps_C[i_I]> + sigma_par * |reps_C[partner_I]>)
    # for cross-paired I, where i_I is the index in reps_C of super_reps[I]
    # and partner_I is the index of its sigma-partner; self-stable has only
    # the first term.
    # Then H_DN[I, J] = <I, sigma_par | H | J, sigma_par>.

    # Index of each super-rep in reps_C
    reps_C_state_to_idx = {int(s): i for i, s in enumerate(reps_C)}
    i_in_C = np.array([reps_C_state_to_idx[int(s)] for s in super_reps], dtype=np.int64)

    # Phase factor lambda_sigma(r) = exp(-i*k*h_sigma(r)).
    # For p in {0, N/2} it is real (+1 or -1); we extract the real part.
    lam_real = np.real(lam)

    for I in range(dim_sup):
        if type_arr[I] == 1:
            # self-stable I: |I> = |reps_C[i_I]>
            j_I_list = [i_in_C[I]]
            c_I_list = [1.0]
            n_I = 1.0
        else:
            partner_I = sR_idx[i_in_C[I]]
            j_I_list = [i_in_C[I], partner_I]
            # Cross-paired eigenvector includes the sigma-phase factor:
            #   |I, sigma_par> ~ |r> + sigma_par * lambda_sigma(r) * |sigma_R(r)>
            c_I_list = [1.0, sigma_par * lam_real[i_in_C[I]]]
            n_I = 1.0 / np.sqrt(2.0)
        for J in range(dim_sup):
            if type_arr[J] == 1:
                j_J_list = [i_in_C[J]]
                c_J_list = [1.0]
                n_J = 1.0
            else:
                partner_J = sR_idx[i_in_C[J]]
                j_J_list = [i_in_C[J], partner_J]
                c_J_list = [1.0, sigma_par * lam_real[i_in_C[J]]]
                n_J = 1.0 / np.sqrt(2.0)
            acc = 0.0
            for ii_loc, ii in enumerate(j_I_list):
                for jj_loc, jj in enumerate(j_J_list):
                    acc += np.conj(c_I_list[ii_loc]) * c_J_list[jj_loc] * H_C[ii, jj]
            H_DN[I, J] = n_I * n_J * acc

    return sp.csr_matrix(H_DN)


def ring_bonds(N):
    return [(i, (i + 1) % N) for i in range(N)]


# ----------------------------------------------------------------
# Tests
# ----------------------------------------------------------------

def test_one(N, s_val, J=1.0):
    print(f"=== N={N}, s={s_val}, J={J} ===")
    overall = True
    M_max = round(N * s_val)
    for M in range(0, M_max + 1):
        for p in [0, N // 2] if (N % 2 == 0) else [0]:
            if p >= N:
                continue
            reps, Ls, dim = enumerate_sector_with_translation(N, s_val, M, p)
            if dim == 0:
                continue

            H_C = build_H_pg_sector(N, s_val, J, reps, Ls, p).toarray()
            H_C = 0.5 * (H_C + H_C.conj().T)
            E_C = np.sort(np.real(np.linalg.eigvalsh(H_C)))

            E_agg = []
            for sigma_par in [+1, -1]:
                super_reps, orbit_lens, type_arr = enumerate_DN(
                    N, s_val, M, p, sigma_par)
                if len(super_reps) == 0:
                    continue
                H_DN = build_H_DN_sparse(N, s_val, J, super_reps,
                                          orbit_lens, type_arr, sigma_par, p).toarray()
                H_DN = 0.5 * (H_DN + H_DN.conj().T)
                E_sigma = np.real(np.linalg.eigvalsh(H_DN))
                E_agg.append(E_sigma)
            E_agg = np.sort(np.concatenate(E_agg)) if E_agg else np.array([])

            if len(E_agg) != len(E_C):
                print(f"  M={M:+d} p={p:2d}: DIM MISMATCH C={len(E_C)} agg={len(E_agg)}  FAIL")
                overall = False
                continue
            diff = np.max(np.abs(E_agg - E_C))
            status = "OK " if diff < 1e-9 else "FAIL"
            print(f"  M={M:+d} p={p:2d} dim_C={dim}, agg max|dE|={diff:.2e} [{status}]")
            if diff > 1e-9:
                overall = False
    print(f"=> {'PASS' if overall else 'FAIL'}\n")
    return overall


if __name__ == "__main__":
    print("Milestone F verification: D_N sigma-restriction at p in {0, N/2}\n")
    ok = True
    ok &= test_one(N=4,  s_val=0.5)
    ok &= test_one(N=6,  s_val=0.5)
    ok &= test_one(N=8,  s_val=0.5)
    ok &= test_one(N=4,  s_val=1.0)
    ok &= test_one(N=10, s_val=0.5)
    ok &= test_one(N=5,  s_val=0.5)   # odd N, only p=0
    print("OVERALL: " + ("ALL PASS" if ok else "SOME FAILED"))
