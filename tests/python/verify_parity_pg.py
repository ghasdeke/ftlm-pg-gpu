"""
Milestone E verification: spin-parity reduction of (M=0, p) PG-FTLM sectors.

The spin-inversion operator P maps |m_1, ..., m_N> to |-m_1, ..., -m_N>,
acting on the integer state encoding as

    P(n) = d_loc^N - 1 - n.

This is verifiable digit-by-digit: a_k -> d_loc - 1 - a_k, hence
P(n) = sum_k (d_loc - 1 - a_k) d_loc^k = (d_loc - 1) (d_loc^N - 1)/(d_loc - 1) - n.

P commutes with translation T and with H, so it acts within each (M=0, k=p)
symmetry-adapted block. In the rep basis, P maps |r, k> to
    P |r, k> = exp(-i k h_min(P r)) | rep(P r), k >
where rep(P r) is the orbit minimum of P(r) and h_min satisfies
T^{h_min}(P r) = rep(P r). This is the per-thread action used here.

For each (M=0, p) sector we verify:
  - P is unitary on the rep basis and squares to identity.
  - P commutes with H_pg.
  - Diagonalizing P yields ±1 eigenvalues with dimensions D_plus and D_minus.
  - Projecting H_pg onto the +/- subspaces and concatenating spectra
    reproduces the full sector spectrum to machine precision.

If all checks pass, parity reduction is exact and an FTLM that samples
random vectors projected to either subspace is unbiased with weight
multiplier (D_sigma / R_eff) per Lanczos chain.
"""

import numpy as np

from verify_milestone_A import (
    enumerate_sector_with_translation,
    build_H_pg_sector,
    min_image_ring,
)


def build_parity_action(reps, orbit_lens, N, s_val, p_irrep):
    """Return (P_idx, P_phase, D_plus, D_minus).

    P_idx[t]   : index in `reps` of the orbit minimum of P(reps[t]).
    P_phase[t] : complex phase such that P |reps[t], p> = P_phase[t] * |reps[P_idx[t]], p>.
    D_plus, D_minus : dimensions of the ±1 parity subspaces.
    """
    d_loc = round(2 * s_val + 1)
    n_total = d_loc ** N
    rep_to_idx = {int(r): i for i, r in enumerate(reps)}
    k_phase = 2 * np.pi * p_irrep / N
    dim = len(reps)

    P_idx   = np.zeros(dim, dtype=np.int64)
    P_phase = np.zeros(dim, dtype=np.complex128)

    trace_P = 0.0 + 0.0j
    for t, r in enumerate(reps):
        Pr = n_total - 1 - int(r)
        rep_P, h_min, _ = min_image_ring(Pr, d_loc, N)
        P_idx[t]   = rep_to_idx[int(rep_P)]
        P_phase[t] = np.exp(-1j * k_phase * h_min)
        if P_idx[t] == t:
            trace_P += P_phase[t]

    # In exact arithmetic trace_P is integer (sum of ±1 contributions).
    D_total = dim
    D_plus  = int(round((D_total + trace_P.real) / 2))
    D_minus = int(round((D_total - trace_P.real) / 2))
    return P_idx, P_phase, D_plus, D_minus


def build_P_matrix(P_idx, P_phase, dim):
    """Dense form of P_pg on the rep basis."""
    P = np.zeros((dim, dim), dtype=np.complex128)
    for t in range(dim):
        P[P_idx[t], t] = P_phase[t]
    return P


def test_one_case(N, s_val, J=1.0):
    print(f"=== N={N}, s={s_val}, J={J} ===")
    overall = True
    for p in range(N):
        reps, Ls, dim = enumerate_sector_with_translation(N, s_val, 0, p)
        if dim == 0:
            continue

        H = build_H_pg_sector(N, s_val, J, reps, Ls, p).toarray()
        H = 0.5 * (H + H.conj().T)
        E_direct = np.sort(np.linalg.eigvalsh(H))

        P_idx, P_phase, D_plus, D_minus = build_parity_action(reps, Ls, N, s_val, p)
        P = build_P_matrix(P_idx, P_phase, dim)

        # 1) Unitarity / involution
        err_unit  = np.max(np.abs(P @ P.conj().T - np.eye(dim)))
        err_invol = np.max(np.abs(P @ P - np.eye(dim)))
        # 2) Commutation [P, H]
        err_comm  = np.max(np.abs(P @ H - H @ P))
        # 3) Dimension check
        dim_ok = (D_plus + D_minus) == dim

        # 4) Project H to ± subspaces and concatenate spectra
        E_P, V_P = np.linalg.eigh(P)
        plus_idx  = np.where(np.abs(E_P - 1) < 1e-8)[0]
        minus_idx = np.where(np.abs(E_P + 1) < 1e-8)[0]
        if len(plus_idx) != D_plus or len(minus_idx) != D_minus:
            print(f"  p={p:2d}: dim mismatch  D_plus/expected = {len(plus_idx)}/{D_plus}, "
                  f"D_minus/expected = {len(minus_idx)}/{D_minus}  FAIL")
            overall = False
            continue
        V_plus  = V_P[:, plus_idx]
        V_minus = V_P[:, minus_idx]
        H_plus  = V_plus.conj().T  @ H @ V_plus
        H_minus = V_minus.conj().T @ H @ V_minus
        E_plus  = np.linalg.eigvalsh(0.5 * (H_plus  + H_plus.conj().T))
        E_minus = np.linalg.eigvalsh(0.5 * (H_minus + H_minus.conj().T))
        E_agg = np.sort(np.concatenate([E_plus, E_minus]))
        err_spec = np.max(np.abs(E_agg - E_direct))

        status = (err_unit < 1e-10 and err_invol < 1e-10 and err_comm < 1e-8 and
                  dim_ok and err_spec < 1e-9)
        if not status:
            print(f"  p={p:2d}: dim={dim}, D+={D_plus}, D-={D_minus}, "
                  f"unit={err_unit:.1e}, invol={err_invol:.1e}, "
                  f"[P,H]={err_comm:.1e}, dE={err_spec:.1e}  FAIL")
            overall = False

    print("  All (M=0, p) sectors PASS" if overall else "  SOME FAILED")
    return overall


if __name__ == "__main__":
    print("Milestone E verification: spin-parity reduction on (M=0, p) sectors\n")
    ok = True
    ok &= test_one_case(N=4, s_val=0.5)
    ok &= test_one_case(N=6, s_val=0.5)
    ok &= test_one_case(N=8, s_val=0.5)
    ok &= test_one_case(N=4, s_val=1.0)
    ok &= test_one_case(N=10, s_val=0.5)
    print()
    print(f"OVERALL: {'ALL PASS' if ok else 'SOME FAILED'}")
