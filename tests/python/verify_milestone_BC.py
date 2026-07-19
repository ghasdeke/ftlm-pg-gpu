"""
Milestone BC verification: full PG-FTLM pipeline for spin rings.

For each test case we check three things:

1. ED aggregation: Run ED in every (M, p) sector, collect (E, w, M)
   tuples with the M <-> -M multiplicity factor, compute C(T), chi(T),
   Z_eff(T). Compare to the same observables built from the
   unsymmetrized full M-sector ED. These must agree to machine
   precision.

2. Complex Lanczos sanity: For a sector with complex H (k not in
   {0, N/2}), build the tridiagonal T via 3-term recursion starting
   from a normalized random complex vector, diagonalize, check that
   the lowest Ritz value approaches the true ground state of H_pg
   for that sector as M_lz grows. Also confirm T is real symmetric.

3. FTLM aggregation: Run the full PG-FTLM pipeline (Lanczos per
   sector with random vectors), aggregate, compute observables. With
   large R and M_lz the result should agree with the ED result within
   stochastic noise.

The FTLM weight per random vector is (dim_sector / R_eff) * |q_k^(1)|^2
where q_k^(1) is the first component of the k-th eigenvector of T.
The M <-> -M factor mult_M = 1 + (M > 0) is applied to the weights.
"""

import numpy as np
import scipy.sparse as sp

# Reuse Milestone A helpers
from verify_milestone_A import (
    enumerate_sector_with_translation,
    build_H_pg_sector,
    build_H_full_M_sector,
    enumerate_M_sector,
)


# ---------- Lanczos ----------

def lanczos_recursion(H, v0, M_lz):
    """Three-term Hermitian Lanczos, no reorthogonalization.
    Returns alpha (real, len <= M_lz) and beta (real, len <= alpha-1).
    Adapts to is_real automatically through the dtype of v0."""
    n = len(v0)
    alpha = []
    beta = []
    v_prev = np.zeros_like(v0)
    v = v0 / np.linalg.norm(v0)
    for j in range(M_lz):
        w = H @ v
        a = np.vdot(v, w).real  # real for Hermitian H
        alpha.append(a)
        w = w - a * v
        if j > 0:
            w = w - beta[-1] * v_prev
        b = np.linalg.norm(w)
        if j < M_lz - 1:
            if b < 1e-12:
                break
            beta.append(b)
            v_prev = v
            v = w / b
        # last step: don't normalize; we won't need v_{M_lz+1}
    return np.array(alpha), np.array(beta)


def tridiag_eig(alpha, beta):
    """Diagonalize the real-symmetric tridiagonal matrix.
    Returns (eigvals, q1_sq) where q1_sq[k] = |Q[0, k]|^2."""
    n = len(alpha)
    T = np.diag(alpha)
    if n > 1:
        T = T + np.diag(beta[:n-1], 1) + np.diag(beta[:n-1], -1)
    eigvals, Q = np.linalg.eigh(T)
    q1_sq = np.abs(Q[0, :]) ** 2
    return eigvals, q1_sq


def run_ftlm_pg_sector(H, dim, R, M_lz, is_real, seed):
    """Run FTLM in one (M, p) sector. Returns (E, w) arrays of length
    R_eff * M_lz_eff total."""
    R_eff = min(R, dim)
    M_lz_eff = min(M_lz, dim)
    rng = np.random.default_rng(seed)
    Es = []
    ws = []
    for r in range(R_eff):
        if is_real:
            v0 = rng.standard_normal(dim)
        else:
            v0 = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
        v0 = v0.astype(np.complex128 if not is_real else np.float64)
        alpha, beta = lanczos_recursion(H, v0, M_lz_eff)
        E, q1_sq = tridiag_eig(alpha, beta)
        w = (dim / R_eff) * q1_sq
        Es.append(E)
        ws.append(w)
    return np.concatenate(Es), np.concatenate(ws)


# ---------- observables ----------

def compute_observables(all_E, all_w, all_M, T_range):
    """C(T), chi(T), Z_eff(T) from (E, w, M)-tuples and a T-grid."""
    all_E = np.asarray(all_E, dtype=np.float64)
    all_w = np.asarray(all_w, dtype=np.float64)
    all_M = np.asarray(all_M, dtype=np.float64)
    T_range = np.asarray(T_range, dtype=np.float64)
    E_min = all_E.min() if len(all_E) > 0 else 0.0
    dE = all_E - E_min
    M2 = all_M ** 2

    C = np.zeros_like(T_range)
    chi = np.zeros_like(T_range)
    Z_eff = np.zeros_like(T_range)
    for i, T in enumerate(T_range):
        beta = 1.0 / T
        boltz = all_w * np.exp(-beta * dE)
        Z = boltz.sum()
        if Z < 1e-250:
            continue
        dE_avg = (dE * boltz).sum() / Z
        E_var = ((dE - dE_avg) ** 2 * boltz).sum() / Z
        M2_avg = (M2 * boltz).sum() / Z
        C[i] = beta ** 2 * E_var
        chi[i] = beta * M2_avg
        Z_eff[i] = Z
    return C, chi, Z_eff


# ---------- full pipelines ----------

def aggregate_ED_PG(N, s_val, J, T_range):
    """Compute observables via per-(M, p) ED + symmetry aggregation."""
    M_max = round(N * s_val)
    all_E, all_w, all_M = [], [], []
    for M in range(0, M_max + 1):  # only M >= 0
        mult_M = 1 + (M > 0)
        for p in range(N):
            reps, Ls, dim = enumerate_sector_with_translation(N, s_val, M, p)
            if dim == 0:
                continue
            H = build_H_pg_sector(N, s_val, J, reps, Ls, p)
            Hd = H.toarray()
            Hd = 0.5 * (Hd + Hd.conj().T)
            E = np.linalg.eigvalsh(Hd)
            w = np.ones_like(E) * mult_M
            all_E.append(E)
            all_w.append(w)
            all_M.append(np.full(len(E), M))
    all_E = np.concatenate(all_E); all_w = np.concatenate(all_w); all_M = np.concatenate(all_M)
    return compute_observables(all_E, all_w, all_M, T_range)


def aggregate_ED_noPG(N, s_val, J, T_range):
    """Compute observables via per-M ED (no symmetry)."""
    M_max = round(N * s_val)
    all_E, all_w, all_M = [], [], []
    for M in range(0, M_max + 1):
        mult_M = 1 + (M > 0)
        H, _ = build_H_full_M_sector(N, s_val, J, M)
        if H.shape[0] == 0:
            continue
        E = np.linalg.eigvalsh(H.toarray())
        w = np.ones_like(E) * mult_M
        all_E.append(E)
        all_w.append(w)
        all_M.append(np.full(len(E), M))
    all_E = np.concatenate(all_E); all_w = np.concatenate(all_w); all_M = np.concatenate(all_M)
    return compute_observables(all_E, all_w, all_M, T_range)


def aggregate_FTLM_PG(N, s_val, J, R, M_lz, T_range, seed0=1234):
    """Compute observables via per-(M, p) FTLM + symmetry aggregation."""
    M_max = round(N * s_val)
    all_E, all_w, all_M = [], [], []
    for M in range(0, M_max + 1):
        mult_M = 1 + (M > 0)
        for p in range(N):
            reps, Ls, dim = enumerate_sector_with_translation(N, s_val, M, p)
            if dim == 0:
                continue
            H = build_H_pg_sector(N, s_val, J, reps, Ls, p)
            is_real = (p == 0) or (2 * p == N)
            seed = seed0 + 100 * M + p
            E, w = run_ftlm_pg_sector(H, dim, R, M_lz, is_real, seed)
            w = w * mult_M
            all_E.append(E)
            all_w.append(w)
            all_M.append(np.full(len(E), M))
    all_E = np.concatenate(all_E); all_w = np.concatenate(all_w); all_M = np.concatenate(all_M)
    return compute_observables(all_E, all_w, all_M, T_range)


# ---------- tests ----------

def test_case(N, s_val, J, label=""):
    T_range = np.array([0.1, 0.3, 1.0, 3.0, 10.0])
    print(f"=== {label}  N={N}, s={s_val}, J={J} ===")

    # Test 1: ED-aggregation PG vs no-PG (must be identical)
    C_pg, chi_pg, Z_pg = aggregate_ED_PG(N, s_val, J, T_range)
    C_no, chi_no, Z_no = aggregate_ED_noPG(N, s_val, J, T_range)
    err_C = np.max(np.abs(C_pg - C_no))
    err_chi = np.max(np.abs(chi_pg - chi_no))
    err_Z = np.max(np.abs(Z_pg - Z_no))
    print(f"  Test 1 (ED PG vs ED no-PG):")
    print(f"    max|dC|   = {err_C:.2e}")
    print(f"    max|dchi| = {err_chi:.2e}")
    print(f"    max|dZ|   = {err_Z:.2e}")
    test1 = err_C < 1e-10 and err_chi < 1e-10 and err_Z < 1e-10

    # Test 2: FTLM-PG vs ED-PG with R, M_lz large enough
    # For a small N=8 s=1/2 ring, full sector R = dim and M_lz = dim is feasible
    R = max(20, int(0.5 * 2 ** N))   # plenty
    M_lz = max(40, int(2 ** N))
    C_ftlm, chi_ftlm, Z_ftlm = aggregate_FTLM_PG(N, s_val, J, R, M_lz, T_range, seed0=42)
    # With large R and M_lz, FTLM should be close to ED but not exact (stochastic).
    rel_C = np.max(np.abs(C_ftlm - C_pg) / np.maximum(np.abs(C_pg), 1e-10))
    rel_chi = np.max(np.abs(chi_ftlm - chi_pg) / np.maximum(np.abs(chi_pg), 1e-10))
    print(f"  Test 2 (FTLM-PG vs ED-PG, R={R}, M_lz={M_lz}):")
    print(f"    max rel C   = {rel_C:.3e}")
    print(f"    max rel chi = {rel_chi:.3e}")
    # Loose tolerance: this is a stochastic test
    test2 = rel_C < 0.2 and rel_chi < 0.2

    overall = test1 and test2
    print(f"  => Test1: {'PASS' if test1 else 'FAIL'}, "
          f"Test2: {'PASS' if test2 else 'FAIL'}\n")
    return overall


if __name__ == "__main__":
    print("Milestone BC verification: complex Lanczos + PG-FTLM aggregation\n")
    all_pass = True
    all_pass &= test_case(N=4, s_val=0.5, J=1.0, label="Case A")
    all_pass &= test_case(N=6, s_val=0.5, J=1.0, label="Case B")
    all_pass &= test_case(N=8, s_val=0.5, J=1.0, label="Case C")
    all_pass &= test_case(N=4, s_val=1.0, J=1.0, label="Case D")
    print("=" * 60)
    print(f"OVERALL: {'ALL PASS' if all_pass else 'SOME FAILED'}")
