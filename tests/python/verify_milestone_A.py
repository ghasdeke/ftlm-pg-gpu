"""
Milestone A verification: rep-basis Heisenberg builder for a ring with
C_N translation symmetry.

For each (N, s) test case:
  1. Build full M-sector Hamiltonian (no symmetry) via direct sparse builder.
  2. For each (M, k), enumerate (M, k) representatives and build the
     symmetry-adapted Hamiltonian using
        M(r, r') = sqrt(L_r / L_{r'}) * c_a * exp(i k h_a)
     where a is the connected state via spin-flip and h_a is the translation
     that brings a's representative back to a (a = T^{h_a} a_R, so
     min_image returns h with T^h(a) = a_R, then h_a = N - h mod N).
  3. Aggregate (M, k) spectra and compare to per-M spectra (sorted).

PASS criterion: max |sort(E_agg_M) - sort(E_full_M)| < tol per M-sector.
"""

import numpy as np
import scipy.sparse as sp


# ---------- helpers ----------

def cyclic_shift(n, d_loc, N):
    """Right cyclic shift of N-digit base-d_loc integer."""
    d_top = d_loc ** (N - 1)
    return n // d_loc + (n % d_loc) * d_top


def min_image_ring(n, d_loc, N):
    """Return (rep, h, L) where rep = min over orbit, T^h(n) = rep, L = orbit length."""
    rep = n
    h = 0
    L = N
    n_cur = n
    for g in range(1, N):
        n_cur = cyclic_shift(n_cur, d_loc, N)
        if n_cur == n:
            L = g
            break
        if n_cur < rep:
            rep = n_cur
            h = g
    return rep, h, L


def digits_of(n, d_loc, N):
    """Return list of N base-d_loc digits, least-significant first."""
    out = []
    for _ in range(N):
        out.append(n % d_loc)
        n //= d_loc
    return out


def enumerate_M_sector(N, s_val, M_target):
    """Brute-force enumeration of all states with total S^z = M_target."""
    d_loc = round(2 * s_val + 1)
    n_total = d_loc ** N
    states = []
    for n in range(n_total):
        m = 0.0
        tmp = n
        for _ in range(N):
            m += (tmp % d_loc) - s_val
            tmp //= d_loc
        if abs(m - M_target) < 1e-9:
            states.append(n)
    return states


def enumerate_sector_with_translation(N, s_val, M_target, p_irrep):
    """Return (reps, orbit_lens, dim) for the (M, k=2pi p/N) sector on a ring."""
    d_loc = round(2 * s_val + 1)
    states_M = enumerate_M_sector(N, s_val, M_target)
    reps = []
    Ls = []
    for n in states_M:
        rep, h, L = min_image_ring(n, d_loc, N)
        if rep == n:
            # n is the representative of its orbit
            # compatibility: p * L mod N == 0
            if (p_irrep * L) % N == 0:
                reps.append(n)
                Ls.append(L)
    reps_arr = np.array(reps, dtype=np.int64)
    Ls_arr = np.array(Ls, dtype=np.int32)
    # already sorted because states_M was sorted
    return reps_arr, Ls_arr, len(reps_arr)


def ring_bonds(N):
    return [(i, (i + 1) % N) for i in range(N)]


def build_H_full_M_sector(N, s_val, J, M_target):
    """Build sparse H on the unsymmetrized M-sector basis."""
    d_loc = round(2 * s_val + 1)
    bonds = ring_bonds(N)
    states = enumerate_M_sector(N, s_val, M_target)
    dim = len(states)
    if dim == 0:
        return sp.csr_matrix((0, 0)), states
    # state -> index
    idx_map = {n: i for i, n in enumerate(states)}
    powers = [d_loc ** k for k in range(N)]

    rows, cols, vals = [], [], []
    for i, n in enumerate(states):
        dg = digits_of(n, d_loc, N)
        m = [dg[k] - s_val for k in range(N)]
        # diagonal
        diag = 0.0
        for (a, b) in bonds:
            diag += J * m[a] * m[b]
        rows.append(i); cols.append(i); vals.append(diag)
        # off-diag
        for (a, b) in bonds:
            ma, mb = m[a], m[b]
            # S+_a S-_b
            if dg[a] < d_loc - 1 and dg[b] > 0:
                coeff = 0.5 * J * \
                    np.sqrt(s_val * (s_val + 1) - ma * (ma + 1)) * \
                    np.sqrt(s_val * (s_val + 1) - mb * (mb - 1))
                n_new = n + powers[a] - powers[b]
                if n_new in idx_map:
                    j = idx_map[n_new]
                    rows.append(j); cols.append(i); vals.append(coeff)
            # S-_a S+_b
            if dg[a] > 0 and dg[b] < d_loc - 1:
                coeff = 0.5 * J * \
                    np.sqrt(s_val * (s_val + 1) - ma * (ma - 1)) * \
                    np.sqrt(s_val * (s_val + 1) - mb * (mb + 1))
                n_new = n - powers[a] + powers[b]
                if n_new in idx_map:
                    j = idx_map[n_new]
                    rows.append(j); cols.append(i); vals.append(coeff)
    H = sp.coo_matrix((vals, (rows, cols)), shape=(dim, dim)).tocsr()
    return H, states


def build_H_pg_sector(N, s_val, J, reps, Ls, p_irrep):
    """Build sparse H on the (M, k) representative basis."""
    d_loc = round(2 * s_val + 1)
    bonds = ring_bonds(N)
    dim = len(reps)
    if dim == 0:
        return sp.csr_matrix((0, 0))
    idx_map = {int(r): i for i, r in enumerate(reps)}
    powers = [d_loc ** k for k in range(N)]
    k_phase = 2 * np.pi * p_irrep / N
    is_real = (p_irrep == 0) or (2 * p_irrep == N)
    dtype = np.float64 if is_real else np.complex128

    rows, cols, vals = [], [], []
    for i, r in enumerate(reps):
        r = int(r)
        L_r = int(Ls[i])
        dg = digits_of(r, d_loc, N)
        m = [dg[k] - s_val for k in range(N)]
        # diagonal (H invariant under T, factor = 1)
        diag = 0.0
        for (a, b) in bonds:
            diag += J * m[a] * m[b]
        rows.append(i); cols.append(i); vals.append(diag)
        # off-diag
        for (a, b) in bonds:
            ma, mb = m[a], m[b]
            # S+_a S-_b
            if dg[a] < d_loc - 1 and dg[b] > 0:
                coeff = 0.5 * J * \
                    np.sqrt(s_val * (s_val + 1) - ma * (ma + 1)) * \
                    np.sqrt(s_val * (s_val + 1) - mb * (mb - 1))
                n_new = r + powers[a] - powers[b]
                rep_a, h_a_minimg, L_a = min_image_ring(n_new, d_loc, N)
                if rep_a in idx_map:
                    j = idx_map[rep_a]
                    # min_image gives T^h_minimg(n_new) = rep_a, so n_new = T^{-h_minimg}(rep_a),
                    # i.e., h_a (the translation with T^{h_a}(rep_a) = n_new) = -h_minimg mod N
                    h_a = (N - h_a_minimg) % N
                    norm = np.sqrt(L_r / L_a)
                    if is_real:
                        phase = np.cos(k_phase * h_a)
                    else:
                        phase = np.exp(1j * k_phase * h_a)
                    val = coeff * norm * phase
                    rows.append(j); cols.append(i); vals.append(val)
            # S-_a S+_b
            if dg[a] > 0 and dg[b] < d_loc - 1:
                coeff = 0.5 * J * \
                    np.sqrt(s_val * (s_val + 1) - ma * (ma - 1)) * \
                    np.sqrt(s_val * (s_val + 1) - mb * (mb + 1))
                n_new = r - powers[a] + powers[b]
                rep_a, h_a_minimg, L_a = min_image_ring(n_new, d_loc, N)
                if rep_a in idx_map:
                    j = idx_map[rep_a]
                    h_a = (N - h_a_minimg) % N
                    norm = np.sqrt(L_r / L_a)
                    if is_real:
                        phase = np.cos(k_phase * h_a)
                    else:
                        phase = np.exp(1j * k_phase * h_a)
                    val = coeff * norm * phase
                    rows.append(j); cols.append(i); vals.append(val)
    H = sp.coo_matrix((vals, (rows, cols)), shape=(dim, dim), dtype=dtype).tocsr()
    return H


def run_one_test(N, s_val, J, tol=None):
    d_loc = round(2 * s_val + 1)
    M_max = round(N * s_val)
    if tol is None:
        tol = 1e-10 * abs(J) * N

    overall_pass = True
    print(f"=== N={N}, s={s_val}, J={J} (d_loc={d_loc}, dim_full=2^N or {d_loc}^N={d_loc**N}) ===")
    for M in range(-M_max, M_max + 1):
        H_full, states_M = build_H_full_M_sector(N, s_val, J, M)
        dim_full = H_full.shape[0]
        if dim_full == 0:
            continue
        E_full = np.sort(np.linalg.eigvalsh(H_full.toarray()))

        E_agg_list = []
        sector_dims = []
        for p in range(N):
            reps, Ls, dim_pg = enumerate_sector_with_translation(N, s_val, M, p)
            if dim_pg == 0:
                continue
            H_pg = build_H_pg_sector(N, s_val, J, reps, Ls, p)
            # H_pg should be Hermitian (or real symmetric)
            herm_err = np.max(np.abs(H_pg.toarray() - H_pg.toarray().conj().T))
            if herm_err > 1e-10:
                print(f"  M={M}, p={p}: WARNING H not Hermitian, err={herm_err:.3e}")
            E_pg = np.sort(np.linalg.eigvalsh(0.5 * (H_pg.toarray() + H_pg.toarray().conj().T)))
            E_agg_list.append(E_pg)
            sector_dims.append((p, dim_pg))

        E_agg = np.sort(np.concatenate(E_agg_list)) if E_agg_list else np.array([])
        if len(E_agg) != len(E_full):
            print(f"  M={M:+d}: DIM MISMATCH full={len(E_full)}, agg={len(E_agg)} (sectors {sector_dims})")
            overall_pass = False
            continue
        diff = np.max(np.abs(E_agg - E_full))
        dim_str = ",".join(f"p={p}:{d}" for p, d in sector_dims)
        status = "OK " if diff < tol else "FAIL"
        print(f"  M={M:+d}: dim_full={dim_full}, dims_pg=[{dim_str}], max|dE|={diff:.3e} [{status}]")
        if diff > tol:
            overall_pass = False
    print(f"=> {'PASS' if overall_pass else 'FAIL'}\n")
    return overall_pass


if __name__ == "__main__":
    print("Milestone A verification: rep-basis vs full Heisenberg ED\n")
    all_pass = True
    all_pass &= run_one_test(N=4, s_val=0.5, J=1.0)
    all_pass &= run_one_test(N=6, s_val=0.5, J=1.0)
    all_pass &= run_one_test(N=8, s_val=0.5, J=1.0)
    all_pass &= run_one_test(N=4, s_val=1.0, J=1.0)
    all_pass &= run_one_test(N=10, s_val=0.5, J=1.0)
    all_pass &= run_one_test(N=4, s_val=0.5, J=-0.7)  # ferromagnetic
    print("=" * 60)
    print(f"OVERALL: {'ALL PASS' if all_pass else 'SOME FAILED'}")
