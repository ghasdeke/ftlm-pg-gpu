"""
Phase beta of Milestone G (full I_h on the icosahedron):
1D irreps A_g and A_u super-rep enumeration + sparse Heisenberg builder
+ ED verification.

For the s=1/2 icosahedron (n_total = 4096) we test:

  1. Orbit structure: every M-sector state has an I_h orbit dividing
     |G| = 120; super-reps are the orbit minima.
  2. A_g compatibility: trivially all super-reps included.
  3. A_u compatibility: super-rep r contributes iff its stabilizer
     contains only proper rotations (det = +1).
  4. Sparse H in the (M, Gamma) symmetry-adapted basis, matrix elements
     M(r, a_R) = sqrt(L_r/L_{a_R}) * c_a * chi_Gamma(g_a) where g_a
     satisfies g_a(a_R) = a (a is the spin-flipped state from r).
  5. Concatenating A_g + A_u + (T1g + T1u + ...) ED spectra must give
     the full M-sector spectrum. Here in phase beta we only compute
     the 1D irreps; we therefore verify dim(M, A_g) + dim(M, A_u) is
     the SUBSPACE dimension that we cover (not the full M dim), and
     compare to direct diagonalisation projected onto the 1D-irrep
     subspaces.
"""

import numpy as np

import sys
sys.path.insert(0, '.')

# Re-use the I_h group construction from Phase alpha
from verify_Ih_full import (
    perm_matrix, C5_perm, C3_perm, Ci_perm,
    A_C5, A_C3, T1_C5, T1_C3, T2_C5, T2_C3,
    F_C5, F_C3, H_C5, H_C3,
    M_list, IH_perm, IH_keys, IH_Ag, IH_Au,
)

print(f"I_h has {len(IH_perm)} elements.")


# ----------------------------------------------------------------
# Vertex labelling info: bonds from C_5 around vertex 1
# ----------------------------------------------------------------

V1_NEIGHBOURS = [2, 5, 9, 10, 12]


def all_bonds():
    """30 nearest-neighbour bonds in the user's vertex labelling."""
    seen = set()
    for k in range(120):
        # Convert permutation matrix to permutation array (0-indexed)
        P = IH_perm[k]
        perm = np.argmax(P, axis=0)   # perm[col] = row, so col -> row
        v1_img = perm[0]              # vertex 1 -> ?
        for nb in V1_NEIGHBOURS:
            v2_img = perm[nb - 1]
            a = min(v1_img, v2_img)
            b = max(v1_img, v2_img)
            seen.add((a, b))
    bonds = sorted(seen)
    assert len(bonds) == 30, len(bonds)
    return bonds


BONDS = all_bonds()


# ----------------------------------------------------------------
# State integer encoding (12 sites, base d_loc)
# ----------------------------------------------------------------

D_LOC = 2
N_SITES = 12
S_VAL = 0.5


def digits_of(n):
    out = []
    for _ in range(N_SITES):
        out.append(n % D_LOC)
        n //= D_LOC
    return out


def apply_perm_to_state(perm_0idx, n):
    """perm_0idx[i] = j means site i goes to site j. New digit at j = old digit at i."""
    digs = digits_of(n)
    new_digs = [0] * N_SITES
    for i in range(N_SITES):
        new_digs[perm_0idx[i]] = digs[i]
    out = 0
    for k in range(N_SITES):
        out += new_digs[k] * (D_LOC ** k)
    return out


# Precompute the 120 permutations in array form (0-indexed)
PERMS = []
for k in range(120):
    P = IH_perm[k]
    perm = np.argmax(P, axis=0)
    PERMS.append(perm)


# Find the identity element index. PERMS[0] is NOT generally identity —
# it's the first generator (e.g., C_5 in I_h). The identity is wherever
# the closure puts it, which for I_h happens to be index 24.
IDENTITY_IDX = None
for _k in range(120):
    if np.array_equal(PERMS[_k], np.arange(N_SITES)):
        IDENTITY_IDX = _k
        break
assert IDENTITY_IDX is not None, "Identity element not found in PERMS"


# ----------------------------------------------------------------
# Min-image search over I_h
# ----------------------------------------------------------------

def min_image_Ih(n):
    """Return (rep, g_min_idx) where rep is the I_h-orbit minimum and
    g_min_idx is the index of a group element with g_min(n) = rep.
    If n is already its own orbit minimum, returns g_min = IDENTITY_IDX
    (NOT 0 — PERMS[0] is the first generator, not the identity)."""
    rep = n
    g_min = IDENTITY_IDX
    for g in range(120):
        n_g = apply_perm_to_state(PERMS[g], n)
        if n_g < rep:
            rep = n_g
            g_min = g
    return rep, g_min


def stabilizer_Ih(n):
    """Return the indices of the group elements that fix n."""
    stab = []
    for g in range(120):
        if apply_perm_to_state(PERMS[g], n) == n:
            stab.append(g)
    return stab


# ----------------------------------------------------------------
# M-sector enumeration
# ----------------------------------------------------------------

def enumerate_M_sector(M_target):
    """All s=1/2 states with M = sum(m_i)/2 = M_target."""
    states = []
    for n in range(2 ** N_SITES):
        m = bin(n).count('1') - N_SITES / 2
        if abs(m - M_target) < 1e-9:
            states.append(n)
    return states


# ----------------------------------------------------------------
# Super-rep enumeration under I_h with 1D irrep compatibility
# ----------------------------------------------------------------

def enumerate_super_reps(M_target, gamma_chars):
    """Find I_h super-reps in the M_target sector compatible with the
    given irrep characters. gamma_chars is a list of length 120 with
    chi_Gamma(g) for g = 0..119."""
    states = enumerate_M_sector(M_target)
    super_reps = []
    orbit_lens = []
    g_min_arr  = []
    for n in states:
        rep, g_min = min_image_Ih(n)
        if rep != n:
            continue   # not a super-rep
        # Compatibility filter: sum over stabilizer of chi(h) != 0
        stab = stabilizer_Ih(n)
        s = sum(gamma_chars[h] for h in stab)
        if abs(s) < 1e-8:
            continue
        super_reps.append(n)
        orbit_lens.append(120 // len(stab))
        g_min_arr.append(g_min)
    return super_reps, orbit_lens, g_min_arr


# 1D irrep characters
def chi_Ag(g): return 1.0
def chi_Au(g): return float(IH_Au[g])   # = det(g)


# ----------------------------------------------------------------
# Sparse Heisenberg in (M, Gamma) basis (1D)
# ----------------------------------------------------------------

def build_H_Ih_1D(super_reps, orbit_lens, gamma_chars):
    """Sparse H in the (M, Gamma) basis for a 1D irrep, using the formula
        M(r', r) = sqrt(L_r/L_{r'}) * c_a * chi(g_a)
    where r is the source super-rep, a is the spin-flipped state, r' is
    a's I_h-rep, and g_a is the group element such that g_a(r') = a.
    """
    dim = len(super_reps)
    if dim == 0:
        return np.zeros((0, 0))
    rep_to_idx = {r: i for i, r in enumerate(super_reps)}
    powers = [D_LOC ** k for k in range(N_SITES)]
    J = 1.0
    s = S_VAL

    H = np.zeros((dim, dim), dtype=np.complex128)

    for i, r in enumerate(super_reps):
        L_r = orbit_lens[i]
        dg = digits_of(r)
        m = [dg[k] - s for k in range(N_SITES)]

        # Diagonal
        diag = 0.0
        for (a, b) in BONDS:
            diag += J * m[a] * m[b]
        H[i, i] += diag

        # Off-diagonal: spin flips
        for (a_site, b_site) in BONDS:
            ma, mb = m[a_site], m[b_site]
            # S+_a S-_b
            if dg[a_site] < D_LOC - 1 and dg[b_site] > 0:
                coeff = 0.5 * J * np.sqrt(s*(s+1) - ma*(ma+1)) * np.sqrt(s*(s+1) - mb*(mb-1))
                n_new = r + powers[a_site] - powers[b_site]
                rep_a, g_a_min = min_image_Ih(n_new)
                # g_a_min: g_min(n_new) = rep_a. We want g_a such that
                # g_a(rep_a) = n_new, i.e., g_a = g_min^{-1}.
                # Equivalently chi(g_a) = chi(g_min^{-1}) = chi(g_min)
                # for real-character (1D real) irreps.
                if rep_a in rep_to_idx:
                    j = rep_to_idx[rep_a]
                    L_a = orbit_lens[j]
                    norm = np.sqrt(L_r / L_a)
                    H[j, i] += coeff * norm * gamma_chars[g_a_min]
            # S-_a S+_b
            if dg[a_site] > 0 and dg[b_site] < D_LOC - 1:
                coeff = 0.5 * J * np.sqrt(s*(s+1) - ma*(ma-1)) * np.sqrt(s*(s+1) - mb*(mb+1))
                n_new = r - powers[a_site] + powers[b_site]
                rep_a, g_a_min = min_image_Ih(n_new)
                if rep_a in rep_to_idx:
                    j = rep_to_idx[rep_a]
                    L_a = orbit_lens[j]
                    norm = np.sqrt(L_r / L_a)
                    H[j, i] += coeff * norm * gamma_chars[g_a_min]

    return H


# ----------------------------------------------------------------
# Reference: full M-sector H without symmetry
# ----------------------------------------------------------------

def build_H_M_full(M_target):
    states = enumerate_M_sector(M_target)
    idx_map = {n: i for i, n in enumerate(states)}
    dim = len(states)
    powers = [D_LOC ** k for k in range(N_SITES)]
    s = S_VAL; J = 1.0
    H = np.zeros((dim, dim))
    for i, n in enumerate(states):
        dg = digits_of(n)
        m = [dg[k] - s for k in range(N_SITES)]
        diag = 0.0
        for (a_s, b_s) in BONDS:
            diag += J * m[a_s] * m[b_s]
        H[i, i] = diag
        for (a_s, b_s) in BONDS:
            ma, mb = m[a_s], m[b_s]
            if dg[a_s] < D_LOC - 1 and dg[b_s] > 0:
                c = 0.5 * J * np.sqrt(s*(s+1) - ma*(ma+1)) * np.sqrt(s*(s+1) - mb*(mb-1))
                n_new = n + powers[a_s] - powers[b_s]
                if n_new in idx_map:
                    H[idx_map[n_new], i] += c
            if dg[a_s] > 0 and dg[b_s] < D_LOC - 1:
                c = 0.5 * J * np.sqrt(s*(s+1) - ma*(ma-1)) * np.sqrt(s*(s+1) - mb*(mb+1))
                n_new = n - powers[a_s] + powers[b_s]
                if n_new in idx_map:
                    H[idx_map[n_new], i] += c
    return H, states


# ----------------------------------------------------------------
# Test: A_g + A_u sub-block eigenvalues from H_M projected via
# group-theoretic projector
# ----------------------------------------------------------------

def build_projector(states, gamma_chars):
    """Projector P_Gamma = (d_Gamma/|G|) Sum_g chi_Gamma(g)^* rho(g),
    here in the M-sector basis. For 1D irreps d_Gamma = 1."""
    dim = len(states)
    idx_map = {n: i for i, n in enumerate(states)}
    P = np.zeros((dim, dim), dtype=np.complex128)
    for g in range(120):
        ch = gamma_chars[g]
        if abs(ch) < 1e-15:
            continue
        for i, n in enumerate(states):
            n_g = apply_perm_to_state(PERMS[g], n)
            if n_g in idx_map:
                j = idx_map[n_g]
                P[j, i] += np.conj(ch)
    return P / 120.0


def verify(M_target):
    print(f"\n--- M = {M_target} ---")
    H_M, states_M = build_H_M_full(M_target)
    dim_M = len(states_M)
    print(f"  dim M-sector (no symmetry) = {dim_M}")

    # Reference: diagonalize the full M-sector H, then project onto A_g and A_u
    P_Ag = build_projector(states_M, [chi_Ag(g) for g in range(120)])
    P_Au = build_projector(states_M, [chi_Au(g) for g in range(120)])

    rank_Ag = np.linalg.matrix_rank(P_Ag, tol=1e-8)
    rank_Au = np.linalg.matrix_rank(P_Au, tol=1e-8)
    print(f"  Projector ranks: A_g = {rank_Ag}, A_u = {rank_Au}")

    # Symmetrise H to A_g block: U^T H U where U columns span A_g subspace
    eig_P_Ag, V_P_Ag = np.linalg.eigh(P_Ag)
    plus_Ag = np.where(eig_P_Ag > 0.5)[0]
    U_Ag = V_P_Ag[:, plus_Ag]
    H_block_Ag = U_Ag.conj().T @ H_M @ U_Ag
    H_block_Ag = 0.5 * (H_block_Ag + H_block_Ag.conj().T)
    E_Ag_ref = np.sort(np.linalg.eigvalsh(H_block_Ag))

    eig_P_Au, V_P_Au = np.linalg.eigh(P_Au)
    plus_Au = np.where(eig_P_Au > 0.5)[0]
    U_Au = V_P_Au[:, plus_Au]
    H_block_Au = U_Au.conj().T @ H_M @ U_Au
    H_block_Au = 0.5 * (H_block_Au + H_block_Au.conj().T)
    E_Au_ref = np.sort(np.linalg.eigvalsh(H_block_Au))

    # Our construction
    reps_Ag, L_Ag, _ = enumerate_super_reps(M_target, [chi_Ag(g) for g in range(120)])
    reps_Au, L_Au, _ = enumerate_super_reps(M_target, [chi_Au(g) for g in range(120)])
    print(f"  Our dim A_g = {len(reps_Ag)}, A_u = {len(reps_Au)}")
    if len(reps_Ag) != rank_Ag or len(reps_Au) != rank_Au:
        print(f"  DIM MISMATCH: ref Ag={rank_Ag} ours={len(reps_Ag)}, ref Au={rank_Au} ours={len(reps_Au)}")
        return False

    H_Ag = build_H_Ih_1D(reps_Ag, L_Ag, [chi_Ag(g) for g in range(120)])
    H_Ag = 0.5 * (H_Ag + H_Ag.conj().T)
    E_Ag = np.sort(np.real(np.linalg.eigvalsh(H_Ag)))
    err_Ag = np.max(np.abs(E_Ag - E_Ag_ref)) if len(E_Ag) > 0 else 0.0

    H_Au = build_H_Ih_1D(reps_Au, L_Au, [chi_Au(g) for g in range(120)])
    H_Au = 0.5 * (H_Au + H_Au.conj().T)
    E_Au = np.sort(np.real(np.linalg.eigvalsh(H_Au)))
    err_Au = np.max(np.abs(E_Au - E_Au_ref)) if len(E_Au) > 0 else 0.0

    status_Ag = "OK " if err_Ag < 1e-10 else "FAIL"
    status_Au = "OK " if err_Au < 1e-10 else "FAIL"
    print(f"  A_g block: dim={len(E_Ag)}, max|dE| = {err_Ag:.2e} [{status_Ag}]")
    print(f"  A_u block: dim={len(E_Au)}, max|dE| = {err_Au:.2e} [{status_Au}]")
    return err_Ag < 1e-10 and err_Au < 1e-10


if __name__ == "__main__":
    print("Phase beta verification: 1D irreps A_g and A_u on s=1/2 icosahedron.\n")
    ok = True
    for M in [6, 5, 4, 3, 2, 1, 0]:
        ok &= verify(M)
    print()
    print(f"OVERALL: {'ALL PASS' if ok else 'SOME FAILED'}")
