"""
Phase gamma.1 of Milestone G:

For the s = 1/2 icosahedron, project the full M-sector Heisenberg
Hamiltonian onto each of the 10 I_h irreducible representations using
the Frobenius projector

    P_Gamma = (d_Gamma / |G|) * sum_g chi_Gamma(g)^* rho_perm(g)

where chi_Gamma(g) = trace of the irrep matrix of g, and rho_perm(g) is
the permutation action on the M-sector basis. Diagonalize H on each
projected subspace.

Verification: the concatenated, sorted spectrum across all 10 irreps
must reproduce the direct M-sector spectrum, sorted, to machine
precision.

This is the ED reference for all 10 I_h irreps. It confirms that the
Phase-alpha group + irrep matrices and the Phase-beta projection
machinery extend correctly to non-Abelian higher-dimensional irreps.

The Frobenius approach diagonalizes a dim_M x dim_M projector per
irrep, which is fine for the s = 1/2 icosahedron (dim_M = 924) but
does NOT scale to s >= 2. Phase gamma.2 will provide the scalable
super-rep + column-picking construction.
"""

import numpy as np

from verify_Ih_full import IH_perm, IH_keys, M_list as _, \
    IH_Ag, IH_Au, IH_T1g, IH_T1u, IH_T2g, IH_T2u, \
    IH_Fg, IH_Fu, IH_Hg, IH_Hu

from verify_Ih_beta import (
    PERMS, apply_perm_to_state, BONDS,
    enumerate_M_sector, build_H_M_full,
)


# ----------------------------------------------------------------
# Characters from the irrep matrix lists.
# ----------------------------------------------------------------

def chars_from_matrices(irrep_list):
    """For 3D/4D/5D irreps with matrix entries; chi(g) = trace."""
    return [complex(np.trace(M)) for M in irrep_list]


def chars_from_scalars(irrep_list):
    """For 1D irreps stored as scalars (Ag, Au)."""
    return [complex(x) for x in irrep_list]


ALL_IRREPS = [
    ("A_g",  1, chars_from_scalars(IH_Ag)),
    ("A_u",  1, chars_from_scalars(IH_Au)),
    ("T1_g", 3, chars_from_matrices(IH_T1g)),
    ("T1_u", 3, chars_from_matrices(IH_T1u)),
    ("T2_g", 3, chars_from_matrices(IH_T2g)),
    ("T2_u", 3, chars_from_matrices(IH_T2u)),
    ("F_g",  4, chars_from_matrices(IH_Fg)),
    ("F_u",  4, chars_from_matrices(IH_Fu)),
    ("H_g",  5, chars_from_matrices(IH_Hg)),
    ("H_u",  5, chars_from_matrices(IH_Hu)),
]


# ----------------------------------------------------------------
# Build Frobenius projector P_Gamma on the M-sector basis.
# ----------------------------------------------------------------

def build_projector_M(states_M, irrep_chars, d_irrep):
    dim = len(states_M)
    idx_map = {n: i for i, n in enumerate(states_M)}
    P = np.zeros((dim, dim), dtype=np.complex128)
    for g in range(120):
        ch = np.conj(irrep_chars[g])   # chi^*
        if abs(ch) < 1e-14:
            continue
        for i, n in enumerate(states_M):
            n_g = apply_perm_to_state(PERMS[g], n)
            j = idx_map.get(n_g)
            if j is not None:
                P[j, i] += ch
    P *= d_irrep / 120.0
    return P


# ----------------------------------------------------------------
# Per-M sector verification
# ----------------------------------------------------------------

def verify_one_M(M_target):
    H_M, states_M = build_H_M_full(M_target)
    dim_M = len(states_M)
    E_M = np.sort(np.linalg.eigvalsh(H_M))

    all_E = []
    dim_check = 0
    per_irrep = []
    for name, d_irrep, chars in ALL_IRREPS:
        P = build_projector_M(states_M, chars, d_irrep)
        # P is Hermitian by construction (since chi(g^-1) = chi(g)^* and
        # we sum chi(g)^* rho(g) which is itself Hermitian up to factor d/|G|).
        P = 0.5 * (P + P.conj().T)
        eigs, V = np.linalg.eigh(P)
        # Eigenvalues should be 0 or 1; basis vectors are columns at eig ~ 1.
        plus_idx = np.where(eigs > 0.5)[0]
        n_basis = len(plus_idx)
        dim_check += n_basis
        per_irrep.append((name, d_irrep, n_basis))
        if n_basis == 0:
            continue
        U = V[:, plus_idx]
        H_block = U.conj().T @ H_M @ U
        H_block = 0.5 * (H_block + H_block.conj().T)
        E_block = np.real(np.linalg.eigvalsh(H_block))
        all_E.extend(E_block)

    all_E = np.sort(all_E)
    pass_dim = (dim_check == dim_M)
    if pass_dim:
        err = np.max(np.abs(all_E - E_M)) if dim_M > 0 else 0.0
    else:
        err = float('inf')

    print(f"  M = {M_target:+d}: dim_M = {dim_M}")
    for name, d_irrep, n_basis in per_irrep:
        if n_basis > 0:
            print(f"    {name:5s} (d={d_irrep}): block dim = {n_basis}")
    print(f"    Total block dims = {dim_check} (expected {dim_M})  [{ 'OK' if pass_dim else 'FAIL' }]")
    print(f"    max|dE| (aggregated vs full) = {err:.2e}  [{ 'OK' if err < 1e-10 else 'FAIL' }]")
    return pass_dim and err < 1e-10


if __name__ == "__main__":
    print("Phase gamma.1: all 10 I_h irrep blocks on s=1/2 icosahedron\n")
    ok = True
    for M in [0, 1, 2, 3]:
        if not verify_one_M(M):
            ok = False
        print()
    print("=" * 60)
    print(f"OVERALL: {'ALL PASS' if ok else 'SOME FAILED'}")
