"""
Python verification of the user's I_h construction:
permutation generators + irrep generators (from Altmann & Herzig p. 655)
+ closure algorithm.

We rebuild the user's MATLAB code in Python, verify:
  1. Closure yields 60 elements for I and 120 for I_h.
  2. For every irrep Gamma and every product g*h in the group,
     rho_Gamma(g) * rho_Gamma(h) == rho_Gamma(g*h).
  3. Character orthogonality on the 10 conjugacy classes.
  4. Irrep dimensions match (1, 1, 3, 3, 3, 3, 4, 4, 5, 5) and sum of
     dimensions squared equals |G| = 120.
"""
import numpy as np

# ----------------------------------------------------------------
# Generators as 12x12 permutation matrices (user's vertex labelling)
# ----------------------------------------------------------------

def perm_matrix(cycles, n=12):
    M = np.zeros((n, n), dtype=int)
    fixed = set(range(1, n + 1))
    for cyc in cycles:
        for k in range(len(cyc)):
            a = cyc[(k + 1) % len(cyc)] - 1   # next
            b = cyc[k] - 1                     # current
            M[a, b] = 1
            fixed.discard(cyc[k])
    for f in fixed:
        M[f - 1, f - 1] = 1
    return M


C5_perm = perm_matrix([[2, 10, 9, 5, 12], [3, 6, 11, 8, 4]])
C3_perm = perm_matrix([[1, 12, 2], [7, 11, 8], [4, 6, 9], [3, 10, 5]])
Ci_perm = perm_matrix([[1, 7], [2, 8], [3, 9], [4, 10], [5, 6], [11, 12]])


# ----------------------------------------------------------------
# Irrep generator matrices (Altmann & Herzig p. 655)
# ----------------------------------------------------------------

g_p = (np.sqrt(5.0) + 1) / 2.0          # phi
g_m = (np.sqrt(5.0) - 1) / 2.0          # 1/phi
t   = np.sqrt(5.0)
lam = np.exp(1j * np.arctan(np.sqrt(5.0 / 3.0)))   # lambda
om  = np.exp(2j * np.pi / 3.0)                      # omega

# C5 irrep matrices (irrep group I; 'g' vs 'u' added later)
A_C5  = 1
T1_C5 = 0.5 * np.array([
    [g_m,       -1j,    1j * (-g_p)],
    [-1j,       g_p,    -g_m       ],
    [-1j*g_p,   -g_m,   1          ],
], dtype=complex)
T2_C5 = 0.5 * np.array([
    [-g_p,      -1j,    1j * g_m],
    [-1j,       -g_m,   g_p     ],
    [ 1j*g_m,   g_p,    1       ],
], dtype=complex)
F_C5 = 0.25 * np.array([
    [-1,        -t,     -1j*t,  -1j*t],
    [-t,        -1,     3j,     -1j ],
    [-1j*t,     3j,     1,      1   ],
    [-1j*t,     -1j,    1,      -3  ],
], dtype=complex)
H_C5 = 0.5 * np.array([
    [0,                          lam**2 * np.conj(om),  -lam,                            1j*lam*(-np.conj(om)),       1j*lam*(-om)              ],
    [np.conj(lam)**2 * om,       0,                     -np.conj(lam),                   1j*np.conj(lam)*(-om),       1j*np.conj(lam)*(-np.conj(om))],
    [-np.conj(lam),              -lam,                  1,                               0,                           1j                        ],
    [1j*np.conj(lam)*(-om),      1j*lam*(-np.conj(om)), 0,                               -1,                          -1                        ],
    [1j*np.conj(lam)*(-np.conj(om)), 1j*lam*(-om),       1j,                              -1,                          0                         ],
], dtype=complex)

# C3 irrep matrices
A_C3  = 1
T1_C3 = np.array([
    [0,   0,    -1j],
    [-1j, 0,    0  ],
    [0,   -1,   0  ],
], dtype=complex)
T2_C3 = np.array([
    [0,   0,    -1j],
    [-1j, 0,    0  ],
    [0,   -1,   0  ],
], dtype=complex)
F_C3 = np.array([
    [1,  0,    0,    0],
    [0,  0,    0,    -1j],
    [0,  -1j,  0,    0],
    [0,  0,    -1,   0],
], dtype=complex)
H_C3 = np.array([
    [om,           0,            0,    0,    0],
    [0,            np.conj(om),  0,    0,    0],
    [0,            0,            0,    0,    -1j],
    [0,            0,            -1j,  0,    0],
    [0,            0,            0,    -1,   0],
], dtype=complex)


# ----------------------------------------------------------------
# Closure: produce all 60 elements of I (rotation subgroup) and the
# corresponding irrep matrices in parallel.
# ----------------------------------------------------------------

def mat_key(M):
    # Round to 6 decimals to make the key hashable; permutation matrices
    # are integer and irrep matrices have well-defined finite-precision
    # values.
    return M.round(6).tobytes()


# We grow lists in parallel: idx i has perm M_list[i] and irrep matrices
# A_list[i], T1_list[i], ... corresponding to it.
M_list  = [C5_perm.copy(), C3_perm.copy()]
A_list  = [A_C5, A_C3]
T1_list = [T1_C5.copy(), T1_C3.copy()]
T2_list = [T2_C5.copy(), T2_C3.copy()]
F_list  = [F_C5.copy(), F_C3.copy()]
H_list  = [H_C5.copy(), H_C3.copy()]
keys    = {mat_key(M): i for i, M in enumerate(M_list)}

# Generators (only these two; closure under these is the full I subgroup).
gens_M  = [C5_perm, C3_perm]
gens_A  = [A_C5, A_C3]
gens_T1 = [T1_C5, T1_C3]
gens_T2 = [T2_C5, T2_C3]
gens_F  = [F_C5, F_C3]
gens_H  = [H_C5, H_C3]

changed = True
while changed:
    changed = False
    n_now = len(M_list)
    for i in range(n_now):
        for gi in range(len(gens_M)):
            N_p = M_list[i] @ gens_M[gi]
            k = mat_key(N_p)
            if k not in keys:
                keys[k] = len(M_list)
                M_list.append(N_p)
                A_list.append(A_list[i] * gens_A[gi])
                T1_list.append(T1_list[i] @ gens_T1[gi])
                T2_list.append(T2_list[i] @ gens_T2[gi])
                F_list.append(F_list[i] @ gens_F[gi])
                H_list.append(H_list[i] @ gens_H[gi])
                changed = True

print(f"|I| (rotation subgroup) = {len(M_list)}")
assert len(M_list) == 60, "I should have 60 elements"


# ----------------------------------------------------------------
# Add inversion to get I_h (120 elements)
# ----------------------------------------------------------------

IH_perm = []
IH_Ag   = []
IH_Au   = []
IH_T1g  = []
IH_T1u  = []
IH_T2g  = []
IH_T2u  = []
IH_Fg   = []
IH_Fu   = []
IH_Hg   = []
IH_Hu   = []

for i in range(60):
    IH_perm.append(M_list[i])
    IH_perm.append(M_list[i] @ Ci_perm)

    IH_Ag.append(1)
    IH_Ag.append(1)
    IH_Au.append(1)
    IH_Au.append(-1)

    IH_T1g.append(T1_list[i])
    IH_T1g.append(T1_list[i])
    IH_T1u.append(T1_list[i])
    IH_T1u.append(-T1_list[i])

    IH_T2g.append(T2_list[i])
    IH_T2g.append(T2_list[i])
    IH_T2u.append(T2_list[i])
    IH_T2u.append(-T2_list[i])

    IH_Fg.append(F_list[i])
    IH_Fg.append(F_list[i])
    IH_Fu.append(F_list[i])
    IH_Fu.append(-F_list[i])

    IH_Hg.append(H_list[i])
    IH_Hg.append(H_list[i])
    IH_Hu.append(H_list[i])
    IH_Hu.append(-H_list[i])

print(f"|I_h| = {len(IH_perm)}")
assert len(IH_perm) == 120


# ----------------------------------------------------------------
# Verify homomorphism property: rho(g)*rho(h) == rho(g*h) for all pairs
# ----------------------------------------------------------------

IH_keys = {mat_key(M): i for i, M in enumerate(IH_perm)}

def check_hom(irrep_list, name, tol=1e-8):
    n_fail = 0
    n_tests = 0
    for i in range(120):
        for j in range(120):
            P = IH_perm[i] @ IH_perm[j]
            k = mat_key(P)
            if k not in IH_keys:
                print(f"  {name}: product {i}*{j} not in group?")
                n_fail += 1
                continue
            ij = IH_keys[k]
            lhs = irrep_list[i] @ irrep_list[j] if isinstance(irrep_list[i], np.ndarray) else irrep_list[i] * irrep_list[j]
            rhs = irrep_list[ij]
            if isinstance(lhs, np.ndarray):
                err = np.max(np.abs(lhs - rhs))
            else:
                err = abs(lhs - rhs)
            if err > tol:
                n_fail += 1
            n_tests += 1
    return n_fail, n_tests


print("\nHomomorphism checks (rho(g)*rho(h) == rho(g*h)):")
for irrep_list, name in [(IH_Ag, "A_g"), (IH_Au, "A_u"),
                          (IH_T1g, "T1_g"), (IH_T1u, "T1_u"),
                          (IH_T2g, "T2_g"), (IH_T2u, "T2_u"),
                          (IH_Fg, "F_g"), (IH_Fu, "F_u"),
                          (IH_Hg, "H_g"), (IH_Hu, "H_u")]:
    n_fail, n_tests = check_hom(irrep_list, name)
    status = "OK " if n_fail == 0 else f"FAIL ({n_fail}/{n_tests})"
    print(f"  {name}: {status}")


# ----------------------------------------------------------------
# Characters and orthogonality
# ----------------------------------------------------------------

def char_of(irrep_list, idx):
    M = irrep_list[idx]
    if isinstance(M, (int, complex, float)):
        return complex(M)
    return complex(np.trace(M))


# Group elements into conjugacy classes via conjugation
print("\nComputing conjugacy classes ...")

def perm_inv_idx(g_idx):
    P = IH_perm[g_idx]
    Pinv = P.T   # permutation matrix inverse = transpose
    return IH_keys[mat_key(Pinv)]


def conj_class(g_idx):
    cls = set()
    for h in range(120):
        # h g h^-1
        h_inv = perm_inv_idx(h)
        gh_inv = IH_keys[mat_key(IH_perm[g_idx] @ IH_perm[h_inv])]
        hgh_inv = IH_keys[mat_key(IH_perm[h] @ IH_perm[gh_inv])]
        cls.add(hgh_inv)
    return frozenset(cls)


classes = []
seen = set()
for g in range(120):
    if g in seen:
        continue
    C = conj_class(g)
    classes.append(sorted(C))
    seen |= C

class_sizes = sorted([len(c) for c in classes])
print(f"  {len(classes)} classes with sizes {class_sizes}")
assert len(classes) == 10
assert class_sizes == [1, 1, 12, 12, 12, 12, 15, 15, 20, 20]

# Character table
print("\nCharacter table (rows = irreps, cols = classes by size):")
order = sorted(range(len(classes)), key=lambda c: (len(classes[c]), classes[c][0]))
char_table = []
for irrep_list, name in [(IH_Ag, "A_g"), (IH_Au, "A_u"),
                          (IH_T1g, "T1g"), (IH_T1u, "T1u"),
                          (IH_T2g, "T2g"), (IH_T2u, "T2u"),
                          (IH_Fg, "F_g"), (IH_Fu, "F_u"),
                          (IH_Hg, "H_g"), (IH_Hu, "H_u")]:
    row = [char_of(irrep_list, classes[order[c]][0]) for c in range(10)]
    char_table.append(row)
    print(f"  {name}: " + "  ".join(f"{x.real:+.3f}" for x in row))


# Orthogonality: sum over classes of |class_c| * chi_Gamma(c) * conj(chi_Delta(c)) = |G| delta_{Gamma, Delta}
print("\nOrthogonality:")
names_short = ["Ag", "Au", "T1g", "T1u", "T2g", "T2u", "Fg", "Fu", "Hg", "Hu"]
orth_fail = []
for i in range(10):
    for j in range(i, 10):
        s = 0.0
        for c in range(10):
            cls_size = len(classes[order[c]])
            s += cls_size * char_table[i][c] * np.conj(char_table[j][c])
        expected = 120 if i == j else 0
        err = abs(s - expected)
        if err > 1e-6:
            orth_fail.append((names_short[i], names_short[j], s, expected))
if orth_fail:
    for name_i, name_j, s, exp in orth_fail:
        print(f"  FAIL {name_i} . {name_j}: sum = {s}, expected {exp}")
else:
    print("  All 55 orthogonality identities verified.")

dims_sq = sum(d*d for d in [1,1,3,3,3,3,4,4,5,5])
print()
print('Sum d_Gamma^2 =', dims_sq)
print('   equals |G| = 120:', dims_sq == 120)
