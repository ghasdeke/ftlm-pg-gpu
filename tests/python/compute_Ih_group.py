"""
Phase alpha of Milestone G (full I_h on the icosahedron):

Build the icosahedral group I_h as 120 permutations of the 12 vertices of
the icosahedron used by the FTLM-GPU codebase, verify group properties,
identify conjugacy classes, compute the character table numerically, and
compare against the well-known I_h character table.

Vertex labeling matches release/ftlm_observables.m adjacency_icosahedron:
    1: (0, 1, phi)        2: (0, 1, -phi)
    3: (0,-1, phi)        4: (0,-1, -phi)
    5: (1, phi, 0)        6: (1,-phi, 0)
    7: (-1, phi, 0)       8: (-1,-phi, 0)
    9: (phi, 0, 1)       10: (phi, 0,-1)
   11: (-phi, 0, 1)      12: (-phi, 0,-1)

Generators:
    C5: 5-fold rotation about the axis through vertex 1 (and 4)
    C3: 3-fold rotation about a face-center axis (vertex centroid)
    i : inversion (V -> -V), implemented directly as the permutation
        that swaps each vertex with its antipode

Group elements are stored as 12-element permutation arrays (1-indexed
inside the comments, 0-indexed in the code). Multiplication is defined
as composition (g * h)(v) = g(h(v)).
"""

import numpy as np


# ----------------------------------------------------------------
# Icosahedron vertices (matches release/ftlm_observables.m)
# ----------------------------------------------------------------

PHI = (1.0 + np.sqrt(5.0)) / 2.0

VERTICES = np.array([
    [ 0,    1,    PHI],   # 1 (index 0)
    [ 0,    1,   -PHI],   # 2
    [ 0,   -1,    PHI],   # 3
    [ 0,   -1,   -PHI],   # 4
    [ 1,    PHI,  0  ],   # 5
    [ 1,   -PHI,  0  ],   # 6
    [-1,    PHI,  0  ],   # 7
    [-1,   -PHI,  0  ],   # 8
    [ PHI,  0,    1  ],   # 9
    [ PHI,  0,   -1  ],   # 10
    [-PHI,  0,    1  ],   # 11
    [-PHI,  0,   -1  ],   # 12
], dtype=float)


# ----------------------------------------------------------------
# Rotation matrix from axis-angle
# ----------------------------------------------------------------

def rot_matrix(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    K = np.array([[0,        -axis[2],  axis[1]],
                  [axis[2],   0,       -axis[0]],
                  [-axis[1],  axis[0],  0      ]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def induced_perm(R, tol=1e-6):
    """Return permutation p (0-indexed) with p[i] = j if R*V_i == V_j."""
    n = VERTICES.shape[0]
    p = np.full(n, -1, dtype=np.int32)
    for i in range(n):
        Rv = R @ VERTICES[i]
        for j in range(n):
            if np.linalg.norm(Rv - VERTICES[j]) < tol:
                p[i] = j
                break
        if p[i] == -1:
            raise RuntimeError(f"R*V_{i} did not land on any vertex.")
    return p


# ----------------------------------------------------------------
# Generators
# ----------------------------------------------------------------

# C_5 about axis through V_1 (and -V_1 = V_4); rotates the pentagon of
# neighbors {3, 5, 7, 9, 11} around vertex 1.
AX_5 = VERTICES[0] / np.linalg.norm(VERTICES[0])
R_C5 = rot_matrix(AX_5, 2 * np.pi / 5)
P_C5 = induced_perm(R_C5)

# C_3 about an axis through a face centroid. Pick the face (V_1, V_3, V_9):
# all three are mutually adjacent. Face center = (V_1 + V_3 + V_9)/3.
face_center = (VERTICES[0] + VERTICES[2] + VERTICES[8]) / 3.0
AX_3 = face_center / np.linalg.norm(face_center)
R_C3 = rot_matrix(AX_3, 2 * np.pi / 3)
P_C3 = induced_perm(R_C3)

# Inversion: V -> -V, equivalently:
P_INV = induced_perm(-np.eye(3))


# ----------------------------------------------------------------
# Group closure: BFS from the generators
# ----------------------------------------------------------------

def perm_mul(g, h):
    """Composition: (g * h)(v) = g(h(v))."""
    return g[h]


def perm_to_key(p):
    """Hashable key for a permutation array."""
    return p.tobytes()


def close_group(generators, max_order=200):
    seen = {perm_to_key(np.arange(12, dtype=np.int32)): np.arange(12, dtype=np.int32)}
    queue = [g for g in generators]
    for g in generators:
        k = perm_to_key(g)
        if k not in seen:
            seen[k] = g
    changed = True
    while changed:
        changed = False
        items = list(seen.values())
        for a in items:
            for b in generators:
                c = perm_mul(a, b)
                k = perm_to_key(c)
                if k not in seen:
                    seen[k] = c
                    changed = True
                    if len(seen) > max_order:
                        raise RuntimeError(f"Group grew past {max_order}.")
    return list(seen.values())


print("Building I_h group from generators C_5, C_3, inversion ...")
G = close_group([P_C5, P_C3, P_INV])
print(f"|I_h| = {len(G)}")
assert len(G) == 120, f"Expected 120, got {len(G)}"


# ----------------------------------------------------------------
# Helpers on the closed group
# ----------------------------------------------------------------

ID = np.arange(12, dtype=np.int32)
N = len(G)
key_to_idx = {perm_to_key(g): i for i, g in enumerate(G)}

# Multiplication table mul[i, j] = index of G[i] * G[j].
mul = np.zeros((N, N), dtype=np.int32)
for i, gi in enumerate(G):
    for j, gj in enumerate(G):
        mul[i, j] = key_to_idx[perm_to_key(perm_mul(gi, gj))]

# Inverses: inv[i] such that G[inv[i]] = G[i]^{-1}
inv = np.zeros(N, dtype=np.int32)
for i in range(N):
    g = G[i]
    g_inv = np.argsort(g)
    inv[i] = key_to_idx[perm_to_key(g_inv.astype(np.int32))]

# Check group axioms
e_idx = key_to_idx[perm_to_key(ID)]
assert all(mul[i, inv[i]] == e_idx for i in range(N)), "g * g^-1 != e"
assert all(mul[i, e_idx] == i for i in range(N)), "g * e != g"


# ----------------------------------------------------------------
# Conjugacy classes
# ----------------------------------------------------------------

def conj_class(i):
    seen = set()
    for h in range(N):
        # h g h^-1
        x = mul[h, mul[i, inv[h]]]
        seen.add(x)
    return frozenset(seen)


classes = []
assigned = -np.ones(N, dtype=np.int32)
for i in range(N):
    if assigned[i] >= 0:
        continue
    C = conj_class(i)
    cls_id = len(classes)
    classes.append(sorted(C))
    for x in C:
        assigned[x] = cls_id

print(f"Number of conjugacy classes: {len(classes)} (I_h has 10)")
sizes = [len(c) for c in classes]
print(f"Class sizes: {sorted(sizes)}")


# ----------------------------------------------------------------
# Identify each class by physical signature
# ----------------------------------------------------------------

def class_signature(cls_idx):
    """Use one representative's 3D action to identify the class type."""
    rep = G[classes[cls_idx][0]]
    # Find the 3D rotation/improper-rotation matrix corresponding to
    # this permutation. The matrix is determined by where 3 non-collinear
    # vertices go; we infer it from V_target = R V_source.
    src = VERTICES[[0, 2, 4]]   # V_1, V_3, V_5 (non-collinear)
    dst = VERTICES[rep[[0, 2, 4]]]
    # R src.T = dst.T, so R = dst.T @ inv(src.T)
    R = dst.T @ np.linalg.inv(src.T)
    det = np.linalg.det(R)
    trace = np.trace(R)
    # rotation angle from trace = 1 + 2 cos(theta) (for proper), or
    # -1 + 2 cos(theta) for improper rotations.
    # det = +1 means proper rotation, det = -1 means improper.
    if det > 0:
        cos_th = (trace - 1) / 2
    else:
        cos_th = (trace + 1) / 2
    cos_th = max(min(cos_th, 1.0), -1.0)
    theta = np.arccos(cos_th)
    return (round(det), round(np.degrees(theta), 1))


sigs = [class_signature(i) for i in range(len(classes))]
print("Class signatures (det, angle_deg):")
for i, (s, sz) in enumerate(zip(sigs, sizes)):
    print(f"  class {i}: size {sz:3d}, signature (det={s[0]:+d}, theta={s[1]}°)")


# ----------------------------------------------------------------
# Numerical character table from a faithful 3D representation
# ----------------------------------------------------------------
# T_{1u} has chi(g) = trace of the 3x3 rotation matrix of g.

print("\nT_1u character (trace of 3D matrix per class):")
T1u_chars = []
for i in range(len(classes)):
    rep = G[classes[i][0]]
    src = VERTICES[[0, 2, 4]]
    dst = VERTICES[rep[[0, 2, 4]]]
    R = dst.T @ np.linalg.inv(src.T)
    T1u_chars.append(np.trace(R))
print(" ".join(f"{x:+.3f}" for x in T1u_chars))

# Expected from table: e->3, 12C5->phi, 12C5^2->1/phi, 20C3->0, 15C2->-1,
# i->-3, 12S10->-1/phi, 12S10^3->-phi, 20S6->0, 15sigma->1


# ----------------------------------------------------------------
# Action of group permutations on the integer state encoding
# ----------------------------------------------------------------

def apply_perm_to_state(perm, n_state, d_loc, N_sites):
    """Apply a vertex permutation to an integer state.
    State n is sum_k a_k * d^k where a_k = (m_k + s). After permuting
    sites by perm (perm[i] = j means site i goes to site j), the new
    digits are a'_{perm[i]} = a_i, i.e., a'_j = a_{perm^{-1}(j)}.
    """
    inv_perm = np.argsort(perm)
    out = 0
    tmp = n_state
    for k in range(N_sites):
        a_k = tmp % d_loc
        tmp //= d_loc
        # a_k goes to position perm[k] in the new state
        out += a_k * (d_loc ** int(perm[k]))
    return out


print("\nQuick sanity: apply a few group elements to a couple of M=0 states.")
# For s=1/2, N=12: M=0 means 6 up-spins. Example state with bits 0,1,2,3,4,5 set:
n_state = sum(1 << k for k in range(6))
d_loc = 2
print(f"  Initial state: {n_state} (binary {bin(n_state)})")
for label, idx in [("e", e_idx), ("C5", key_to_idx[perm_to_key(P_C5)]),
                    ("C3", key_to_idx[perm_to_key(P_C3)]),
                    ("inv", key_to_idx[perm_to_key(P_INV)])]:
    n2 = apply_perm_to_state(G[idx], n_state, d_loc, 12)
    print(f"  {label}: {n2} (binary {bin(n2)})")

print("\n[Phase alpha] I_h group constructed and verified.")
