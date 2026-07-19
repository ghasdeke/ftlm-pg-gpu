# Symmetry-adapted basis toolbox (pure MATLAB)

From the generators of a site-permutation symmetry group of a Heisenberg model
this toolbox builds the whole group, all irreducible representation matrices
(numerically, via Dixon's method) and the symmetry-adapted linear combinations
(SALCs) of the spin-1/2 product basis. It uses only base MATLAB (no toolboxes)
and sparse matrices for the `2^N`-dimensional operators.

## Files

| File                 | Purpose                                                              |
|----------------------|---------------------------------------------------------------------|
| `group_closure.m`    | Build the group from generators (Dimino), multiplication table, inverses, conjugacy classes |
| `dixon_irreps.m`     | All unitary irrep matrices + character table (Dixon's method)       |
| `salc_projectors.m`  | Spin permutation rep, isotypic projectors, transfer operators, SALC basis |
| `heisenberg_ring.m`  | Sparse Heisenberg Hamiltonian `H = sum_<i,j> S_i.S_j`               |
| `demo.m`             | Full N=6 ring (group D6, |G|=12): group → irreps → SALCs → block-diagonal H |
| `run_tests.m`        | Verification suite with explicit tolerance and PASS/FAIL output     |

## Quick start

```matlab
demo        % full worked example for the 6-site ring
run_tests   % verification suite
```

## Conventions

* A permutation `P` is a vector with `P(i)` = image of site `i`.
* Group multiplication is function composition `(A*B)(i) = A(B(i))`, i.e. `A(B)`.
* Spin state index `b` (0-based) stores the spin of site `i` in bit `i`
  (value `2^(i-1)`); bit `1` = up.
* Site permutation `g` acts on states by `(g.s)(i) = s(g^{-1} i)`.

## Changing the example

Edit the top of `demo.m` (or call the functions yourself):

```matlab
N    = 6;
T    = [2 3 4 5 6 1];   % translation
R    = [6 5 4 3 2 1];   % reflection
G    = group_closure({T, R}, N);
[ir, info] = dixon_irreps(G, 42, 1e-10);   % seed 42 -> reproducible
S    = salc_projectors(ir, G, N, 1e-10);
H    = heisenberg_ring(N);
Hs   = S.U' * H * S.U;                      % block diagonal
```

Use only translation `T` to get the cyclic group `C_N` (whose irreps are
genuinely complex — the code handles that).

## Icosahedron / dodecahedron (group I_h, |G| = 120)

A second example stresses the toolbox on the full icosahedral group I_h
(10 irreps, dims 1,3,3,4,5 in both parities) for the spin-1/2 Heisenberg
antiferromagnet `H = sum_<i,j> S_i.S_j`, and returns the ground-state energy
in the total-`S_z = 0` sector resolved by irrep.

| File                              | Purpose                                                       |
|-----------------------------------|---------------------------------------------------------------|
| `polyhedron_model.m`              | Vertices + nearest-neighbour bonds (icosahedron / dodecahedron) |
| `polyhedron_symmetries.m`         | Point-group symmetries from geometry, as vertex permutations  |
| `polyhedron_irrep_groundstates.m` | Group → Dixon irreps → M=0 sector → ground energy per irrep via `eigs` |
| `demo_polyhedra.m`                | Runs both solids, prints tables, saves a bar chart            |
| `run_tests_polyhedra.m`           | Verification (group, dims, sector counts, eigs vs dense)      |

```matlab
demo_polyhedra          % both solids
run_tests_polyhedra     % verification
out = polyhedron_irrep_groundstates('dodecahedron');   % programmatic use
```

Key points:
* The dodecahedron M=0 sector has dimension `C(20,10) = 184756`; no explicit
  symmetry-adapted basis is formed. For irrep `a` the ground energy is the
  smallest eigenvalue of `H` on `range(P_a)`, obtained as the dominant
  eigenpair of the shifted projected operator `M_a = P_a(cI - H)P_a` with
  `eigs(..., 'largestabs')` — robust because the huge `P_a` null space maps
  to 0 instead of dominating the Krylov space.
* Irreps are labelled physically (A_g, T1g, T2g, G_g, H_g and the `u`
  partners) using the inversion character (g/u) and the C5 character (T1/T2),
  identified from the geometric 3×3 matrices.

Results (J = 1):

| solid        | M=0 dim | global ground | irrep |
|--------------|---------|---------------|-------|
| icosahedron  | 924     | −6.18788996   | A_u   |
| dodecahedron | 184756  | −9.72218535   | A_u   |

## Square lattice A×B (space group, periodic torus)

Demonstrates the fully general, *generator-only* workflow: only candidate
symmetry operations are proposed; the procedure keeps the bond-preserving ones
and closes them. The space group of the periodic A×B square lattice is
`T ⋊ P` with translations `T = Z_A × Z_B` and point group
`P = D₄` (if A = B, |G| = 8A²) or `P = D₂` (if A ≠ B, |G| = 4AB). The
translation symmetry makes the irreps **genuinely complex** (lattice momenta).

| File                    | Purpose                                                          |
|-------------------------|------------------------------------------------------------------|
| `square_lattice_group.m`| Candidate generators (Tx,Ty,σx,σy,r4,σd) → bond-preservation filter → `group_closure` |
| `demo_square.m`         | Group + irrep dims for several lattices; sample complex irrep matrices |
| `run_tests_square.m`    | Verification (|G|, #classes=#irreps, Σd²=|G|, homomorphism, unitarity, Schur) |

```matlab
demo_square                              % several lattices
[G, info] = square_lattice_group(6, 6);  % |G| = 288 (D4 torus)
[irreps, di] = dixon_irreps(G, 1, 1e-10);
```

Results (periodic):

| lattice | point group | \|G\| | #irreps | irrep dims |
|---------|-------------|------|---------|------------|
| 3×3 | D₄ | 72  | 9  | 1·4, 2, 4·4 |
| 4×4 | D₄ | 128 | 20 | 1·8, 2·6, 4·6 |
| 6×6 | D₄ | 288 | 27 | 1·8, 2·6, 4·12, 8 |
| 3×4 | D₂ | 48  | 15 | 1·8, 2·6, 4 |
| 4×6 | D₂ | 96  | 30 | 1·16, 2·12, 4·2 |

Reuses `group_closure` and `dixon_irreps` unchanged; only the lattice-specific
generator builder is new. Practical size limit is set by Dixon on the |G|×|G|
regular representation (comfortable to |G| ≈ 500).

## Realification of irreps (Frobenius-Schur)

`realify_irreps` works on the output of `dixon_irreps` for **any** group. It
classifies each irrep by its Frobenius-Schur indicator
`ν = (1/|G|) Σ_g χ(g²)` and, for the real-type ones, returns matrices that are
**exactly real** (orthogonal); machine-epsilon imaginary residuals are removed.

| ν   | type        | can be made real? |
|-----|-------------|-------------------|
| +1  | real / orthogonal | yes            |
| 0   | complex (χ not real) | no          |
| −1  | pseudoreal / quaternionic | no     |

For ν = +1 the conjugate-rep intertwiner `S = Σ_g conj(D(g)) X D(g)'` is
Takagi-factorised, `conj(S) = U Uᵀ`, and `B(g) = U' D(g) U` is real.

| File                  | Purpose                                                  |
|-----------------------|----------------------------------------------------------|
| `realify_irreps.m`    | FS classification + realification (any group)            |
| `demo_realify.m`      | FS table across groups; complex→real before/after        |
| `run_tests_realify.m` | Verifies exactly-real, orthogonality, homomorphism, χ    |

```matlab
[ir, di]  = dixon_irreps(G, 1, 1e-10);
[rir, fs] = realify_irreps(ir, G);   % real-type irreps now exactly real
```

Examples: C6 has 2 real + 4 complex irreps (the complex ones are correctly left
complex); D6, I_h and the square lattices have only real-type irreps, all made
exactly real. `run_tests_realify` passes 9/9 (residuals removed ≈ 1e-17…1e-14).

