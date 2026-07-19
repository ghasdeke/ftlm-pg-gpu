# Adding a new geometry (the provider contract)

A new system = **three building blocks**, then one `case` in each driver, then a
validation. The generic `irreps_from_group` + space-group machinery makes new
lattices cheap — **no kernel change** as long as `max d <= 12` and `|G| <= 65535`.

## 0. Shortcut: only the permutation generators (no new MATLAB code)

If you just want to run a symmetry/system that is **not hard-coded**, you do not
need to write a provider at all. Supply only the **permutation generators** and
the **bond list** in an input file with `geometry = 'generators'`:

```matlab
geometry = 'generators';
gens  = { p1, p2, ... };          % cell/matrix of 1xN site perms, P(i)=image of i
bonds = [ 1 2; 2 3; ... ];        % nearest-neighbour bonds (geometry, 1-based)
point_group = 'my group';         % optional, cosmetic
sys_name    = 'mysys';            % optional, used for the output filename tag
s_val = 0.5; J = 1.0; R = 8; M_lz = 80; T_range = logspace(-1,1,40);
```

The driver calls `group_from_generators(gens)` → closes the generators into the
full group (Dimino, via `group_closure`) in the exact struct layout below, then
`irreps_from_group` + `realify_irreps` as usual. The **bond list is geometry,
not symmetry**: it is a separate input and **must be invariant under every group
element** (the central correctness condition — the validation tests it). See
`examples/input_generators_square4x4_ED.m` and `tests/validate_generators_square4x4.m`
(reproduces the native `square_lattice_spacegroup(4,4)` and the full ED purely
from hand-written generators). Writing a dedicated provider (below) is only worth
it when you want the generators built programmatically from a size parameter.

## 1. Group provider → a struct

Return a struct with these fields (reference: `icosahedron_Ih_full`,
`square_lattice_spacegroup`, `kagome_spacegroup`, `triangular_spacegroup`):

| field | type | meaning |
|---|---|---|
| `N` | scalar | number of sites |
| `order` | scalar | group order \|G\| (must be ≤ 65535) |
| `perms` | `[order × N]` | `perms(k,i) = j`: site `i` → `j` under element `k` (1-based) |
| `perm_mats` | `[N × N × order]` | permutation matrices (`P(a,b)=1 iff b→a`) |
| `inv` | `[order × 1]` int32 | index of each element's inverse |
| `mul` | `[order × order]` int32 | multiplication table, `mul(a,b) = idx(G(a)·G(b))` |
| `identity` | int32 | index of the identity |
| `class_idx` / `class_size` | | conjugacy-class index / sizes |
| `irreps` | `1 × n_irrep` struct: `.name/.d/.mats[d×d×order]` | the generic irrep list |

The `irreps` field is obtained for free from **`irreps_from_group(group)`**
(decomposes the regular representation via a random Hermitian commutant element;
cross-validated against I_h). It needs only `order/mul/inv/identity`. For abelian
translation groups the provider builds the 1-D momentum irreps directly.

## 2. Adjacency provider → `bonds [n_bonds × 2]`

The nearest-neighbour bond list, **invariant under every group element** — this is
the central correctness condition (test it explicitly).

## 3. `min_image` is already generic

`min_image_Ih` uses only `group.N / perms / order / identity` — no per-system code.

## 4. Driver case (both drivers)

Add a `case '<name>'` to the `switch lower(geometry)` in `ftlm_observables_pg_Ih`
and `ftlm_observables_pg_gpu_Ih`: select the group + bonds, set `group.irreps =
irreps_from_group(group)`, and the `sys_label / sym_label / geo_tag` strings. The
GPU driver auto-enables `compact_v` for `max d >= 2` and asserts `max d <= 12`.

## 5. Validation standard (do this before scaling)

- `test_<sys>_spacegroup.m`: group axioms, Latin-square `mul`, inverses, **bond
  invariance** under every element, `min_image` consistency vs brute force.
- `validate_<sys>.m`: a small ED cross-check — symmetry-adapted run vs an
  independent full ED (`ed_full_heisenberg`): **sum rule = (2s+1)^N exact** and
  `C_symED(T)` vs full `ED(T)` to ~1e-13.

Only then scale up. Run `estimate_feasibility(group, bonds, s, R, M_lz)` first to
predict n_reps / n_entries / host RAM / VRAM mode and the hard `d` / `|G|` limits.

## Gotchas

- **Chiral cells** (2-D hexagonal supercells with `a ≠ b`, both ≠ 0): the mirror is
  not a torus symmetry; the providers filter point-group generators by
  `Tm\(M*Tm) ∈ ℤ` (→ C_6 instead of C_6v) to avoid the BFS closure blowing up.
- **`d > 8`**: needs a kernel `#define MAX_D` bump + recompile (watch register
  pressure); the pre-flight guard refuses such a run with a clear message.
- An input filename passed to `run()` must not start with `_`.
