# Symmetry-adapted GPU-FTLM for Heisenberg spin systems

Finite-temperature Lanczos (FTLM) and exact diagonalization (ED) for the
nearest-neighbour Heisenberg model, reduced **per irreducible representation of a
space/point group** and accelerated with a mixed FP32-GPU / FP64-CPU pipeline.
One CUDA SpMV kernel serves all systems; new lattices need no kernel change.

> This package generalizes an earlier published total-S^z-sector GPU-FTLM code
> by full point-/space-group symmetry (icosahedral and octahedral polyhedra,
> square/kagome/triangular lattices, and **arbitrary user-supplied permutation
> groups**).

## Method

Per total-S^z sector `M` and per irrep `Gamma`, the symmetry-reduced Hamiltonian
block is built from group "super-representatives" and run through a complex
block-Lanczos FTLM (FP32 GPU kernel) or dense ED (small blocks). Observables
`C(T)`, `chi(T)`, the spectrum and the FTLM sum rule are aggregated with the
`mult_M * d_Gamma` multiplicities. Drivers: `ftlm_observables_pg_Ih` (CPU FP64),
`ftlm_observables_pg_gpu_Ih` (mixed GPU).

Irreps of real type (Frobenius-Schur indicator +1) are realified by default and
take a real-FP32 kernel path -- about half the VRAM and gather traffic of the
complex layout (`force_complex` restores the complex baseline). Optionally the
three Lanczos work vectors can be stored in FP16 (`FTLM_FP16=1`; all arithmetic
stays FP32), halving the dominant random-gather traffic and the Krylov VRAM
again; this mode is not bit-identical to FP32 storage and is gated against an
accuracy envelope in the test suite.

> **A note on names:** many core functions carry an `_Ih` suffix
> (`ftlm_observables_pg_gpu_Ih`, `collect_clt_entries_Ih`,
> `build_entry_skeleton_Ih`, the CUDA kernel, …). The suffix is historical —
> the pipeline was first written for the icosahedral (I_h) polyhedra and
> later generalized. **The code behind these names is group-generic**: the
> same `_Ih` functions serve every geometry below, including arbitrary
> user-supplied permutation groups. The names are kept for continuity with
> the development history and existing runs/checkpoints.

## Requirements

- MATLAB (developed on R2025b) + **Parallel Computing Toolbox**
- A CUDA-capable NVIDIA GPU + matching CUDA toolkit (for `mexcuda`)
- A C++ compiler (`mex -setup C++`) for the schnack lookup MEX
- Reference hardware: RTX 4000 SFF Ada (20.5 GB VRAM), 63 GB host RAM

## Build & test

```matlab
build_all          % compiles the CUDA kernel + the schnack MEX (toolchain checks)
setup_paths        % adds tests/ examples/ docs/ to the path
run_all_tests      % aggregated regression suite (providers + ED + GPU kernel)
```

## Quickstart

```matlab
setup_paths
% N=12 triangular cluster, full ED cross-check vs the symmetry decomposition:
validate_triangular12
% A GPU FTLM run from an example input (M=0 only):
ftlm_observables_pg_gpu_Ih('input_triangular16_M0.m')
% Physical thermodynamics (full M sweep -> chi(T) != 0):
ftlm_observables_pg_gpu_Ih('input_triangular16_full.m')
```

Each input file is a small script setting `geometry`, `s_val`, `J`, `R`, `M_lz`,
`T_range` and options — the complete parameter reference (every knob, default
and effect, including the environment switches `FTLM_FP16`, `FTLM_R3`, ...) is
[`docs/INPUT_REFERENCE.md`](docs/INPUT_REFERENCE.md); a guided
tour of the worked decks is [`examples/README.md`](examples/README.md).

## Capabilities

| `geometry` | Provider | Symmetry | Spin | Validation |
|---|---|---|---|---|
| `icosahedron` | `icosahedron_Ih_full` | I_h, N=12, \|G\|=120 | s=1/2 … 7/2 | ED + sum rule |
| `dodecahedron` | `dodecahedron_Ih_full` | I_h, N=20 | s=1/2, 3/2 | N=20 vs sparse Lanczos 1.4e-14 |
| `icosidodecahedron` | `icosidodecahedron_Ih_full` | I_h, N=30 | s=1/2 | sum rule exact |
| `cuboctahedron` | `cuboctahedron_Oh` + `irreps_from_group` | O_h, N=12, \|G\|=48 | s=1/2 | N=12 full ED |
| `square_lattice` | `square_lattice_translation_group` | C_Lx×C_Ly (abelian) | s=1/2 | 4×4 ED; 5×6 full-M |
| `square_lattice_c4v` | `square_lattice_spacegroup` + `irreps_from_group` | C_4v / C_2v | s=1/2 | 4×4 ED 2.1e-13 |
| `kagome` | `kagome_spacegroup` + `irreps_from_group` | C_6v / C_2v | s=1/2 | N=12 ED 1.1e-13 |
| `triangular` | `triangular_spacegroup` + `irreps_from_group` | C_6v / C_6 | s=1/2 | N=12 ED 4.9e-14 |
| `generators` | `group_from_generators` + `irreps_from_group` | **any user permutation group** | any | 4×4 C_4v from generators == native == full ED |

**M-sector control:** `M_sectors = [...]` (or `only_M0 = true`) restricts to chosen
`|M|` sectors; unset = full sweep `0..M_max` → physical `C(T)` and `chi(T)`.

The full validation matrix — which reference method backs which system, with
the achieved agreement — is [`docs/VALIDATION.md`](docs/VALIDATION.md).

## Bring your own system (custom topology + symmetry)

Any spin cluster whose symmetry you can write down as site permutations runs
WITHOUT touching the code: set `geometry = 'generators'` and supply only the
permutation **generators** plus the bond list —

```matlab
geometry = 'generators';
gens  = { [2 3 4 1  6 7 8 5], ...  % each: 1xN permutation, P(i) = image of site i
          [1 4 3 2  5 8 7 6] };    % generators only -- the group is closed for you
bonds = [1 2; 2 3; 3 4; 4 1];      % [n_bonds x 2], 1-based, same site labelling
point_group = 'my C_4';            % optional label
sys_name    = 'myring8';           % optional tag for output files
s_val = 0.5;  J = 1.0;  R = 8;  M_lz = 60;  T_range = logspace(-1, 1, 40);
```

The pipeline closes the generators into the full group (Dimino), computes **all
irreps generically**, realifies the FS=+1 ones (→ the real FP32 GPU kernel),
and runs the identical machinery as the built-in geometries. Complete worked
example: [`examples/input_generators_square4x4_ED.m`](examples/input_generators_square4x4_ED.m)
(the 4×4 C_4v space group, order 128, from four hand-written generators —
suite-validated against the native provider AND full ED, bit for bit).

Startup guards close the two silent-wrongness traps: the group is a group by
construction (closure), and the **bond list is asserted invariant under every
group element** — this check matters because the FTLM sum rule is a trace
identity that would NOT catch a wrong bond list.

**Recommended workflow for a new system:**
1. Start small with `ed_thresh = inf`: every (M, Γ) block is exactly
   diagonalised — cross-check `C(T)` against an independent full ED (pattern:
   `tests/validate_generators_square4x4.m`, or `ed_full_heisenberg` directly).
2. Check the printed **sum rule** (must equal the Hilbert-space dimension to
   machine precision) and the pre-flight `estimate_feasibility` report.
3. Then scale up: GPU FTLM, `only_M0` or the full M sweep, spin-flip Z2
   (`use_spin_flip`, M=0), and the out-of-core options as needed.

Limits for user groups: `|G| <= 65535`, irrep `d <= 12`, `n_reps < 2^31` per
(M, Γ) block; uniform nearest-neighbour Heisenberg (one scalar `J`). Practical
size for the generic irrep builder: `|G|` up to a few hundred.

## Hamiltonian & known limits

- **Hamiltonian:** uniform isotropic nearest-neighbour Heisenberg only
  (`H = J * sum_<ij> S_i·S_j`). No anisotropy / DM / further-range / single-ion.
- **`s > 1/2`** is implemented (uint8 c-index path) but only **validated on I_h** —
  not yet ED-checked on a lattice.
- **Kernel hard caps:** irrep dimension `d <= MAX_D = 12` (covers every shipped
  geometry incl. the triangular `(6,0)` N=36 `d=12` irrep); group order
  `|G| <= 65535` (uint16 g-index); `n_reps < 2^31` per (M, Γ) block — basis
  OFFSETS are 64-bit, so `n_basis = d*n_reps` beyond 2^31 is supported. A larger
  case is **refused at a pre-flight feasibility guard** (`estimate_feasibility`).
- **Performance:** the SpMV is memory-bound on the random gather. Only the Lanczos
  work vectors must reside in VRAM; the entry table tiers automatically
  (device-resident -> chunked -> host-streamed -> disk-mapped); the block width
  B is chosen at run time from free device memory (power of two, gather-aware
  cap), from 4 GB workstation cards up to 192 GB B200-class GPUs; disk-mapped
  tables can use a pinned-ring double-buffered streamer (`FTLM_R3=1`). Demonstrated envelope:
  kagome N=36 M=0 on a 20 GB workstation card; a full M sweep of the s=3/2
  dodecahedron (n_total ≈ 1.1e12, largest Lanczos vector 3.55e9 elements = 14.2 GB)
  and the 5×8 square lattice M=0 sector (1.4e11) on one 192 GB B200.
- **Full limits & scaling guide:** see [`docs/LIMITS.md`](docs/LIMITS.md) — hard caps,
  memory tiers, worked feasibility examples, and where the single-GPU design ends
  (kagome N=42 case study).

## Architecture (why new lattices are cheap)

All heavy lifting is in **one** CUDA file, `cuda_lanczos_clut_block_pg_Ih.cu` (~6
active device kernels: the on-the-fly SpMV + block-Lanczos vector ops), plus one
CPU MEX (`schnack_query_mex.cpp`) for the combinatorial-ranking state lookup used
when `n_total > 2^32`. Everything else — geometry, spin, symmetry (irrep dimension
`d`), the bitmap-vs-schnack lookup, and the B2 / streaming out-of-core machinery —
is handled in MATLAB and fed to that one kernel as data. **Adding a lattice needs no
kernel change** (unless `d > 12` or `|G| > 65535`). See
[`docs/adding_a_geometry.md`](docs/adding_a_geometry.md) for the provider contract.

## Repository layout

```
*.m, *.cu, *.cpp   core: drivers, providers, kernels, collect/enumerate, lookup
build_all.m        compile the kernel + MEX
run_all_tests.m    regression suite          setup_paths.m   path setup
tests/             test_* / validate_* regression suite
examples/          input_*.m driver inputs (guided tour: examples/README.md)
docs/              input reference, validation matrix, limits/scaling guide, provider contract
```

## License & authorship

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universität Dresden, and
Helmholtz-Zentrum Dresden-Rossendorf e.V.
