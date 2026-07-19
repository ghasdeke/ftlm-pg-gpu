# Input-deck reference

An input deck is a **plain MATLAB script** (see `examples/input_*.m`). The
driver executes it with `run(input_file)` and reads the variables it defines;
anything not set falls back to the defaults below. Two drivers consume the
same deck format:

- `ftlm_observables_pg_Ih('deck.m')` — CPU FP64 (FTLM or dense ED per block),
- `ftlm_observables_pg_gpu_Ih('deck.m')` — mixed GPU FP32 / CPU FP64.

Both share the core parameters; the GPU driver additionally reads the
GPU/memory/caching knobs in the last section (the CPU driver silently ignores
them). Every default in this file was read off the driver source
(`ftlm_observables_pg_gpu_Ih.m` / `ftlm_observables_pg_Ih.m`), not off the
examples. A minimal deck:

```matlab
geometry = 'triangular';  tri_a = 2;  tri_b = 2;      % N = 12
s_val = 0.5;  J = 1.0;  R = 8;  M_lz = 60;
T_range = logspace(-1, 1, 40);
only_M0 = true;
```

## Required parameters (both drivers)

| Name | Type / constraint | Meaning |
|---|---|---|
| `s_val` | positive half-integer (0.5, 1, 1.5, …) | Local spin `s`. Local dimension is `d_loc = 2s+1`; the Hilbert space is `d_loc^N`. |
| `J` | finite real scalar | Uniform nearest-neighbour Heisenberg exchange, `H = J * sum_<ij> S_i·S_j`. `J > 0` is antiferromagnetic. |
| `R` | integer ≥ 1 | FTLM random vectors per (M, Γ) block (clamped to the block dimension). Statistical accuracy improves with `R`; enumerate/collect cost is R-independent. |
| `M_lz` | integer ≥ 1 | Lanczos iterations per random vector (clamped to the block dimension). Kernel time is linear in `M_lz`; use `lz_diag` to tune it from data. |
| `T_range` | vector of positive finite values | Temperature grid (units of `J/k_B`) for `C(T)`, `chi(T)`, `Z_eff(T)`. |

A missing required variable is a hard error naming the variable.

## Geometry selection

| Name | Default | Meaning |
|---|---|---|
| `geometry` | `'icosahedron'` | Which provider supplies (group, bonds, N). One of the nine cases below (case-insensitive). |

| `geometry` value | System | Symmetry | Extra deck variables |
|---|---|---|---|
| `'icosahedron'` | N=12 icosahedron | I_h, \|G\|=120, named irreps | — |
| `'dodecahedron'` | N=20 dodecahedron | I_h, \|G\|=120, named irreps | — |
| `'icosidodecahedron'` | N=30 icosidodecahedron | I_h, \|G\|=120, named irreps | — |
| `'cuboctahedron'` | N=12 cuboctahedron | O_h, \|G\|=48, generic irreps | — |
| `'square_lattice'` | Lx×Ly periodic square lattice | translations C_Lx × C_Ly (abelian) | `Lx`, `Ly` (required) |
| `'square_lattice_c4v'` | Lx×Ly periodic square lattice | full space group (point group C_4v if Lx==Ly, else C_2v) | `Lx`, `Ly` (required) |
| `'kagome'` | kagome torus, supercell T1=(a,b), T2=R6·T1 | C_6v space group for non-chiral supercells (b=0 or a=b); chiral (a,b) drop the mirror → C_6 (provider warns) | `kag_a`, `kag_b` (required) |
| `'triangular'` | triangular torus, supercell T1=(a,b), T2=R6·T1 | C_6v space group (non-chiral supercells, as for kagome) | `tri_a`, `tri_b` (required) |
| `'generators'` | **user-defined** spin cluster | closure of user permutation generators | `gens`, `bonds` (required); `point_group`, `sys_name` (optional) |

### `generators` sub-parameters

| Name | Default | Meaning |
|---|---|---|
| `gens` | — (required) | Permutation generators of the symmetry group: cell array (or rows of a matrix) of 1×N site permutations, `P(i)` = image of site `i`, 1-based. The group is closed for you (Dimino); irreps are computed generically and FS=+1 irreps realified. |
| `bonds` | — (required) | `[n_bonds × 2]` nearest-neighbour bond list, 1-based, same site labelling as `gens`. Geometry is deliberately separate from symmetry. **Asserted invariant under every group element at startup** — a non-invariant bond list would give silently wrong physics (the sum rule would not catch it). |
| `point_group` | `'custom'` | Label used in printouts only. |
| `sys_name` | `'custom'` | Tag used in output/cache/checkpoint filenames (`geo_tag`). Set it when running several generator systems in one directory. |

Practical limits: irrep dimension `d ≤ 12` (kernel `MAX_D`) and `|G| ≤ 65535`
(uint16 g-index) are asserted at pre-flight / collect. The per-block rep count
is capped at `2^31 − 257` (int32 rep index) — enforced in the GPU kernel MEX
at block initialization, i.e. **after** the potentially long enumerate/collect,
so for a large run check the `n_reps` line of the `estimate_feasibility`
pre-flight report yourself. See “Bring your own system” in the README for the
recommended validation workflow.

## Sector and irrep control (both drivers)

| Name | Default | Meaning |
|---|---|---|
| `M_sectors` | `[]` (= full sweep) | Vector of non-negative \|M\| sectors to run, e.g. `0`, `[0 1 2]`. The ±M mirror is folded in via a multiplicity factor, so list only non-negative M. **Overrides `only_M0`.** Default `[]` runs the full sweep `0..M_max` → physical `C(T)` and `chi(T)`; a subset gives that subset's partial result (the sum rule then checks against dim of the covered sectors). |
| `only_M0` | `false` | Shortcut for `M_sectors = 0`. M=0 contains every S multiplet exactly once, so spectra/heat-capacity studies often start here at a fraction of the cost; `chi(T)` needs the full sweep. |
| `irrep_list` | all irreps | Cell array of irrep names to include. I_h named irreps: `'A_g'`, `'A_u'`, `'T1g'`, `'T1u'`, `'T2g'`, `'T2u'`, `'F_g'`, `'F_u'`, `'H_g'`, `'H_u'`. Generic providers (cuboctahedron, space groups, generators) name irreps `d<d>_<j>` (dimension + discovery index, e.g. `'d3_9'`, `'d6_14'`) — run once and read the names off the banner. Any unknown name is a hard error (the message lists the known names). With `use_spin_flip`, list **base** names (the ± doubling happens afterwards). |
| `use_spin_flip` | `false` | Spin-flip Z2 extension at **M=0 only**: G → G × Z2 by the spin inversion `Sz → −Sz`, splitting each Γ into Γ±. Roughly halves `n_reps` / `n_entries` / block sizes in the (dominant) M=0 sector. M>0 sectors are unaffected. |
| `ed_thresh` | `0` | Blocks with `n_basis <= ed_thresh` are densely diagonalized (deterministic) instead of FTLM. `0` = FTLM throughout; `inf` = exact ED everywhere (the validation reference); a small value (e.g. `50`) catches tiny blocks where FTLM statistics would be wasteful. |
| `force_complex` | `false` | Keep the I_h named irreps in their historical complex form and use the complex-GPU layout even for real irreps. Only for reproducing pre-realification baselines byte-for-byte or A/B-benchmarking the real vs complex kernel; leave off otherwise (realified irreps → real FP32 kernel path: ~half the VRAM and gather traffic). |

## Output (both drivers)

| Name | Default | Meaning |
|---|---|---|
| `output_dir` | `'.'` | Created up front by both drivers. Receives the result `.mat` (`ftlm_pg_gpu_Ih_<geo>_s<s>.mat` / `ftlm_pg_Ih_<geo>_s<s>.mat`; `s` as `1o2`-style for half-integers), the checkpoint, the precompute cache and the out-of-core scratch (the latter two move to `precompute_dir` only when that is set **and** `precompute_cache` is on — see below). |

The GPU result file contains `T_range, C_T, chi_T, Z_eff`, the run
configuration, the per-sector diagnostics (`sector_M/G/dims/path`) and the raw
Ritz data (`all_E, all_w, all_M`). The CPU result file carries the same
observables, configuration and `sector_M/G/dims`, but **not** `sector_path` or
the raw Ritz data — those are available through the CPU driver's optional
return struct (`out = ftlm_observables_pg_Ih(...)`). Every run ends with a
**sum-rule check** (`sum_i w_i` against the covered Hilbert-space dimension)
printed as `rel.err` — expect ~1e-16; treat anything larger as a red flag.

## GPU driver only

Read by `ftlm_observables_pg_gpu_Ih`; ignored by the CPU driver.

### Kernel / dispatch

| Name | Default | Meaning |
|---|---|---|
| `B_gpu` | `0` | Block size of the block-Lanczos kernel. `0` = adaptive: `min(R_eff, MAX_B=8, VRAM-fit, gather-cap)`, rounded down to a power of two (gather coalescing). Under `FTLM_FP16=1` the per-column footprint shrinks, so a larger B may fit. An explicit value forces `min(B_gpu, R_eff, 8, VRAM-fit)` and bypasses the gather cap (benchmarking). |
| `skip_feasibility` | `false` | Skip the **entire** pre-flight `estimate_feasibility` call — both the host-RAM gate and the informative report. The kernel hard caps (`d ≤ 12`, `|G| ≤ 65535`) are asserted **unconditionally**, regardless of this flag. |

### Entry-table pipeline (memory levers)

Policy: resident-when-fits is the fast path — reach for these levers per
system when the pre-flight says you must, not by default.

| Name | Default | Meaning |
|---|---|---|
| `lookup_method` | `'bitmap'` | State → super-rep index backend used during collect. `'bitmap'`: 32-state bitmap, `n_total/32*8` bytes, requires `n_total ≤ 2^32`. `'schnack'`: combinatorial ranking (Schnack/Hage/Schmidt 2007), few-KB table + sorted per-rep ranks — **mandatory for `n_total > 2^32`** (e.g. N=36 at s=1/2, dodecahedron at s=3/2). |
| `entries_storage` | `'host'` | Where collect keeps its working set. `'host'` is the production path — **prefer it**. `'gpu'`: stores the collected chunks as gpuArrays (VRAM-heavy; historically inflated MATLAB's pinned host pool, so host RAM can go **up**, not down; silently bypassed when `entries_on_disk=true`). `'gpu_native'`: fully GPU-native collect (`collect_clt_entries_Ih_gpu`) — a **deprecated experiment kept for reference** (measured +~5 GB host RAM per sector); hard-wires the Schnack lookup (`lookup_method` ignored), incompatible with `entries_on_disk`, disables `precompute_cache`. |
| `entries_on_disk` | `false` | Out-of-core entry table: collect streams entries to a bucket-sorted file (`ondisk_*/` under `output_dir`, or under `precompute_dir` with the cache), and the SpMV memory-maps it (needs the `mmap_file` MEX from `build_all`). A resident-prefix is pinned in leftover VRAM automatically; only the tail streams per SpMV (bit-identical either way). For tables that exceed host RAM (dodecahedron s=3/2: ~86 GB). Put `output_dir` on fast NVMe. |
| `ondisk_scratch_dir` | `''` (-> `output_dir`) | Optional node-local scratch directory for the out-of-core entry files (`ondisk_*/`). The finalize transiently holds ~2x the table on disk, so putting the scratch on fast local NVMe while results/checkpoints stay on a shared filesystem avoids quota pressure in per-M job farms. Ignored when the precompute cache owns the on-disk files. |

### Precompute cache / restart

| Name | Default | Meaning |
|---|---|---|
| `precompute_cache` | `false` | Persist the per-M enumerate+collect results (`precompute_<geo>_s<s>_M<M>[_sf].mat`; the `_sf` suffix is the spin-flip M=0 cache, kept side by side with the base one). They depend only on (geometry, `s_val`, M, `J`, `use_spin_flip`) — **not** on `R`/`M_lz`/irreps — so reruns and R-upgrades skip the minutes-long precompute (a `J` change recomputes). Config-stamped with bond+permutation checksums (any provider change → recompute). Kept on disk after the run. |
| `precompute_dir` | `''` (→ `output_dir`) | Separate directory for the precompute cache (and its on-disk entry files). Lets several sector-parallel workers share one cache/entry table (see `ftlm_orchestrate_sectors`), or a new run reuse an old run's cache. |
| `precompute_only` | `false` | Phase-0 mode: run only enumerate+collect for every selected M, write the caches, then return — no FTLM, no result `.mat`. Requires `precompute_cache = true`. Used by the orchestrator to serialize cache misses. |
| `checkpoint` | `false` | Per-irrep checkpoint/resume (`ckpt_ftlm_gpu_<geo>_s<s>.mat` in `output_dir`): after every (M, Γ) block the accumulated spectrum is written atomically, so a crash costs at most one block (plus that M's precompute, unless cached). Config-stamped (mismatch → fresh start); deleted on successful completion. Essential for multi-hour runs. |

### Diagnostics

| Name | Default | Meaning |
|---|---|---|
| `mem_diag` | `false` | Print GPU/host memory snapshots at every pipeline stage. First run at a new scale: turn it on and hold the numbers against the pre-flight budget. |
| `lz_diag` | `false` | Per-block Lanczos convergence report (Ritz residual counts, β floor, E0 residual, unconverged weight). Print-only, results byte-identical, and **not** part of the checkpoint config stamp — safe to toggle across a resume. Use it to tune `M_lz`/`R` from data. |

## Environment switches (GPU kernel)

Read via `getenv` at kernel/driver level; all default OFF/empty.

| Variable | Effect |
|---|---|
| `FTLM_FP16=1` | Store the three Lanczos work vectors in FP16 (real-irrep path only); ALL arithmetic stays FP32 (load-convert / float accumulate / convert-store). Halves the dominant random gather traffic and the Krylov VRAM (doubles the feasible B). NOT bit-identical to FP32 storage; gated in the suite against an accuracy envelope across all memory tiers (`test_fp16_smoke`, `test_device_sizing_matrix`). Not available for the chunked-V0 path (n_basis > 2^31). |
| `FTLM_FP16_SCALE=0` | Disable the per-block power-of-two storage scale of the FP16 mode. By default the stored vectors are scaled by 2^k, k = round(log2 sqrt(n_basis)), which keeps the components of normalized Lanczos vectors on the NORMAL fp16 grid for every block size (for n_basis > 2^28 the typical component 1/sqrt(n_basis) would otherwise fall into the subnormal range, where the absolute quantum 2^-24 makes the effective storage roundoff grow like sqrt(n_basis)*2^-24). The scale factors are exact powers of two, applied as un-contractible multiplies, so `FTLM_FP16_SCALE=0` reproduces the unscaled fp16 grid bit-exactly. Diagnostics print the per-block exponent. |
| `FTLM_R3=1` | Pinned-ring double-buffered streaming for disk-mapped (out-of-core) entry tables: a CPU worker stages pages into two pinned ring buffers, the copy stream DMAs at full PCIe speed, the compute stream overlaps. Bit-identical to the synchronous path (`test_r3_stream`). Tuning: `FTLM_R3_THREADS=1..8` (default `min(8, cores/2)`); `FTLM_R3_TIMEOUT=<s>` stall guard per tile (default 60 s). |
| `FTLM_LEVER_A=1` | Double-buffered streaming for host-RAM (non-mmap) tables: pins the streamed tail once, then overlaps H2D copy and compute. Bit-identical (`test_stream_prefix`). |
| `FTLM_EBS_PAR` | Parallel finalize of the on-disk entry table (external bucket sort): unset/`auto` = parallel from 4 GB table size when the Parallel Computing Toolbox is present; `0` = force serial; `N` = N workers. Byte-identical output either way. |
| `FTLM_D_TEMPLATES=1` | Opt-in d-templated SpMV kernel instantiations (default OFF after a measured regression on Ada-class GPUs); results identical. |

Test hooks used by the suite (not for production): `FTLM_FAKE_FREE_VRAM_GB`,
`FTLM_DEBUG_DROP_PREFIX`, `FTLM_EBS_FAIL_BUCKET`.

Experimental deck options (default off; may change the class-C summation
order, the sum rule stays exact): `prefix_budget_gb`, `esort_src`,
`tiled_spmv`, `tiled_window_gb` -- see the driver source comments.

## Reproducibility

FTLM random vectors use fixed per-block seeds derived from (M, irrep index),
so repeated runs of the same deck are deterministic on the same code/path.
The CPU-FP64 and GPU-FP32 drivers use different seed bases (and different
arithmetic), so their FTLM estimates agree statistically, not bitwise;
`ed_thresh = inf` makes both exact and comparable to machine precision.
