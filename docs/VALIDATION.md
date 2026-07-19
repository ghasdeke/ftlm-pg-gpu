# Validation matrix

What backs the correctness of this package, system by system. All numbers
below are from a fresh run of the shipped tests (2026-07-12, MATLAB R2025b,
RTX 4000 SFF Ada, suite `run_all_tests` **35/35 PASS**); they are
regenerated — not copied — whenever the suite runs, since every `validate_*`
entry asserts these checks internally.

Three layers of evidence:

1. **Exactness vs an independent reference** (this page's main table): the
   symmetry-adapted block decomposition, diagonalized exactly
   (`ed_thresh = inf`), is compared against a *symmetry-free* full ED or
   sparse Lanczos built by independent code.
2. **Bit-identity gates**: every memory/performance lever (real kernel,
   tiling, streaming, out-of-core, caching, keep-table, pinned-ring
   streaming) is gated to be bit-identical to the plain resident path —
   levers cannot silently change results. The one deliberate exception is
   FP16 vector storage, which is gated against an accuracy envelope instead
   (its default-off state IS bit-gated).
3. **The sum rule in every run**: both drivers always print
   `sum_i w_i` against the covered Hilbert-space dimension. This is an exact
   bookkeeping identity (independent of `R`/`M_lz`), so any symmetry-,
   multiplicity- or bookkeeping error at any scale shows up as a nonzero
   `rel.err` — including at production scales where no reference is
   computable.

## 1. System × reference × agreement

| System (test) | Symmetry | Reference method | Agreement (fresh run) |
|---|---|---|---|
| Square 4×4, s=1/2 (`validate_square_4x4_c4v`) | C_4v space group, \|G\|=128 | independent full ED (dim 65536) | Σw = 65536 **exact** (rel.err 0.0); C(T) max rel. err **2.1e-13** |
| Square 4×4 from user generators (`validate_generators_square4x4`) | closure of 4 hand-written generators | native provider (structural) + full ED | closed group **set-identical** to the native C_4v provider; Σw = 65536 exact; C(T) **3.1e-13** |
| Kagome N=12, s=1/2 (`validate_kagome12`) | C_6v space group, \|G\|=48 | independent full ED (dim 4096) | Σw = 4096 exact; C(T) **1.4e-13** |
| Triangular N=12, s=1/2 (`validate_triangular12`) | C_6v space group, \|G\|=144 | independent full ED | Σw = 4096 exact; C(T) **3.7e-14** |
| Dodecahedron N=20, s=1/2 (`validate_dodecahedron20`) | I_h, \|G\|=120 | independent symmetry-free **sparse Lanczos** (Sz=0, dim 184756) | Σw = C(20,10) = 184756 exact; lowest 5 distinct levels max \|ΔE\| **1.2e-14**; character orthogonality 1.1e-13. E0 = **−9.72218535 J** (E0/bond = −0.324073), matching published results for this cluster |
| Cuboctahedron N=12, s=1/2 (`validate_cuboctahedron12`) | O_h, \|G\|=48 | independent full ED | Σw = 4096 exact; C(T) **2.0e-13**; E0 = **−5.444875 J** (frustrated, 8 corner-sharing triangles) |
| Spin-flip Z2, CPU path (`validate_spinflip12`): kagome N=12 s=1/2 **and** square 3×3 s=1 | G × Z2 at M=0 (\|G\| 48→96 / 72→144) | direct M=0-sector ED | aggregated spectrum **6.7e-15 / 8.1e-15**; Γ⁺ ∪ Γ⁻ == unsplit Γ block **2.1e-15 / 4.7e-15**; sum rules exact. The s=1 system exercises the non-power-of-two digit path |
| Spin-flip in the GPU driver (`validate_spinflip_driver12`), kagome N=12 | `use_spin_flip`, end-to-end | independent full ED | full M sweep: Σw = 4096 exact, C(T) **1.0e-13**; M=0 GPU-kernel FTLM: sum rule = dim(M=0) = 924, rel.err **2.5e-16** |
| Icosahedron / icosidodecahedron (I_h) | I_h, \|G\|=120 | ED cross-checks + the kernel regressions below; every large run's sum rule | see §3 for at-scale sum rules (N=30 full sweep exact) |

The abelian translation-only path has its own FTLM-vs-ED check with a plot
(`tests/validate_square_4x4.m`, not part of the suite).

Note on accuracy scales: the ~1e-13 C(T) agreements are CPU-FP64 ED vs ED —
they validate the **decomposition** (group, irreps, multiplicities, blocks).
FTLM production runs additionally carry FP32-GPU roundoff and statistical
(`R`, `M_lz`) error, which the sum rule and `lz_diag` monitor.

## 2. Bit-identity gates (suite, GPU)

Each lever is asserted **bit-identical** (ΔE = Δw = 0) against the plain
resident complex path on the same matrix:

| Gate (test) | What it pins down |
|---|---|
| `test_real_kernel` | real FP32 kernel == complex kernel at V_im = 0, over resident/B2/stream/prefix × packed × c-index × spin-flip |
| `test_pipeline_opts` | skeleton/OTF kernel path == reference CLT path |
| `test_b2` | B2 entry-tiling == resident |
| `test_stream`, `test_stream_s1` | streamed rep-tiles (s=1/2 and indexed-c s≥1) == resident |
| `test_stream_mmap`, `test_entries_on_disk` | out-of-core mmap streaming == in-RAM, unit and end-to-end |
| `test_stream_prefix` | resident-prefix pinning == full streaming |
| `test_precompute_cache` | cache hit == cache miss; stale stamps rejected |
| `test_gpu_int64_fallback` | Blackwell forward-compat digit paths (float / int64 / host) bit-identical |
| `test_external_bucket_sort`, `test_mmap_file` | external sort == in-RAM sort; file mapping round-trips |
| `test_fp16_smoke` | FP16 storage: default-env bit gate + accuracy envelope vs FP32 |
| `test_r3_stream` | pinned-ring mmap streaming == synchronous path (bit-identical, incl. prefix + fp16) |
| `test_device_sizing_matrix` | emulated 4-180 GB cards x FP32/FP16: B power-of-two + monotone, results bit-identical |
| `test_keep_table` | kept entry table across irreps == fresh inits (incl. drop + re-grow) |
| `test_perM_merge`, `test_split_v0`, `test_c4v_spmv` | per-M merge == full sweep; chunked V0 == direct; OTF SpMV == CPU sparse H |

## 3. Sum rule at production scale

The exact bookkeeping identity holds in the large runs (driver-printed,
from the run logs):

| Run | Covered dimension | rel. err |
|---|---|---|
| Icosidodecahedron N=30 s=1/2, **full M sweep**, R=24, GPU | 2^30 = 1 073 741 824 | **0.0** |
| Square 5×6 N=30 s=1/2, full M sweep, GPU | 2^30 | **0.0** |
| Icosidodecahedron N=30, M=0, R=8 | dim(M=0) ≈ 1.55e8 | 3.8e-16 |
| Kagome N=36, M=0, spin-flip, streamed table | dim(M=0) ≈ 9.075e9 | 4.2e-16 |

(A float-double accumulation over ~1e9 weights reproducing the dimension to
1e-16 requires every block dimension, irrep multiplicity and ±M fold to be
correct.)

## Reproducing this page

```matlab
setup_paths
run_all_tests          % 35 tests; all validate_* entries assert the above
validate_dodecahedron20   % individual test, prints its numbers
```
