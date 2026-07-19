# Examples — a guided tour

Every file here is a complete input deck: run it with either driver,

```matlab
setup_paths
ftlm_observables_pg_Ih('input_kagome12_ED.m')        % CPU FP64 (ED / FTLM)
ftlm_observables_pg_gpu_Ih('input_kagome12_gpu.m')   % mixed GPU FP32 / CPU FP64
```

Parameter reference: [`docs/INPUT_REFERENCE.md`](../docs/INPUT_REFERENCE.md).
What to look at after **every** run: the banner (system / |G| / irrep count),
and the final **sum-rule line** — `rel.err` must be ~1e-16 (an exact
bookkeeping identity; anything larger means something is wrong).

## Start here

| Deck | What it shows | Cost |
|---|---|---|
| `input_kagome12_ED.m` | CPU, exact ED of every (M, Γ) block on the N=12 kagome torus — the pattern used by all `validate_*` tests | seconds |
| `input_square_4x4_c4v_gpu.m` | first GPU run: 4×4 square, full C_4v space group, full M sweep, mixed ED/FTLM (`ed_thresh = 50`) | seconds |
| `input_triangular16_full.m` | physical thermodynamics: full M sweep → C(T) **and** χ(T) on the N=16 triangular torus | ~a minute |
| `input_dodecahedron20_M0.m` | all 10 I_h irreps of the N=20 dodecahedron at M=0 on the GPU (dim(M=0) = 184756) | ~a minute |

## ED reference decks (`ed_thresh = inf`, CPU driver)

Exact diagonalization of every block — deterministic, no FTLM statistics.
These are the decks the validation suite runs against independent full ED
(see [`docs/VALIDATION.md`](../docs/VALIDATION.md) for the achieved
agreements):

`input_kagome12_ED.m`, `input_triangular12_ED.m`,
`input_cuboctahedron12_ED.m`, `input_square_4x4_c4v_ED.m`,
`input_square_4x4_s12_ED.m` (translation-only), `input_dodecahedron20_ED.m`
(M=0), `input_generators_square4x4_ED.m` (the **bring-your-own-generators**
worked example — C_4v from four hand-written permutations),
`input_icosahedron_pg_example.m`, `input_ring_pg_example.m`.

## Physics runs (FTLM, GPU driver)

- Full M sweep (→ C(T) and χ(T)): `input_triangular16_full.m`,
  `input_square_4x4_c4v_gpu.m`, `input_icosidodecahedron_s12_full.m`
  (N=30, hours) and its `_R24` variant (more FTLM statistics),
  `input_square_5x6_s12_allM.m` (N=30).
- M=0 only (spectra / C(T) studies at a fraction of the cost):
  `input_dodecahedron20_M0.m`, `input_icosidodecahedron_s12_M0.m`,
  `input_square_5x6_c4v_M0.m`, `input_triangular16_M0.m`.
- Higher local spin: `input_icosahedron_s1.m`, `input_icosahedron_s3o2.m`,
  `input_icosahedron_s2_M0.m`, `input_dodecahedron20_s1_M0.m`.

## Feature demos (one knob each)

| Deck | Feature |
|---|---|
| `input_triangular16_M2.m` | `M_sectors` — run a single \|M\| sector |
| `input_icosahedron_s2_M0_Hg.m` | `irrep_list` — restrict to one irrep |
| `input_square_4x4_s12_ckpt.m`, `input_square_4x4_c4v_ckpt.m`, `input_square_5x4_s12_ckpt.m` | `checkpoint` — per-M / per-irrep checkpoint + resume |
| `input_square_4x4_c4v_pccache.m` | `precompute_cache` — persist enumerate+collect across reruns |
| `input_kagome12_gpu.m` | minimal GPU kernel smoke test |
| `input_ring_24_s12.m` | plain spin ring (translation group) |

## At-scale showcases (N = 36, 20-GB-class GPU)

These need the Schnack lookup (`n_total > 2^32`) and, depending on VRAM,
the streaming machinery — read each deck's header comment first:

- `input_kagome36_M0.m` / `_bounded` / `_short` / `_stream` — N=36 kagome
  M=0 over the full C_6v space group; the `_stream` variant runs the d≥3
  blocks by streaming the entry table in rep-tiles per SpMV.
- `input_square_6x6_c4v_M0.m` / `_first6` and `input_square_6x6_s12_M0.m`
  — N=36 square lattice at M=0.
- `input_triangular36_M0.m` — N=36 triangular `(6,0)`, includes a d=12
  irrep (the kernel `MAX_D` showcase).

**Deliberately kept as a negative example:** `input_triangular28_M0.m` — its
`(4,2)` supercell is **chiral**, so the provider drops the mirror and runs
with C_6 instead of C_6v (it warns). Kept to document that behaviour; do not
use it as a C_6v reference.

## Paper production decks

| Deck | System |
|---|---|
| `input_icosahedron_s32_R100.m` | icosahedron s=3/2, full physics run (R=100) |
| `input_cuboctahedron_s32_R100.m` | cuboctahedron s=3/2, full physics run (R=100) |
| `input_dodecahedron_s32_perM.m` | dodecahedron s=3/2 campaign, one M sector per job (heavy!) |
| `input_square_5x8_M0.m` | 5x8 square lattice s=1/2, M=0 sector (space group) |

The cross-platform benchmark decks of the paper (Table 4) live in the
repository root as `bench_fp32n_*_in.m` (FP32 default) and `bench_fp16_*_in.m`
(same decks, run with `FTLM_FP16=1`).
