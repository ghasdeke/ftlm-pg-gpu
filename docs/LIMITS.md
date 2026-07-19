# System-size limits and scaling envelope

This page documents what the code can and cannot compute, which resource binds
first as systems grow, and how to estimate feasibility for a new system before
running anything. All limits below are enforced by explicit guards that fail
early with an actionable message (`estimate_feasibility` pre-flight, per-site
asserts); none fail silently.

## 1. Hard structural caps

| Quantity | Cap | Origin | When exceeded |
|---|---|---|---|
| Representatives per M sector, `n_reps` | `2^31 − 257` | rep indices are int32 in the CUDA kernel (entry lists, thread ids) | refused at pre-flight; needs an index-widening code change |
| Block dimension `n_basis = Σ_r n_Γ(r)` | ≈ `2.6×10^10` (= d·n_reps cap) | basis *offsets* are int64; `n_basis > 2^31` is supported (start vectors upload in chunks, `B = 1`) | — |
| Irrep dimension `d_Γ` | 12 (`MAX_D`) | fixed kernel register layout | refused at init |
| Group order `\|G\|` | 65 535 | uint16 group-element index per entry | refused at build |
| Lanczos batch width `B` | 8 (`MAX_B`) | kernel register budget | clamped automatically |
| Sites `N` | ≤ 63 for s = 1/2 (int64 state labels); `(2s+1)^N < 2^63` in general | integer state encoding | refused at setup |
| Local dimension `2s+1` | ≤ 16 | digit encoding | refused at setup |
| Full Hilbert dimension `n_total` | `≤ 2^32` for the **bitmap** lookup and for the dense-ED branch (`ed_thresh > 0` on a full M sweep); unlimited (within `N`) for the **ranking** lookup | bitmap memory is linear in `n_total` (~0.25 B/state) | choose `lookup_method='schnack'`, set `ed_thresh = 0` |

The single most important cap in practice is **`n_reps < 2^31` per M sector**:
it is what ultimately bounds the reachable system size, independently of any
amount of memory (see §4).

## 2. What must fit where

Only the **Lanczos work vectors** must reside in GPU memory:

```
VRAM (per (M,Γ) block):  4 × n_basis × B × 4 B   (real irreps)
                         8 × n_basis × B × 4 B   (complex irreps)
```

Everything else tiers automatically, gated per irrep and verified bit-identical
to the resident path in the test suite:

1. **device-resident** entry table (fastest; small/medium blocks),
2. **device-resident via chunked upload** (`n_entries > 2^31`),
3. **host-streamed**: table stays in host RAM, streamed to the GPU in
   representative tiles per SpMV, with a VRAM-resident leading prefix sized to
   the free device memory,
4. **disk-mapped**: table collected and sorted on disk (external bucket sort,
   6–7 B/entry), memory-mapped, same tile streaming (`entries_on_disk = true`).

Host RAM during preparation needs roughly `12 + N` bytes per representative
(rep list + int8 digit matrix) plus bounded transients; the sort runs on disk.
Entry-table size ≈ `n_reps × N_B/2 × 6–7 B` for s = 1/2 (more flips per
representative for s > 1/2).

## 3. Demonstrated envelope

| Hardware | Demonstrated | Largest single Lanczos vector |
|---|---|---|
| 20 GB workstation (RTX 4000 SFF Ada) | icosidodecahedron s=1/2 full-M; kagome N=36 M=0 (streamed, 17.6/20 GB) | ~1.3×10^9 elements |
| 1× NVIDIA B200 (192 GB) + 2.3 TB host | dodecahedron s=3/2 **full M sweep** (n_total = 4^20 ≈ 1.1×10^12, on-disk tables up to ~175 GB); 5×8 square lattice M=0 (D = 1.4×10^11) | **3.55×10^9 elements = 14.2 GB** (dodec s=3/2, M=1, H_g; above 2^31, chunked-upload path) |

Rule of thumb for a 192 GB card: blocks up to `n_basis ≈ 10^10` fit
(4 buffers × 4 B at B = 1 ≈ 160 GB); the entry table does not bind (streamed
from host/disk); `n_reps` per M sector must stay below 2.1×10^9.

## 4. A boundary case: kagome N = 42

The N = 42 kagome torus (14 unit cells, PBC) illustrates where the current
single-GPU design ends — for an instructive reason. A hexagonal torus admits
C₃/C₆ rotations only if its cell count is a Löschian number (a²+ab+b²: 1, 3,
4, 7, 9, 12, 13, 16, …); **14 is not**, so the largest available space group
is only ~T₁₄⋊C₂ᵥ (|G| ≤ 56, ×2 with the M = 0 spin flip). Consequently:

- `n_reps(M=0) ≈ C(42,21)/112 ≈ 4.8×10^9` — **exceeds the int32
  representative cap** (§1) by ~2.2×, at M = 1 by ~4.3×. Blocked today,
  independent of memory; would need 64-bit rep indices (a known, contained
  kernel/pipeline change).
- After such a widening, the d = 4 blocks would need ~4 × 77 GB of Lanczos
  buffers — **beyond one 192 GB GPU**; d ≤ 2 blocks would fit. This is the
  multi-GPU regime (basis partitioned along the representative axis).
- The M = 0 entry table would be ~1.2 TB (fits host RAM/NVMe; each SpMV then
  streams ~1 TB, making the iteration transfer-bound).

In short: N ≈ 40 spin-1/2 sites with a *large* symmetry group (e.g. the 5×8
square lattice, |G| = 160) fit a single B200; N = 42 kagome, with its small
torus group, does not.

## 5. Feasibility check before running

`estimate_feasibility` runs automatically at driver start and refuses
early (with the binding resource named) rather than failing hours in. To check
a planned system without computing anything, run the driver with a deck whose
`M_sectors` selects one small sector, or call `estimate_feasibility` directly;
group order, irrep dimensions, `n_reps`, and per-block `n_basis` are printed
for every M before any GPU work starts.
