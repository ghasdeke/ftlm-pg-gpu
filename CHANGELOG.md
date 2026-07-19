# Changelog

Versioning follows [Semantic Versioning](https://semver.org); dates are ISO 8601.

## [1.0.0] — 2026-07-12

First public release.

- Symmetry-adapted finite-temperature Lanczos (FTLM) and exact diagonalization
  for the nearest-neighbour Heisenberg model, block-decomposed per total-S^z
  sector and per irreducible representation of a finite space/point group.
- Built-in geometries: icosahedron, dodecahedron, icosidodecahedron (I_h);
  cuboctahedron (O_h); square-lattice translation and C_4v/C_2v space groups;
  kagome and triangular C_6v space groups — plus a bring-your-own-symmetry
  front-end (`geometry = 'generators'`): user permutation generators are closed
  into the full group, irreps computed generically and realified, with runtime
  bond-invariance and closure guards.
- One group-generic CUDA SpMV/block-Lanczos kernel serving all systems (irrep
  dimension d <= 12, |G| <= 65535); real-FP32 fast path for realified irreps
  (about half the VRAM and gather traffic of the complex layout), optional FP16
  storage of the Lanczos vectors (`FTLM_FP16=1`, all arithmetic FP32),
  spin-flip Z2 extension at M = 0.
- Automatic memory tiering of the entry table: device-resident -> chunked ->
  host-streamed -> disk-mapped out-of-core with a VRAM-resident prefix;
  optional pinned-ring double-buffered streaming (`FTLM_R3=1`) and parallel
  external-bucket-sort finalize.
- Device-aware batch sizing (power-of-two block width from free VRAM,
  gather-aware cap), 64-bit basis offsets (blocks with n_basis > 2^31), GPU
  forward compatibility incl. Blackwell-class devices.
- Precompute cache, per-irrep checkpoint/resume, and sector-parallel multi-GPU
  orchestration with a shared on-disk entry table.
- CPU-FP64 and mixed-GPU drivers consuming the same input decks; complete
  parameter reference (`docs/INPUT_REFERENCE.md`), validation matrix
  (`docs/VALIDATION.md`), limits/scaling guide (`docs/LIMITS.md`), provider
  contract (`docs/adding_a_geometry.md`), worked example decks incl. the paper
  benchmark systems.
- Regression/validation suite (35 tests: per-geometry ED cross-checks,
  bit-identity gates for every memory/performance lever, FP16
  accuracy-envelope gate, emulated device-size sweep 4-180 GB) + CPU-only CI
  subset (`run_all_tests('ci')`).
