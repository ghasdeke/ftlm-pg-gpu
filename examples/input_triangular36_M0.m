%% input_triangular36_M0.m
%  *** RUNNABLE since the 2026-06-06 kernel bump (#define MAX_D 12): the (6,0)
%  space group has a d=12 irrep (dims [1x4,2x4,3x4,4x1,6x6,12x1], 33% of the
%  M=0 weight). The OTF SpMV handles d=12 (register-light: it caches V_t[:,kp]
%  plus B accumulators per thread, not d*d). Earlier this input was refused at
%  the pre-flight feasibility guard under MAX_D=8; that cap is now 12. Re-check
%  -Xptxas -v occupancy before any further MAX_D increase. ***
%  ================================================================
%  Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
%  and Helmholtz-Zentrum Dresden-Rossendorf e.V.
%  Licensed under the Apache License, Version 2.0.
%  ================================================================
%  N=36 triangular torus on the C_6-symmetric (a,b)=(6,0) supercell (36 cells,
%  1 site/cell), s=1/2, M=0 only, full space group C_6v semidirect translations
%  (|G|=432). The canonical maximally-symmetric 36-site triangular cluster
%  (frustrated Heisenberg AFM -- 120-degree order / RVB).
%
%  dim(M=0) = C(36,18) = 9.08e9 ; n_reps ~ dim/432 ~ 2.1e7 (3x FEWER than the
%  N=36 kagome thanks to the 3x larger group) ; ~20 irreps with dims up to d=12
%  (needs kernel MAX_D >= 12, set 2026-06-06). compact-V auto-enabled (max d >= 2).
%
%  FEASIBILITY (estimated, anchored to the measured kagome N=36 run):
%    n_entries ~ 1.2e9  < 2^31 = 2.147e9  -> fits as ONE gpuArray, B2 likely
%      NOT needed (kagome's 2.3e9 forced B2; triangular's larger group avoids it).
%    Host entries table ~10-11 GB, host peak ~25-30 GB / 63 (kagome was ~52).
%    Per-irrep resident VRAM: table ~7 GB + Lanczos 9*n_reps*d*4 B
%      (d=1 ~0.8, d=6 ~4.5 GB) -> even the d=6 block fits ~11-12 GB / 20.5
%      RESIDENT (NO streaming-B2 needed; kagome had to stream d>=3).
%  ==> VERIFY with mem_diag=true and a small-irrep run BEFORE the full sweep.
%
%  n_total = 2^36 > 2^32 -> bitmap impossible, schnack lookup MANDATORY.
%  Per-irrep checkpointing (M=0-only long run; reboot-safe).
%  HEAVY RUN: do not launch without a
%  fresh feasibility check.
%  ================================================================

%% Required inputs
s_val   = 0.5;
J       = 1.0;
R       = 2;            % fast-feasibility config (matches kagome N=36); higher-R rerun later
M_lz    = 60;
T_range = logspace(-1, 1.5, 60);

%% Optional inputs
geometry        = 'triangular';
tri_a           = 6;
tri_b           = 0;
only_M0         = true;
ed_thresh       = 50;
mem_diag        = true;
lookup_method   = 'schnack';      % n_total = 2^36 > 2^32 -> mandatory
entries_storage = 'host';
checkpoint      = true;           % per-irrep checkpoint + resume
B_gpu           = 0;
