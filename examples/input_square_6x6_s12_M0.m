%% input_square_6x6_s12_M0.m
%  ================================================================
%  Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
%  and Helmholtz-Zentrum Dresden-Rossendorf e.V.
%  Licensed under the Apache License, Version 2.0.
%  ================================================================
%  6 x 6 periodic square lattice, s = 1/2, M = 0 only, all 36 irreps of the
%  translation group C_6 x C_6. THIS IS THE N=36 TARGET.
%
%  !!! FEASIBILITY WARNING (run estimate_square_feasibility(6,6) first) !!!
%      dim(M=0) = C(36,18) = 9.075e9 ; n_reps ~ 2.52e8 ; n_entries ~ 9.3e9.
%      With translation-only symmetry (order 36, NOT I_h's 120) the M=0
%      sector does NOT fit the current RTX 4000 SFF Ada / 63 GB host:
%        - host entries table ~ 84 GB  (> 63 GB RAM)
%        - GPU skeleton/irrep ~ 47 GB  (> 20 GB VRAM)  -> needs B2 entry-tiling
%      M=0 is the LARGEST sector, so restricting to M=0 saves TIME but not
%      memory. Launching as-is would thrash through the ~20-30 min enumerate
%      and then fail in collect/skeleton. DO NOT run until either:
%        (a) B2 entry-tiling + host entry compaction (out-of-core) is added, or
%        (b) the square point group C_4v is included (order 36*8=288 -> ~8x
%            smaller; would fit), or
%        (c) a larger machine (>=128 GB host, >=48 GB VRAM) is used.
%
%  lookup_method MUST be 'schnack' here: n_total = 2^36 > 2^32 makes the
%  32-state bitmap CLT impossible. checkpoint=true so a reboot only costs the
%  in-progress M (here only M=0).
%  ================================================================

%% Required inputs
s_val   = 0.5;
J       = 1.0;
R       = 8;
M_lz    = 100;
T_range = logspace(-1, 1.5, 60);

%% Optional inputs
geometry      = 'square_lattice';
Lx            = 6;
Ly            = 6;
only_M0       = true;
ed_thresh     = 50;
mem_diag      = true;
lookup_method = 'schnack';          % MANDATORY at N=36 (bitmap needs n_total<=2^32)
entries_storage = 'host';
checkpoint    = true;
B_gpu         = 0;                   % VRAM-adaptive (will clamp B down on this card)
