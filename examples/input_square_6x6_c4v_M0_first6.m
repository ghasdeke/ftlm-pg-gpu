%% input_square_6x6_c4v_M0_first6.m
%  ================================================================
%  Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
%  and Helmholtz-Zentrum Dresden-Rossendorf e.V.
%  Licensed under the Apache License, Version 2.0.
%  ================================================================
%  6x6 square C_4v M=0, s=1/2 -- BOUNDED variant: only the FIRST 6 irreps
%  (d1_1..d1_6, all d=1). A quick (~23 min) end-to-end check of the detached
%  launch + precompute cache on the real N=36 target, without the full ~65 min
%  27-irrep sweep. The precompute cache it writes (precompute_sq6x6sg_s1o2_M0.mat,
%  keyed by geometry+s+M, NOT by irrep_list) is reused verbatim by a later full
%  27-irrep run -> that run then loads it in seconds instead of recomputing.
%  ================================================================

%% Required inputs
s_val   = 0.5;
J       = 1.0;
R       = 2;
M_lz    = 60;
T_range = logspace(-1, 1.5, 60);

%% Optional inputs
geometry        = 'square_lattice_c4v';
Lx              = 6;
Ly              = 6;
only_M0         = true;
ed_thresh       = 50;
mem_diag        = true;
lookup_method   = 'schnack';               % MANDATORY: n_total = 2^36 > 2^32
entries_storage = 'host';
checkpoint      = true;
precompute_cache = true;                    % writes ~12 GB cache (reused by full run)
B_gpu           = 0;
irrep_list      = {'d1_1','d1_2','d1_3','d1_4','d1_5','d1_6'};   % first 6 only
