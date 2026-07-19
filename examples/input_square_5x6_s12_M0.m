%% input_square_5x6_s12_M0.m
%  ================================================================
%  Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
%  and Helmholtz-Zentrum Dresden-Rossendorf e.V.
%  Licensed under the Apache License, Version 2.0.
%  ================================================================
%  Input for the s = 1/2 Heisenberg 5 x 6 periodic square lattice, M = 0
%  sector only, ALL irreps of the translation group C_5 x C_6.
%
%  System size:
%      N_sites  = 30
%      n_total  = 2^30 = 1 073 741 824   (< 2^32 -> bitmap lookup is valid)
%      dim(M=0) = C(30, 15) = 155 117 520
%  Symmetry: lattice translations C_5 x C_6 (order 30, 30 one-dimensional
%      momentum irreps (px, py), px=0..4, py=0..5). All d = 1, so the
%      cheapest / most-optimised kernel path (const-c, no d x d contraction).
%      With the order-30 reduction each (M=0, k) block is ~ dim/30 ~ 5.17e6.
%
%  Same FTLM settings as the icosidodecahedron production input so the two
%  runs are directly comparable (R = 8, M_lz = 100, T-grid identical).
%  ================================================================

%% Required inputs
s_val   = 0.5;
J       = 1.0;
R       = 8;                        % FTLM random vectors per (M, Gamma) block
M_lz    = 100;
T_range = logspace(-1, 1.5, 60);

%% Optional inputs
geometry      = 'square_lattice';   % triggers square_lattice_translation_group + adjacency_square_lattice
Lx            = 5;
Ly            = 6;
only_M0       = true;               % M = 0 only, all 30 momentum irreps
ed_thresh     = 50;                 % tiny blocks deterministic; M=0 blocks are huge -> FTLM
mem_diag      = true;               % per-phase GPU + host memory snapshots
lookup_method = 'bitmap';           % n_total = 2^30 < 2^32 -> bitmap CLT valid
entries_storage = 'host';
B_gpu         = 0;                  % 0 = VRAM-adaptive block size
