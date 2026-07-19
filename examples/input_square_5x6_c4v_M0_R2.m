%% input_square_5x6_c4v_M0_R2.m
%  ================================================================
%  Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
%  and Helmholtz-Zentrum Dresden-Rossendorf e.V.
%  Licensed under the Apache License, Version 2.0.
%  ================================================================
%  5 x 6 periodic square lattice, s = 1/2, M = 0 only, FULL space group
%  (C_2v x translations, |G| = 120), realified FS=+1 irreps. Fast R=2 test
%  variant of input_square_5x6_c4v_M0.m (R 8 -> 2, ~4x faster; mem_diag /
%  checkpoint off for a clean quick run).
%
%  dim(M=0) = C(30,15) = 1.551e8 ; n_total = 2^30 < 2^32 -> bitmap lookup OK.
%  ================================================================

%% Required inputs
s_val   = 0.5;
J       = 1.0;
R       = 2;
M_lz    = 100;
T_range = logspace(-1, 1.5, 60);

%% Optional inputs
geometry        = 'square_lattice_c4v';   % full space group (C_2v x translations for 5x6)
Lx              = 5;
Ly              = 6;
only_M0         = true;
ed_thresh       = 50;
mem_diag        = false;
lookup_method   = 'bitmap';               % n_total = 2^30 < 2^32
entries_storage = 'host';
checkpoint      = false;
B_gpu           = 0;
