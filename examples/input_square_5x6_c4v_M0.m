%% input_square_5x6_c4v_M0.m
%  ================================================================
%  Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
%  and Helmholtz-Zentrum Dresden-Rossendorf e.V.
%  Licensed under the Apache License, Version 2.0.
%  ================================================================
%  5 x 6 periodic square lattice, s = 1/2, M = 0 only, FULL space group.
%  5x6 is RECTANGULAR -> point group C_2v (order 4) -> space group order
%  30*4 = 120 (coincidentally = |I_h|). n_reps ~ dim(M=0)/120 ~ 1.29e6
%  (like the icosidodecahedron), so this is a fast, fully-feasible run that
%  exercises the complete space-group pipeline (translations x point group,
%  with d=1 and d=2 irreps) and per-IRREP checkpointing.
%
%  dim(M=0) = C(30,15) = 1.551e8 ; n_total = 2^30 < 2^32 -> bitmap lookup OK.
%  ================================================================

%% Required inputs
s_val   = 0.5;
J       = 1.0;
R       = 8;
M_lz    = 100;
T_range = logspace(-1, 1.5, 60);

%% Optional inputs
geometry        = 'square_lattice_c4v';   % full space group (C_2v x translations for 5x6)
Lx              = 5;
Ly              = 6;
only_M0         = true;
ed_thresh       = 50;
mem_diag        = true;
lookup_method   = 'bitmap';               % n_total = 2^30 < 2^32
entries_storage = 'host';
checkpoint      = true;                    % per-irrep checkpoint + resume
B_gpu           = 0;
