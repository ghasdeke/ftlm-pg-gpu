%% input_square_4x4_c4v_pccache.m
%  ================================================================
%  Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
%  and Helmholtz-Zentrum Dresden-Rossendorf e.V.
%  Licensed under the Apache License, Version 2.0.
%  ================================================================
%  Tiny 4x4 C_4v (|G|=128) M=0 run with the precompute cache ENABLED, used by
%  tests/test_precompute_cache.m. Same geometry path as the 6x6 target but
%  instant (dim(M=0)=C(16,8)=12870). n_total=2^16 < 2^32 -> bitmap lookup OK.
%  ================================================================
s_val   = 0.5;
J       = 1.0;
R       = 2;
M_lz    = 20;
T_range = logspace(-1, 1.5, 20);

geometry        = 'square_lattice_c4v';
Lx              = 4;
Ly              = 4;
only_M0         = true;
ed_thresh       = 50;
mem_diag        = false;
lookup_method   = 'bitmap';
entries_storage = 'host';
checkpoint      = false;
precompute_cache = true;        % <-- exercise the precompute cache
B_gpu           = 0;
