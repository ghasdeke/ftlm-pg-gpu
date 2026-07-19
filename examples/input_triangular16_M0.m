%% input_triangular16_M0.m  -- ramp smoke (N=16) for the triangular GPU path.
%  Triangular torus (a,b)=(4,0): ncells=16, N=16, full C_6v space group |G|=192.
%  s=1/2, M=0 only. dim(M=0)=C(16,8)=12870. n_total=2^16<2^32 -> bitmap OK.
%  Exercises the GPU driver + space-group reduction; sum rule must = dim(M=0).
%  ================================================================

%% Required inputs
s_val   = 0.5;
J       = 1.0;
R       = 2;
M_lz    = 60;
T_range = logspace(-1, 1.5, 60);

%% Optional inputs
geometry        = 'triangular';
tri_a           = 4;
tri_b           = 0;
only_M0         = true;
ed_thresh       = 50;
mem_diag        = true;
lookup_method   = 'bitmap';       % n_total = 2^16 < 2^32 -> bitmap fine
entries_storage = 'host';
checkpoint      = false;          % short run
B_gpu           = 0;
