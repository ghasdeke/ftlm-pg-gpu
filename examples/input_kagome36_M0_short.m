%% input_kagome36_M0_short.m
%  N=36 kagome torus (2,2), M=0, FULL space group (|G|=144), s=1/2 -- but a
%  SHORT run: only TWO irreps (one d=1 + the largest d=6) to validate B2
%  entry-tiling end-to-end at N=36 scale (n_entries ~2.3e9 > 2^31) without the
%  full ~5-8 h sweep. enumerate+collect+eskel(B2) run once regardless.
s_val   = 0.5;
J       = 1.0;
R       = 8;
M_lz    = 100;
T_range = logspace(-1, 1.5, 60);

geometry        = 'kagome';
kag_a           = 2;
kag_b           = 2;
only_M0         = true;
ed_thresh       = 50;
mem_diag        = true;
lookup_method   = 'schnack';
entries_storage = 'host';
checkpoint      = true;
B_gpu           = 0;
irrep_list      = {'d1_1', 'd6_14'};      % one d=1 + the largest d=6 block
