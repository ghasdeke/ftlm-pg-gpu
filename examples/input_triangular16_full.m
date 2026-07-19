%% input_triangular16_full.m  -- FULL M sweep (physical C(T)+chi(T)) on N=16.
%  Triangular (4,0) N=16 C_6v, s=1/2; all M = 0..8 (only_M0 unset/false, no
%  M_sectors). Expected sum-rule: sum_i w_i = n_total = 2^16 = 65536.
geometry      = 'triangular';
tri_a         = 4;  tri_b = 0;
s_val         = 0.5;  J = 1.0;
R             = 2;  M_lz = 60;
T_range       = logspace(-1, 1.5, 60);
only_M0       = false;        % full sweep
ed_thresh     = 50;
mem_diag      = false;
lookup_method = 'bitmap';
B_gpu         = 0;
