%% input_triangular16_M2.m  -- single-sector test of the M_sectors input option.
%  Triangular (4,0) N=16 C_6v, s=1/2; run ONLY the M=2 sector.
%  Expected sum-rule: sum_i w_i = mult_M * dim(M=2) = 2 * C(16,10) = 2*8008 = 16016.
geometry      = 'triangular';
tri_a         = 4;  tri_b = 0;
s_val         = 0.5;  J = 1.0;
R             = 2;  M_lz = 60;
T_range       = logspace(-1, 1.5, 60);
M_sectors     = 2;            % <-- run only |M|=2 (the new option)
ed_thresh     = 50;
mem_diag      = false;
lookup_method = 'bitmap';
B_gpu         = 0;
