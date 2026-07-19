%% input_triangular28_M0.m  -- *** BROKEN / DO NOT USE *** (2026-06-05)
%  The (a,b)=(4,2) cell is CHIRAL: triangular_spacegroup drops the mirror -> C_6
%  (|G|=168, NOT the C_6v |G|=336 the original header wrongly claimed), AND the
%  (4,2) torus is geometrically DEGENERATE -> build_bonds_tri finds 81 != 3N=84
%  NN bonds and ERRORS. There is NO clean C_6v even-N triangular cell between
%  N=16 and N=36 (non-chiral cells are N=a^2 {16,36} or 3a^2 {12,48}; 25/27 are
%  odd). Kept only as a record of the ramp attempt -- do not run.
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
tri_b           = 2;
only_M0         = true;
ed_thresh       = 50;
mem_diag        = true;
lookup_method   = 'schnack';      % mirror the N=36 path
entries_storage = 'host';
checkpoint      = false;          % short enough; sum-rule gate
B_gpu           = 0;
