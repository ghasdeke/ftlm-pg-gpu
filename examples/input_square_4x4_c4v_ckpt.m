% 4x4 C_4v space group, s=1/2, M=0, GPU, per-IRREP checkpointing.
% Smoke test for the per-irrep checkpoint/resume logic.
geometry        = 'square_lattice_c4v';
Lx              = 4;
Ly              = 4;
s_val           = 0.5;
J               = 1.0;
R               = 8;
M_lz            = 60;
only_M0         = true;
ed_thresh       = 50;
lookup_method   = 'bitmap';
entries_storage = 'host';
checkpoint      = true;
B_gpu           = 0;
T_range         = logspace(-1, 1, 40);
