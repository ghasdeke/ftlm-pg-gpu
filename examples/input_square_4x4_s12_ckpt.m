% 4x4 square lattice, s=1/2, FULL M sweep, GPU, with per-M CHECKPOINTING.
% Small/fast case used to validate the checkpoint + resume logic.
geometry        = 'square_lattice';
Lx              = 4;
Ly              = 4;
s_val           = 0.5;
J               = 1.0;
R               = 8;
M_lz            = 80;
ed_thresh       = 50;
lookup_method   = 'bitmap';
entries_storage = 'host';
checkpoint      = true;        % <-- per-M checkpoint + resume
B_gpu           = 0;
T_range         = logspace(-1, 1, 40);
