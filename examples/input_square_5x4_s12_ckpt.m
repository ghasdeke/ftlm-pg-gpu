% 5x4 square lattice (N=20), s=1/2, FULL M sweep, GPU, per-M CHECKPOINTING.
% Mid-sized/slow-enough case to validate checkpoint + RESUME after an
% interrupted run. Larger R/M_lz purely to make each M take ~1-2 s so the
% interruption lands mid-sweep deterministically.
geometry        = 'square_lattice';
Lx              = 5;
Ly              = 4;
s_val           = 0.5;
J               = 1.0;
R               = 30;
M_lz            = 150;
ed_thresh       = 50;
lookup_method   = 'bitmap';
entries_storage = 'host';
checkpoint      = true;
B_gpu           = 0;
T_range         = logspace(-1, 1, 40);
