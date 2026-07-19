% Input: 4x4 periodic square lattice, s = 1/2, Heisenberg AFM (J = 1).
% GPU driver (mixed FP32/FP64). FTLM throughout (ed_thresh = 0) so every
% block goes through the CUDA kernel (init_skel_ref) -> exercises the GPU
% path for the C_4 x C_4 translation group (order 16, not 120).

geometry        = 'square_lattice';
Lx              = 4;
Ly              = 4;

s_val           = 0.5;
J               = 1.0;

R               = 120;     % FTLM random vectors per (M, Gamma) block
M_lz            = 150;     % Lanczos steps per random vector
ed_thresh       = 0;       % FTLM (GPU kernel) throughout

lookup_method   = 'bitmap';
entries_storage = 'host';

T_range         = logspace(-1, 1, 60);   % MUST match input_square_4x4_s12.m
