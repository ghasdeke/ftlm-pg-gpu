% 4x4 square lattice, s=1/2, FULL C_4v space group (order 128), full M sweep,
% GPU. ed_thresh=50 so the larger (M,Gamma) blocks run through the CUDA kernel
% with compact-V (d up to 4) -- end-to-end GPU space-group check vs full ED.
geometry      = 'square_lattice_c4v';
Lx            = 4;
Ly            = 4;
s_val         = 0.5;
J             = 1.0;
R             = 8;
M_lz          = 80;
ed_thresh     = 50;
lookup_method = 'bitmap';
entries_storage = 'host';
B_gpu         = 0;
T_range       = logspace(-1, 1, 40);
