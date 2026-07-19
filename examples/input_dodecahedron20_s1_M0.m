% N=20 dodecahedron, full I_h (|G|=120), s=1, Sz=0 sector, GPU FTLM.
% Orientation/test run: all 10 I_h irreps within M=0, per-irrep timings,
% and the ground-state energy E0 = min(all_E).
%   d_loc = 3, n_total = 3^20 = 3 486 784 401 (< 2^32),
%   dim(M=0) = central coeff of (1+x+x^2)^20 = 377 379 369.
geometry      = 'dodecahedron';
s_val         = 1.0;
J             = 1.0;
R             = 8;
M_lz          = 100;
only_M0       = true;
ed_thresh     = 50;             % tiny blocks deterministic, FTLM otherwise
mem_diag      = true;           % per-phase GPU + host (MATLAB memory) snapshots
checkpoint    = true;           % per-irrep ckpt (run may approach the long-job cap)
lookup_method = 'schnack';      % combinatorial ranking (robust for d_loc=3)
T_range       = logspace(-1, 1.5, 60);
