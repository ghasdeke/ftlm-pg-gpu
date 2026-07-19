% N=20 dodecahedron, full I_h (|G|=120), s=1/2, Sz=0 sector, GPU FTLM.
% All 10 I_h irreps within M=0. Small + fully feasible (dim(M=0)=184756,
% n_total=2^20); a quick demonstrator for the dual polyhedron.
geometry      = 'dodecahedron';
s_val         = 0.5;
J             = 1.0;
R             = 8;
M_lz          = 100;
only_M0       = true;
ed_thresh     = 50;             % tiny blocks deterministic, FTLM otherwise
mem_diag      = true;
lookup_method = 'bitmap';       % n_total = 2^20 << 2^32 -> bitmap is fine
T_range       = logspace(-1, 1.5, 60);
