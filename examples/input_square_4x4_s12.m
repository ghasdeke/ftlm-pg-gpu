% Input: 4x4 periodic square lattice, s = 1/2, Heisenberg AFM (J = 1).
% Symmetry: lattice translations C_4 x C_4 (order 16, 16 momentum irreps).
% FTLM throughout (ed_thresh = 0). Used by VALIDATE_SQUARE_4X4 and runnable
% directly via  ftlm_observables_pg_Ih('input_square_4x4_s12.m').

geometry = 'square_lattice';
Lx       = 4;
Ly       = 4;

s_val    = 0.5;
J        = 1.0;

R        = 120;      % FTLM random vectors per (M, Gamma) block
M_lz     = 150;      % Lanczos steps per random vector
ed_thresh = 0;       % FTLM throughout

T_range  = logspace(-1, 1, 60);   % temperatures in units of J / k_B
