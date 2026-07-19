% Input: 4x4 periodic square lattice, s = 1/2, Heisenberg AFM (J = 1).
% Same system as input_square_4x4_s12.m but EXACT diagonalisation of every
% (M, Gamma) block (ed_thresh = inf) -> symmetry-adapted exact reference.
% The aggregated spectrum must reproduce the full no-symmetry ED.

geometry = 'square_lattice';
Lx       = 4;
Ly       = 4;

s_val    = 0.5;
J        = 1.0;

R        = 30;       % unused for ED, but the driver requires it
M_lz     = 100;      % unused for ED, but the driver requires it
ed_thresh = inf;     % exact ED for every block

T_range  = logspace(-1, 1, 60);   % MUST match input_square_4x4_s12.m
