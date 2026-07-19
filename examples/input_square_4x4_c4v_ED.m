% 4x4 square lattice, s=1/2, FULL space group (C_4v x translations, order 128),
% FULL M sweep, EXACT diagonalisation of every (M,Gamma) block (ed_thresh=inf).
% Used by VALIDATE_SQUARE_4X4_C4V to check the aggregated spectrum vs full ED.
geometry  = 'square_lattice_c4v';
Lx        = 4;
Ly        = 4;
s_val     = 0.5;
J         = 1.0;
R         = 8;            % unused for ED
M_lz      = 80;          % unused for ED
ed_thresh = inf;         % exact ED for every block
T_range   = logspace(-1, 1, 40);
