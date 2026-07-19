% N=12 triangular torus (a,b)=(2,2), full space group C_6v (|G|=144), s=1/2.
% FULL M sweep, EXACT ED of every (M,Gamma) block -> validate vs full ED
% (see validate_triangular12). Triangular = frustrated AFM (120-deg / RVB).
geometry  = 'triangular';
tri_a     = 2;
tri_b     = 2;
s_val     = 0.5;
J         = 1.0;
R         = 8;            % unused for ED
M_lz      = 60;          % unused for ED
ed_thresh = inf;         % exact ED for every block
T_range   = logspace(-1, 1, 40);
