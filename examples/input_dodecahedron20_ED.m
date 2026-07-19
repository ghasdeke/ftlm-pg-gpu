% N=20 dodecahedron (dual of the icosahedron), full I_h (|G|=120), s=1/2.
% Sz=0 sector, EXACT ED of every (M,Gamma) block -> validate vs an
% independent symmetry-free sparse Lanczos (see validate_dodecahedron20).
geometry  = 'dodecahedron';
s_val     = 0.5;
J         = 1.0;
R         = 8;            % unused for ED
M_lz      = 60;           % unused for ED
only_M0   = true;
ed_thresh = inf;          % exact ED for every block
T_range   = logspace(-1, 1, 40);
