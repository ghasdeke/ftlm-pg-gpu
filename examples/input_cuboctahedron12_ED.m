% 12-site cuboctahedron (O_h, |G|=48), s=1/2, FULL M sweep, EXACT ED of every
% (M,Gamma) block -> used by VALIDATE_CUBOCTAHEDRON12 against independent full
% ED (dim 4096). The 8 corner-sharing triangles make this a classic frustrated
% magnet (cf. Schnack et al.); irreps are built generically (10 O_h irreps,
% max d=3, all realified).
geometry  = 'cuboctahedron';
s_val     = 0.5;
J         = 1.0;
R         = 8;            % unused for ED
M_lz      = 60;           % unused for ED
ed_thresh = inf;          % exact ED for every block
T_range   = logspace(-1, 1, 40);
