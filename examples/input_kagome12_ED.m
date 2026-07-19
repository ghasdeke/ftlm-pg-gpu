% N=12 kagome torus (a,b)=(2,0), full space group C_6v (|G|=48), s=1/2.
% FULL M sweep, EXACT ED of every (M,Gamma) block -> validate vs full ED.
geometry  = 'kagome';
kag_a     = 2;
kag_b     = 0;
s_val     = 0.5;
J         = 1.0;
R         = 8;            % unused for ED
M_lz      = 60;          % unused for ED
ed_thresh = inf;         % exact ED for every block
T_range   = logspace(-1, 1, 40);
