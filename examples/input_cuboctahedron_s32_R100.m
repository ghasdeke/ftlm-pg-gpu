% Physics production deck: cuboctahedron s=3/2 (full physics run, R=100).
geometry      = 'cuboctahedron';
s_val         = 1.5;
J             = 1.0;
R             = 100;
M_lz          = 100;
use_spin_flip = true;
ed_thresh     = 200;
lookup_method = 'bitmap';
T_range       = logspace(-2, 1.5, 200);
output_dir    = 'runs_cubo_s32_R100';
