% Paper benchmark deck (Table 4, FP16 run): set the environment variable
% FTLM_FP16=1 before running (FP16 storage of the Lanczos vectors; all
% arithmetic stays FP32).
% PAPER BENCHMARK 2/4: icosidodecahedron N=30, s=1/2, FULL M sweep, R=24.
% Identical deck on both machines. n_total = 2^30 -> bitmap lookup.
geometry      = 'icosidodecahedron';
s_val         = 0.5;
J             = 1.0;
R             = 24;
M_lz          = 60;
use_spin_flip = true;
ed_thresh     = 200;
lookup_method = 'bitmap';
T_range       = logspace(-2, 1, 60);
output_dir    = 'runs_bench_fp16/icosido_s12';
