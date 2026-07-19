% Paper benchmark deck (Table 4, FP32 run).
% PAPER BENCHMARK 3/4: dodecahedron N=20, s=1, FULL M sweep, R=24.
% Identical deck on both machines. n_total = 3^20 = 3.49e9 < 2^32 -> bitmap.
geometry      = 'dodecahedron';
s_val         = 1.0;
J             = 1.0;
R             = 24;
M_lz          = 60;
use_spin_flip = true;
ed_thresh     = 200;
lookup_method = 'bitmap';
T_range       = logspace(-2, 1, 60);
output_dir    = 'runs_bench_fp32new/dodec_s1';
