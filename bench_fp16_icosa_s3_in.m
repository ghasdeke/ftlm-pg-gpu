% Paper benchmark deck (Table 4, FP16 run): set the environment variable
% FTLM_FP16=1 before running (FP16 storage of the Lanczos vectors; all
% arithmetic stays FP32).
% PAPER BENCHMARK 1/4: icosahedron N=12, s=3, FULL M sweep, R=24.
% Identical deck on both machines (RTX 4000 SFF Ada vs 1x B200); timings
% from the perf block. n_total = 7^12 = 1.38e10 > 2^32 -> schnack mandatory.
% Timing hygiene: no cache, no checkpoint, no mem_diag.
geometry      = 'icosahedron';
s_val         = 3.0;
J             = 1.0;
R             = 24;
M_lz          = 60;
use_spin_flip = true;
ed_thresh     = 0;    % ALL blocks via GPU: the dense-ED builder needs n_total <= 2^32 (7^12 exceeds it)
lookup_method = 'schnack';
T_range       = logspace(-2, 1, 60);
output_dir    = 'runs_bench_fp16/icosa_s3';
