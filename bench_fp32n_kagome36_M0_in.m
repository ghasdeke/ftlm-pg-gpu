% Paper benchmark deck (Table 4, FP32 run).
% PAPER BENCHMARK 4/4: kagome N=36 torus (a,b)=(2,2), s=1/2, M=0 only, R=8.
% THE cross-machine data-movement comparison: the ~11.5 GB entry table is
% VRAM-resident on the B200 but must be host-STREAMED (auto prefix + tiles)
% on a 20 GB card -- same code, automatic degradation. n_total = 2^36 ->
% schnack mandatory. checkpoint ON (long run on the workstation); per-irrep
% sector_t timings are unaffected.
geometry      = 'kagome';
kag_a         = 2;
kag_b         = 2;
s_val         = 0.5;
J             = 1.0;
R             = 8;
M_lz          = 60;
only_M0       = true;
use_spin_flip = true;
ed_thresh     = 0;    % 2^36 > 2^32: dense-ED builder unavailable (blocks are huge anyway)
lookup_method = 'schnack';
checkpoint    = true;
T_range       = logspace(-1, 1.5, 60);
output_dir    = 'runs_bench_fp32new/kagome36_M0';
