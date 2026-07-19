%% input_kagome36_M0_bounded.m
%  N=36 kagome torus (2,2), M=0, FULL space group C_6v (|G|=144), s=1/2.
%  BOUNDED runtime-optimised validation run (2026-06-04 session):
%    - R=2 (was 8): ~4x fewer Lanczos columns -> ~4x less of the dominant
%      random V[src] gather cost (the d1_1>25min bottleneck).
%    - M_lz=60 (was 100): a further ~1.67x (speed/feasibility test; a proper
%      R/M_lz convergence study is still owed before production thermodynamics).
%    - run_ftlm B2 block-size fix active (B sized AFTER the ~11.5 GB table).
%  Two irreps spanning the d-scaling: d1_1 (d=1) + d2_5 (d=2). The d=4/d=6
%  blocks do NOT fit the 20.5 GB card with the resident table (need ~20.6 /
%  25.1 GB) -> deferred to streaming-B2. enumerate+collect+eskel(B2) run ONCE.
s_val   = 0.5;
J       = 1.0;
R       = 2;
M_lz    = 60;
T_range = logspace(-1, 1.5, 60);

geometry        = 'kagome';
kag_a           = 2;
kag_b           = 2;
only_M0         = true;
ed_thresh       = 50;
mem_diag        = true;
lookup_method   = 'schnack';      % n_total = 2^36 > 2^32 -> mandatory
entries_storage = 'host';
checkpoint      = true;           % per-irrep checkpoint + resume
B_gpu           = 0;              % adaptive (now B2-aware)
irrep_list      = {'d1_1', 'd2_5'};
