%% input_icosahedron_s2_M0_Hg.m
%  ================================================================
%  Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
%  and Helmholtz-Zentrum Dresden-Rossendorf e.V.
%  Licensed under the Apache License, Version 2.0.
%  ================================================================
%  Single-block benchmark input: s = 2 icosahedron, M = 0, ONE irrep
%  (H_g). H_g sits at d_Gamma = 5 with the largest block dimension at
%  M = 0, so it is the most informative single block for the CPU-vs-GPU
%  comparison. Total wall time per driver: a few minutes (compared to
%  ~ 30-60 min for the full 10-irrep sweep).
%
%  Invoke as:
%      ftlm_observables_pg_gpu_Ih('input_icosahedron_s2_M0_Hg.m')
%      ftlm_observables_pg_Ih('input_icosahedron_s2_M0_Hg.m')
%  ================================================================

s_val      = 2.0;
J          = 1.0;
R          = 8;
M_lz       = 100;
T_range    = logspace(-1, 1.5, 60);
only_M0    = true;
irrep_list = {'H_g'};
ed_thresh  = 0;        % force FTLM (no ED fallback) -- exercise the kernel
