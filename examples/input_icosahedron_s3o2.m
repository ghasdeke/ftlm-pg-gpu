%% input_icosahedron_s3o2.m
%  ================================================================
%  Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
%  and Helmholtz-Zentrum Dresden-Rossendorf e.V.
%  Licensed under the Apache License, Version 2.0.
%  ================================================================
%  Input for the s = 3/2 icosahedron.
%
%  Invoke with either driver:
%      ftlm_observables_pg_gpu_Ih('input_icosahedron_s3o2.m')   % GPU FP32
%      ftlm_observables_pg_Ih('input_icosahedron_s3o2.m')       % CPU FP64
%
%  System size:
%      d_loc   = 4, n_total = 4^12 = 16 777 216
%      M_max   = 18, so M sectors run from 0 to 18.
%      dim(M=0) ~ 1.7 million; with I_h reduction the block dimensions
%      land in the low thousands. This is the regime where the GPU
%      kernel finally saturates and beats the CPU FP64 path.
%
%  Wall-time expectation:
%      The brute-force M-sector enumeration is O(n_total) per (M, Gamma)
%      call. For 18 M values x 10 irreps = up to 180 calls, expect a
%      non-trivial pre-Lanczos cost in MATLAB. Total wall time is
%      typically tens of minutes to an hour, depending on hardware.
%
%  Diagnostics (after the run):
%      load('ftlm_pg_gpu_Ih_icos_s3o2.mat')
%
%      % Sum-rule (deterministic invariant; must equal n_total = 16777216).
%      assert(abs(sum(all_w_) - 16777216) < 1, 'sum-rule violated');   %#ok
%      % Note: all_w is not saved by default; check the driver printout
%      % "Sum-rule check" instead.
%
%      % Curie limit at highest T: chi(T_hi) * T_hi -> N * s*(s+1)/3
%      %   = 12 * (3/2)*(5/2) / 3 = 15.0.
%      fprintf('chi(T=%.1f) * T = %.3f  (Curie limit: 15.0)\n', ...
%              T_range(end), chi_T(end) * T_range(end));
%
%  ================================================================

%% Required inputs

s_val   = 1.5;
J       = 1.0;
R       = 8;                       % small R as suggested for first run
M_lz    = 100;
T_range = logspace(-1, 1.5, 60);

%% Optional inputs (defaults shown; lines below may be deleted)

% only_M0 = false;
% irrep_list = {'A_g'};            % e.g., restrict to a single irrep for
                                    % a quick warm-up / ground-state probe
ed_thresh = 50;                    % small blocks deterministic, FTLM elsewhere
% B_gpu      = 0;                  % GPU block size (0 = adaptive: min(R, 16))
% output_dir = '.';
