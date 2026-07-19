%% input_icosahedron_s2_M0.m
%  ================================================================
%  Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
%  and Helmholtz-Zentrum Dresden-Rossendorf e.V.
%  Licensed under the Apache License, Version 2.0.
%  ================================================================
%  Input for the s = 2 Heisenberg icosahedron, M = 0 sector only.
%  All 10 I_h irreps are computed within M = 0.
%
%  This is the CPU-vs-GPU benchmark sweetspot for our pipeline:
%      n_total = 5^12 = 244 140 625
%      dim(M=0) ~ 1.98e7 (about 8% of the full Hilbert space)
%      With I_h reduction the (M=0, Gamma) block dimensions land in
%      the 100k-500k range, which is large enough to amortise GPU
%      kernel-launch overhead and exploit the full SpMV throughput.
%
%  Memory expectation:
%      Peak RAM during enumeration ~ 4-5 GB (dense state lookup table
%      + transient digit-decomposition arrays). Per-block CLT memory
%      a few hundred MB. Make sure you have ~ 16 GB free.
%      Peak VRAM on the GPU: a few hundred MB per sector. Easy fit
%      on any 8 GB+ card.
%
%  Wall-time expectation (RTX 4000 SFF Ada / typical workstation CPU):
%      Pre-Lanczos enumeration: 1-2 min (one-shot per cache_M).
%      Per irrep: a few minutes for CLT (CPU) / sparse H (CPU) build,
%      then seconds-to-minutes of Lanczos on the GPU / CPU
%      respectively. Total: 20-40 min GPU, 30-60 min CPU.
%
%  Compare them with:
%      compare_cpu_vs_gpu_s2_M0
%  which runs both drivers, measures wall times, and reports the
%  per-sector breakdown.
%
%  ================================================================

%% Required inputs

s_val   = 2.0;
J       = 1.0;
R       = 8;                       % FTLM random vectors per sector
M_lz    = 100;
T_range = logspace(-1, 1.5, 60);

%% Optional inputs

only_M0    = true;                 % restrict to M = 0 (the test scope)
ed_thresh  = 50;                   % tiny blocks deterministic, FTLM otherwise
%
%   For a larger R (better statistics + more GPU work per sector,
%   better saturation), set R = 30 here. The block sizes are large
%   enough that R_eff = R for all FTLM blocks, no clamping.
% R = 30;
%
% B_gpu      = 0;                  % 0 = adaptive (min(R, 16))
% output_dir = '.';
