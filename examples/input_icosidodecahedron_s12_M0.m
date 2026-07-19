%% input_icosidodecahedron_s12_M0.m
%  ================================================================
%  Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
%  and Helmholtz-Zentrum Dresden-Rossendorf e.V.
%  Licensed under the Apache License, Version 2.0.
%  ================================================================
%  Input for the s = 1/2 Heisenberg icosidodecahedron, M = 0 sector only.
%  All 10 I_h irreps are computed within M = 0.
%
%  System size:
%      N_sites = 30
%      n_total = 2^30 = 1 073 741 824
%      dim(M=0) = C(30, 15) = 155 117 520 (~ 14.4 % of the full Hilbert space)
%      With I_h reduction (factor ~ 120) the (M=0, Gamma) block dimensions
%      land in the 1.0e6 - 6.5e6 range (H_g being the largest at ~ d_Gamma
%      x ~ 1.3 M = 6.5 M states), large enough to saturate the GPU SpMV.
%
%  Geometry: triggered via the 'geometry' variable below, which the
%  driver FTLM_OBSERVABLES_PG_GPU_IH dispatches to
%  ICOSIDODECAHEDRON_IH_FULL + ADJACENCY_ICOSIDODECAHEDRON_IH.
%
%  Memory expectations (production code path on the RTX 4000 SFF Ada):
%      Peak HOST RAM:
%        - dense state lookup table (int32, n_total entries) ~ 4 GB
%        - per-sector entry buffers (a few GB transient during CLT build)
%        Make sure you have ~ 24-32 GB of RAM available.
%      Peak VRAM:
%        - per-sector CLT M_tens (complex double, d x d x n_entries) up to
%          ~ 2 GB on H_g. Easily fits on a 20 GB card.
%
%  Wall-time expectation (RTX 4000 SFF Ada, post-consolidation pipeline):
%      Roughly ~ 8x the s = 2 icosahedron M = 0 sweep, since the largest
%      block scales 8x in basis dim. Order-of-magnitude estimate
%      ~ 8 x 75 s = 600 s for the 10-irrep M=0 sweep. May be higher on
%      the first run (CUDA / MKL warm-up) and on the host CLT-build
%      phase, which is the current bottleneck.
%
%  Spin convention: only s = 1/2 is computationally feasible here.
%      For s >= 1, n_total = 3^30 > 10^14 is out of reach without
%      additional symmetry reductions beyond I_h.
%
%  ================================================================

%% Required inputs

s_val   = 0.5;
J       = 1.0;
R       = 8;                       % FTLM random vectors per sector
M_lz    = 100;
T_range = logspace(-1, 1.5, 60);

%% Optional inputs

geometry   = 'icosidodecahedron';   % triggers the 30-site I_h provider
only_M0    = true;                  % restrict to M = 0 (the test scope)
ed_thresh  = 50;                    % tiny blocks deterministic, FTLM otherwise
mem_diag   = true;                  % per-phase GPU+host memory snapshots
lookup_method = 'schnack';          % combinatorial ranking (Stufe 6a);
                                    % set to 'bitmap' for the bitmap CLT
                                    % path (the legacy default).
% Three coexisting paths:
%   'host'       : production default. Entries (src/tgt/g/c) on host.
%                  Works at any N within host RAM. Schnack lookup also
%                  on host. The cleanest MATLAB-level configuration.
%   'gpu'        : Stufe 6b WITHDRAWN. Per-chunk gpuArray(host) uploads
%                  inflated MATLAB's pinned-host pool by ~3.6 GB per
%                  sector at N=30. DO NOT USE.
%   'gpu_native' : Phase B.2.1 WITHDRAWN. Even with all gpuArrays born
%                  on-device (no host-uploads), MATLAB's internal
%                  gpuArray pool accumulates ~3 GB per sector worth of
%                  host shadows across the many gpuArray ops in
%                  collect_clt_entries_Ih_gpu. The host-RAM impact was
%                  WORSE than 'host'. Code retained in repository for
%                  reference but DO NOT USE in production. The lesson:
%                  MATLAB's gpuArray management has a structural
%                  host-side overhead of order 2-3x the GPU data, which
%                  cannot be removed at MATLAB level. To get truly
%                  host-free entry storage the path forward is a CUDA
%                  MEX kernel that builds entries directly in VRAM
%                  without going through MATLAB gpuArrays (Phase B.2.2).
entries_storage = 'host';
%
% R = 30;                           % larger statistics, more GPU work / sector
% B_gpu      = 0;                   % 0 = adaptive (min(R, 16))
% output_dir = '.';
