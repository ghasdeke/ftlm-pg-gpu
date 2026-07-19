%% input_ring_24_s12.m
%  ================================================================
%  Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
%  and Helmholtz-Zentrum Dresden-Rossendorf e.V.
%  Licensed under the Apache License, Version 2.0.
%  ================================================================
%  PG-FTLM input: spin-1/2 Heisenberg ring with N = 24 sites.
%
%  Invoke as
%      ftlm_observables_pg_gpu('input_ring_24_s12.m')
%
%  System sizes (s = 1/2):
%     n_total          = 2^24 = 16 777 216
%     dim(M = 0)       = C(24, 12) ~ 2.70e6  unsymmetrized
%     dim(M = 0, p)    ~ C(24, 12) / 24 ~ 1.12e5  per momentum sector
%
%  Memory budget on a typical workstation GPU (FP32 vectors):
%     real sectors  (p = 0, p = N/2):  3 * dim * B * 4 bytes
%     cplx sectors  (p != 0, p != N/2): 6 * dim * B * 4 bytes
%
%     For dim ~ 1.1e5 and B = 8 that is ~10 MB and ~20 MB respectively;
%     well within the 20 GB of an RTX 4000 Ada.
%
%  Expected wall time on an RTX 4000 Ada (R = 20, M_lz = 120):
%     enumeration phase :  ~2 s   (vectorized over M sectors)
%     GPU FTLM (real)   :  ~1 s per sector
%     GPU FTLM (complex):  ~2 s per sector
%     total             :  on the order of one minute
%  ================================================================

%% Required inputs
N_ring = 24;
s_val  = 0.5;
J      = 1.0;
R      = 20;
M_lz   = 120;
T_range = logspace(-1.5, 1.5, 80);

%% Optional inputs

% Set to true for a fast smoke test (one M sector, one p sector).
% only_M0 = false;
% only_p0 = false;

% Exploit the complex-conjugate k <-> N-k pair symmetry of the
% irreps under a real H. When true (default), the canonical range
% p = 0 .. floor(N/2) is looped over and partner sectors are absorbed
% via mult_p = 2. Halves the number of complex-p sectors that need to
% run. Set to false for explicit verification.
% merge_kbar = true;

% Exploit the spin-inversion Z_2 symmetry in (M = 0, p) sectors by
% projecting random FTLM starting vectors onto +/- parity subspaces.
% R is split as R_+ = ceil(R/2), R_- = R - R_+, so the total Lanczos
% work matches the no-parity case (instead of doubling). Yields a
% spectrum cleanly labeled by parity without changing SpMV cost.
% Default: true.
% use_spin_parity = true;

% Use ED instead of FTLM for sectors with dim <= ed_thresh. Sensible
% range here: 0 (everything FTLM) up to a few thousand if you want the
% small high-M sectors to be exact at zero stochastic cost.
ed_thresh = 200;

% GPU block size (0 = adaptive based on L2 size).
B_gpu = 0;
L2_cache_bytes = 48e6;

% output_dir = '.';
