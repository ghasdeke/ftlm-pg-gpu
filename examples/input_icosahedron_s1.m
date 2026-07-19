%% input_icosahedron_s1.m
%  ================================================================
%  Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
%  and Helmholtz-Zentrum Dresden-Rossendorf e.V.
%  Licensed under the Apache License, Version 2.0.
%  ================================================================
%  Input file for ftlm_observables_pg_Ih on the s = 1 icosahedron.
%
%  Invoke as
%      ftlm_observables_pg_Ih('input_icosahedron_s1.m')
%
%  System size:
%      n_total = 3^12 = 531441 states
%      M_max   = 12, so M sectors run from 0 to 12.
%
%  Expected wall time on a typical CPU: ~ 5-30 minutes, depending on
%  R, M_lz, and which CPU. The biggest cost is the per-(M, Gamma)
%  build of the sparse H block plus FTLM Lanczos. The largest blocks
%  sit at M = 0 and dim_block is in the low hundreds; Lanczos with
%  M_lz = 100 is therefore much shorter than full exhaustion.
%
%  Sanity checks to look at in the output:
%      - Sum-rule:    sum_i w_i == 531441 within rounding error.
%      - chi(T)*T at high T should approach the Curie limit
%                       chi*T = N * s*(s+1)/3 = 12 * 1*2 / 3 = 8.0
%        (in our units; check the limit at T = 30).
%      - Ground state energy E_0 = min(all_E) should match the s = 1
%        Heisenberg icosahedron literature value
%        (E_0 / N_bonds ~ -1.55 J for spin-1 icosahedron; see e.g.
%         Coffey & Trugman, PRB 46, 12717 (1992)). For 30 bonds this
%        gives E_0 ~ -46.5 J. The exact value depends on convention.
%  ================================================================

%% Required inputs

% Local spin.
s_val  = 1.0;

% Heisenberg exchange coupling. H = +J * sum_{<i,j>} S_i . S_j.
J      = 1.0;

% Number of FTLM random vectors per (M, Gamma) sector. For s = 1 the
% block dimensions are large enough that R = 30 gives a meaningful
% statistical average; on the larger M = 0 blocks (~ few hundred) the
% noise floor 1 / sqrt(D) is below 5%, so C(T) and chi(T) at moderate
% to high T should be tens of times more accurate than the s = 1/2
% smoke-test case.
R      = 30;

% Lanczos iterations per random vector. With M_lz = 100 we capture the
% ground-state subspace of every block to high accuracy without paying
% for full exhaustion of the larger blocks.
M_lz   = 100;

% Temperature grid (units J/k_B). Spans the low-T regime (ground-state
% physics, kT << gap) through the crossover (kT ~ J) into the
% paramagnetic Curie tail (kT >> J).
T_range = logspace(-1.0, 1.5, 60);

%% Optional inputs (defaults shown; lines below may be deleted)

% Restrict to M = 0 (chi(T) becomes zero in that case, but the spectrum
% includes the ground state). Useful as a fast warm-up to confirm the
% pipeline runs on s = 1 before the full M loop.
% only_M0 = false;

% Restrict to a subset of irreps. For a smoke test of one block try
% irrep_list = {'A_g'};
% irrep_list = {'A_g', 'A_u', 'T1g', 'T1u', 'T2g', 'T2u', 'F_g', 'F_u', 'H_g', 'H_u'};

% Blocks with n_basis <= ed_thresh are diagonalised exactly instead
% of FTLM. For s = 1 the high-M sectors give some very small blocks
% (~ a few) that benefit from this; ed_thresh = 20 is a reasonable
% balance between exactness and exercising the Lanczos kernel.
ed_thresh = 20;

% Output directory for the .mat file.
% output_dir = '.';
