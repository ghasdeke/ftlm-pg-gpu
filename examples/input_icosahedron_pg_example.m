%% input_icosahedron_pg_example.m
%  ================================================================
%  Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
%  and Helmholtz-Zentrum Dresden-Rossendorf e.V.
%  Licensed under the Apache License, Version 2.0.
%  ================================================================
%  Example input file for ftlm_observables_pg_Ih.
%
%  Invoke as
%      ftlm_observables_pg_Ih('input_icosahedron_pg_example.m')
%
%  This example computes C(T) and chi(T) for the 12-site, s=1/2
%  Heisenberg icosahedron, decomposed into all (M, Gamma) blocks with
%  M = 0..6 and Gamma running over the 10 I_h irreps.
%  ================================================================

%% Required inputs

% Local spin (0.5, 1, 1.5, 2, ...). Only s = 0.5 is fully validated on
% the icosahedron pipeline at this point; for s >= 1 the pipeline runs
% but n_total grows as 3^12, 4^12, ... and CPU-FP64 FTLM may become
% slow without the upcoming GPU port.
s_val  = 0.5;

% Heisenberg exchange coupling. H = +J * sum_{<i,j>} S_i . S_j.
J      = 1.0;

% Number of FTLM random vectors per (M, Gamma) sector.
R      = 30;

% Lanczos iterations per random vector.
M_lz   = 80;

% Temperature grid (units J/k_B).
T_range = logspace(-1, 1, 80);

%% Optional inputs (defaults shown; lines below may be deleted)

% Restrict to M = 0 sectors (then chi(T) is zero). Useful for quick
% smoke tests of the (M=0, Gamma) decomposition.
% only_M0 = false;

% Restrict to a subset of irreps. Names must match exactly:
%   'A_g', 'A_u', 'T1g', 'T1u', 'T2g', 'T2u', 'F_g', 'F_u', 'H_g', 'H_u'.
% Useful for diagnosing individual irrep blocks (e.g., the ground state
% sits in A_g for the s=1/2 icosahedron).
% irrep_list = {'A_g'};

% Dense ED for blocks with n_basis <= ed_thresh. For the s=1/2
% icosahedron every (M, Gamma) block is at most ~46 dimensional, so
% setting ed_thresh = inf reproduces the exact ED result deterministically
% and is the recommended mode for verification. ed_thresh = 0 forces
% FTLM throughout and is the realistic mode for larger spins.
% ed_thresh = 0;

% Output directory for the .mat file.
% output_dir = '.';
