%% input_ring_pg_example.m
%  ================================================================
%  Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
%  and Helmholtz-Zentrum Dresden-Rossendorf e.V.
%  Licensed under the Apache License, Version 2.0.
%  ================================================================
%  Example input file for ftlm_observables_pg.
%
%  Invoke as
%      ftlm_observables_pg('input_ring_pg_example.m')
%
%  This example computes C(T) and chi(T) for a 12-site, s=1/2
%  Heisenberg ring with periodic boundary, decomposed into all
%  twelve C_12 translation sectors and three S^z >= 0 sectors.
%  ================================================================

%% Required inputs

% Number of sites on the ring (>= 3). Use a small value for the first
% smoke test (e.g., 8, 10, 12). The pure-MATLAB path does not scale
% to very large rings; that is the job of the upcoming GPU kernel.
N_ring = 12;

% Local spin (0.5, 1, 1.5, 2, ...).
s_val  = 0.5;

% Heisenberg exchange coupling. H = +J * sum_{<i,j>} S_i . S_j.
J      = 1.0;

% Number of FTLM random vectors per (M, p) sector.
R      = 30;

% Lanczos iterations per random vector.
M_lz   = 80;

% Temperature grid (units J/k_B).
T_range = logspace(-1, 1, 80);

%% Optional inputs (defaults shown; lines below may be deleted)

% Restrict to M = 0 sectors (then chi(T) is zero). Useful for quick smoke tests.
% only_M0 = false;

% Restrict to p = 0 (trivial-irrep) sector. Use during Milestone B
% bring-up of the k = 0 GPU kernel; the result is then the partition
% function on the totally symmetric subspace only and not a physical
% thermal observable.
% only_p0 = false;

% Use dense ED for (M, p) sectors with dim_(M, p) <= ed_thresh. With
% ed_thresh = inf (or any value >= max sector dim) every sector is
% diagonalized exactly; the result is then deterministic and identical
% to the unsymmetrized full ED. Use this mode for unit testing.
% ed_thresh = 0;

% Output directory for the .mat file.
% output_dir = '.';
