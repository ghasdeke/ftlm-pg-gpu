%% input_icosidodecahedron_s12_full_R24.m
%  ================================================================
%  Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
%  and Helmholtz-Zentrum Dresden-Rossendorf e.V.
%  Licensed under the Apache License, Version 2.0.
%  ================================================================
%  s = 1/2 icosidodecahedron (N=30), FULL M sweep, R=24 (3x the R=8 run) to
%  reduce the FTLM stochastic scatter in the low-T C(T) (~1/sqrt(R) -> ~1.7x
%  cleaner than R=8). Same physics target as input_icosidodecahedron_s12_full.m;
%  enumerate+collect are R-independent (cached), only the Lanczos cost scales
%  with R -> ~3x the R=8 irrep-time, ~33 min total.
%  NOTE: result .mat name is R-independent (ftlm_pg_gpu_Ih_icosido_s1o2.mat) ->
%  back up the R=8 result before running this.
%  ================================================================

%% Required inputs
s_val   = 0.5;
J       = 1.0;
R       = 24;
M_lz    = 100;
T_range = logspace(-1, 1.5, 60);

%% Optional inputs
geometry      = 'icosidodecahedron';
only_M0       = false;             % full M>=0 sweep (physical C(T) + chi(T))
ed_thresh     = 50;
mem_diag      = false;
lookup_method = 'schnack';
entries_storage = 'host';
checkpoint    = true;              % per-(M,irrep) checkpoint (reboot-prone machine)
precompute_cache = true;          % cache enumerate+collect per M (~0.5 GB, reused by reruns)
B_gpu         = 0;
