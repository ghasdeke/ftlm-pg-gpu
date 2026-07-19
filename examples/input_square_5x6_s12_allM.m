%% input_square_5x6_s12_allM.m
%  ================================================================
%  Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
%  and Helmholtz-Zentrum Dresden-Rossendorf e.V.
%  Licensed under the Apache License, Version 2.0.
%  ================================================================
%  5 x 6 periodic square lattice, s = 1/2, Heisenberg AFM (J = 1).
%  FULL M sweep (M = 0 .. 15), all 30 irreps of C_5 x C_6 -> the complete
%  thermodynamics: physical C(T) AND chi(T) (the M=0-only run gives chi == 0).
%
%  Covers the whole Hilbert space (n_total = 2^30), so ~ 7x the work of the
%  M=0-only benchmark (sum_M dim(M) = 2^30 vs dim(M=0) = 1.55e8).
%  ================================================================

%% Required inputs
s_val   = 0.5;
J       = 1.0;
R       = 8;                        % FTLM random vectors per (M, Gamma) block
M_lz    = 100;
T_range = logspace(-1, 1.5, 60);

%% Optional inputs
geometry      = 'square_lattice';
Lx            = 5;
Ly            = 6;
only_M0       = false;              % FULL M sweep (this is the difference)
ed_thresh     = 50;                 % tiny high-M blocks deterministic; large ones FTLM
mem_diag      = true;
lookup_method = 'bitmap';           % n_total = 2^30 < 2^32 -> bitmap valid
entries_storage = 'host';
checkpoint    = true;               % per-M checkpoint + resume (reboot-resilient)
B_gpu         = 0;
