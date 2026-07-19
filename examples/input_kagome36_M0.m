%% input_kagome36_M0.m
%  ================================================================
%  Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
%  and Helmholtz-Zentrum Dresden-Rossendorf e.V.
%  Licensed under the Apache License, Version 2.0.
%  ================================================================
%  N=36 kagome torus on the C_6-symmetric (a,b)=(2,2) supercell (12 cells),
%  s=1/2, M=0 only, FULL space group C_6v semidirect translations (|G|=144).
%  The canonical maximally-symmetric 36-site kagome cluster.
%
%  dim(M=0) = C(36,18) = 9.08e9 ; n_reps ~ dim/144 ~ 6.3e7 ; 15 irreps with
%  dims up to d=6 (<= MAX_D=12, no kernel change). compact-V auto-enabled.
%  n_total = 2^36 > 2^32 -> bitmap impossible, schnack lookup MANDATORY.
%  Per-irrep checkpointing on (M=0-only long run; reboot-safe).
%  ================================================================

%% Required inputs
s_val   = 0.5;
J       = 1.0;
R       = 8;
M_lz    = 100;
T_range = logspace(-1, 1.5, 60);

%% Optional inputs
geometry        = 'kagome';
kag_a           = 2;
kag_b           = 2;
only_M0         = true;
ed_thresh       = 50;
mem_diag        = true;
lookup_method   = 'schnack';      % n_total = 2^36 > 2^32 -> mandatory
entries_storage = 'host';
checkpoint      = true;           % per-irrep checkpoint + resume
B_gpu           = 0;
