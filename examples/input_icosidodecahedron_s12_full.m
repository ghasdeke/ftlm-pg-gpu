%% input_icosidodecahedron_s12_full.m
%  ================================================================
%  Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
%  and Helmholtz-Zentrum Dresden-Rossendorf e.V.
%  Licensed under the Apache License, Version 2.0.
%  ================================================================
%  s = 1/2 Heisenberg icosidodecahedron (N=30), FULL M sweep: ALL M = 0..15
%  sectors x ALL 10 I_h irreps. Unlike the M=0-only input this gives the
%  PHYSICAL thermodynamics C(T) AND chi(T) (chi != 0 needs M-fluctuations).
%  The driver prints a per-(M,Gamma) timing line for every sector.
%
%  n_total = 2^30 = 1 073 741 824 < 2^32  -> sum-rule target = n_total exactly.
%  M_max = N*s = 15. Largest sector is M=0 (dim C(30,15)=1.551e8); higher-M
%  sectors shrink fast, so the full sweep is only ~4x the M=0-only cost.
%  checkpoint=true (per-(M,irrep)) -> reboot-safe on this machine.
%  ================================================================

%% Required inputs
s_val   = 0.5;
J       = 1.0;
R       = 8;
M_lz    = 100;
T_range = logspace(-1, 1.5, 60);

%% Optional inputs
geometry      = 'icosidodecahedron';
only_M0       = false;              % <-- FULL M>=0 sweep (physical C(T) + chi(T))
ed_thresh     = 50;                % dense ED for tiny blocks, FTLM otherwise
mem_diag      = false;             % clean log; per-(M,irrep) timing lines still print
lookup_method = 'schnack';         % matches the validated M=0 run (n_total<2^32 -> bitmap also OK)
entries_storage = 'host';
checkpoint    = true;              % per-(M,irrep) checkpoint (reboot-prone machine)
B_gpu         = 0;                 % VRAM-adaptive
