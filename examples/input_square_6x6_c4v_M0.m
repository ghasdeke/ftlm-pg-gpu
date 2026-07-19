%% input_square_6x6_c4v_M0.m
%  ================================================================
%  Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
%  and Helmholtz-Zentrum Dresden-Rossendorf e.V.
%  Licensed under the Apache License, Version 2.0.
%  ================================================================
%  6 x 6 periodic square lattice, s = 1/2, M = 0 only, FULL SPACE GROUP
%  (C_4v x translations, |G| = 8 * 36 = 288), realified FS=+1 irreps.
%  THIS IS THE N=36 SQUARE-LATTICE TARGET.
%
%  Why the space group (and NOT translation-only): with C_4v the order rises
%  36 -> 288 (~8x), so n_reps ~ dim(M=0)/288 ~ 3.15e7 and n_entries ~ 1.17e9
%  (< 2^31 -> RESIDENT, no B2/streaming). estimate_feasibility predicts
%  ~16 GB VRAM / ~27 GB host -> fits the RTX 4000 SFF Ada / 63 GB host. The
%  translation-only 6x6 (order 36) does NOT fit (~84 GB host) -- see
%  input_square_6x6_s12_M0.m's DO-NOT-RUN warning.
%
%  Irreps: 27 of the 6x6 space group, dims [8 x d1, 6 x d2, 12 x d4, 1 x d8],
%  Sum d^2 = 288. The lone d=8 irrep block is the long pole (n_basis ~ 8*n_reps
%  ~ 2.5e8); compact-V keeps its per-rep V tiny. max d = 8 <= MAX_D = 12,
%  |G| = 288 <= 65535 (g stored uint16, unpacked).
%
%  dim(M=0) = C(36,18) = 9 075 135 300. Sum-rule Sum_i w_i must equal this
%  EXACTLY (completeness), independent of R / M_lz.
%
%  lookup_method MUST be 'schnack': n_total = 2^36 > 2^32 makes the 32-state
%  bitmap CLT impossible. checkpoint=true (per-irrep): M=0-only -> per-M ckpt
%  would never fire mid-run, and this machine reboots unpredictably; per-irrep
%  ckpt makes a reboot cost at most one of the 27 irreps.
%
%  Config: R=2 / M_lz=60 fast variant (Sum-rule exact at any R/M_lz; this run
%  is the feasibility + per-irrep-timing demonstration). Expect ~2 h.
%  ================================================================

%% Required inputs
s_val   = 0.5;
J       = 1.0;
R       = 2;
M_lz    = 60;
T_range = logspace(-1, 1.5, 60);

%% Optional inputs
geometry        = 'square_lattice_c4v';   % full space group (C_4v x translations)
Lx              = 6;
Ly              = 6;
only_M0         = true;
ed_thresh       = 50;
mem_diag        = true;                    % watch VRAM/host on the first 6x6 run
lookup_method   = 'schnack';               % MANDATORY: n_total = 2^36 > 2^32
entries_storage = 'host';
checkpoint      = true;                     % per-irrep ckpt (reboot-prone machine)
precompute_cache = true;                    % cache enumerate+collect (~18 min) to
                                            % disk (~12 GB) so a resume/rerun skips
                                            % it (loads in ~seconds). Kept for reuse.
B_gpu           = 0;                        % VRAM-adaptive
