%% input_kagome36_M0_stream.m
%  N=36 kagome torus (2,2), M=0, FULL space group C_6v (|G|=144), s=1/2.
%  STREAMING-B2 validation run (2026-06-04): the d>=3 blocks do NOT fit the
%  20.5 GB card with the ~11.7 GB entry table resident (d=6 needs ~25 GB), so
%  run_ftlm auto-STREAMS the host entry table in rep-tiles per SpMV. Two blocks:
%    d3_9  (d=3, n_basis~1.9e8) -- streams; faster, validates streaming at scale
%    d6_14 (d=6, n_basis~3.8e8) -- streams; the 50%-of-M=0 block, infeasible
%                                  before streaming-B2.
%  Verified bit-identical to the resident path at small scale (test_stream:
%  unpacked + packed, multi-tile, dE=dW=0). R=2, M_lz=60 (speed/feasibility;
%  convergence study still owed). Per-irrep checkpoint on.
s_val   = 0.5;
J       = 1.0;
R       = 2;
M_lz    = 60;
T_range = logspace(-1, 1.5, 60);

geometry        = 'kagome';
kag_a           = 2;
kag_b           = 2;
only_M0         = true;
ed_thresh       = 50;
mem_diag        = true;
lookup_method   = 'schnack';
entries_storage = 'host';
checkpoint      = true;
B_gpu           = 0;
irrep_list      = {'d3_9', 'd6_14'};
