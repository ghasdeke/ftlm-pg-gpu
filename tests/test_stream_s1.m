function test_stream_s1()
%TEST_STREAM_S1  s>=1 streaming (indexed c) == resident, bit-identical.
%   For s>=1 the off-diagonal coefficient varies (uint8 c_idx into a small
%   c_table). The streaming path now streams the c_idx tile alongside src/g and
%   the kernel reads c = c_table[c_idx[e]] per tile-local entry. This must match
%   the resident path exactly (the SpMV math is unchanged; only WHERE c_idx lives
%   differs). Uses the icosahedron s=1 M=0 (small; T1g d=3; packed srcg + c_idx).
%
%   See also TEST_STREAM, INIT_SKEL_STREAM (cuda_lanczos_clut_block_pg_Ih.cu).

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    gpu_h = gpuDevice;
    g = icosahedron_Ih_full();  b = adjacency_icosahedron_Ih();
    cache = enumerate_M_orbits_Ih_gpu(1.0, 0, g);         % s=1, M=0
    entries = collect_clt_entries_Ih(cache.super_reps, b, 1.0, 1.0, g, 'bitmap', 'host');
    assert(isfield(entries, 'c_is_indexed') && entries.c_is_indexed, ...
        'test_stream_s1: expected indexed c for s=1');

    R = 4; Mlz = 50; B = 2; seed = 999;
    ir = g.T1g; d = 3;
    [reps, V, eg, npr, ~, triv] = apply_irrep_to_orbits(cache, ir, d, g);

    % Resident reference (init_skel_ref, c_idx as gpuArray).
    eskel0 = build_entry_skeleton_Ih(entries);
    clt0 = build_clt_skeleton_from_entries_Ih(entries, reps, V, eg, npr, ir, d, g, triv, eskel0, true);
    [E0, w0] = run_ftlm_pg_sector_gpu_Ih(clt0, R, Mlz, B, seed, gpu_h);
    clt0 = []; wait(gpu_h); %#ok<NASGU>

    % Streaming (init_skel_stream, c_idx streamed in tiles).
    n_coll = numel(entries.src_sorted);  tiny = ceil(n_coll / 7);
    eskel = build_entry_skeleton_Ih(entries, true, tiny);
    clt = build_clt_skeleton_from_entries_Ih(entries, reps, V, eg, npr, ir, d, g, triv, eskel, true);
    clt.force_stream = true;
    clt.prefix_budget = 0;   % explizit AUS: dieser Test gated den GESTREAMTEN Tail
    nt = numel(clt.tile_e_count);
    [E, w] = run_ftlm_pg_sector_gpu_Ih(clt, R, Mlz, B, seed, gpu_h);
    clt = []; wait(gpu_h); %#ok<NASGU>

    dE = max(abs(E - E0));  dW = max(abs(w - w0));
    fprintf('icosa s=1 T1g d=3  is_packed=%d n_tiles=%d : stream vs resident max|dE|=%.3e max|dW|=%.3e\n', ...
            eskel.is_packed, nt, dE, dW);
    assert(dE == 0 && dW == 0 && nt >= 2, ...
        's>=1 streaming differs from resident (or only 1 tile): dE=%.2e dW=%.2e nt=%d', dE, dW, nt);
    fprintf('\nPASS: s>=1 streaming (indexed c) == resident (bit-identical).\n');
end
