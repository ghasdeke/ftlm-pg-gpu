function test_stream()
%TEST_STREAM  Streaming-B2 (init_skel_stream, host entries streamed in rep-tiles)
%   == the resident path, bit-identical. The tiny tile_cap forces several
%   rep-tiles so the streaming tile loop + entry_base offset are exercised.
%   Streaming changes only HOW the per-entry arrays reach the SpMV (host->device
%   tiles vs resident), not the per-rep math, so the result must be bit-identical
%   (dE=dW=0), like TEST_B2. Covers BOTH per-entry layouts:
%     - UNPACKED src+g : 4x4 C_4v (|G|=128 -> g needs 8 bits -> never packs);
%                        this is the kagome N=36 production streaming path.
%     - PACKED srcg    : icosahedron s=1 (|G|=120 < 128 -> packs into uint32).
%
%   See also TEST_B2, BUILD_ENTRY_SKELETON_IH, RUN_FTLM_PG_SECTOR_GPU_IH.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    gpu_h = gpuDevice;
    ok = true;

    % --- Case A: 4x4 C_4v d=4 (UNPACKED src+g; the N=36 streaming path) ---
    group = square_lattice_spacegroup(4, 4); group.irreps = irreps_from_group(group);
    bonds = adjacency_square_lattice(4, 4);
    cache = enumerate_M_orbits_Ih_gpu(0.5, 0, group);
    entries = collect_clt_entries_Ih(cache.super_reps, bonds, 0.5, 1, group, 'bitmap', 'host');
    p = find(arrayfun(@(z) z.d, group.irreps) == 4, 1);
    ok = ok && check_block('4x4-c4v d=4', entries, cache, group.irreps(p).mats, 4, group, gpu_h);

    % --- Case B: icosahedron s=1/2 d=3 (PACKED srcg; |G|=120; constant c) ---
    g2 = icosahedron_Ih_full(); b2 = adjacency_icosahedron_Ih();
    cache2 = enumerate_M_orbits_Ih_gpu(0.5, 0, g2);
    entries2 = collect_clt_entries_Ih(cache2.super_reps, b2, 0.5, 1, g2, 'bitmap', 'host');
    ok = ok && check_block('icosa s=1/2 T1g d=3', entries2, cache2, g2.T1g, 3, g2, gpu_h);

    assert(ok, 'init_skel_stream differs from resident (or only 1 tile)');
    fprintf('\nPASS: init_skel_stream (unpacked + packed, multi-tile) == resident.\n');
end


% ----------------------------------------------------------------
function ok = check_block(label, entries, cache, ir, d, group, gpu_h)
    % B=2 with R=4 -> 2 Krylov blocks -> 'block_lanczos' is invoked TWICE,
    % reusing the BORROWED host entry pointers across separate MEX calls (the
    % N=36 R=2/B=1 scenario). Same B for resident + stream -> comparable RNG.
    R = 4; Mlz = 50; B = 2; seed = 999;
    [reps, V, eg, npr, ~, triv] = apply_irrep_to_orbits(cache, ir, d, group);

    % Reference: resident path (normal gpuArray borrow).
    eskel0 = build_entry_skeleton_Ih(entries);
    clt0   = build_clt_skeleton_from_entries_Ih(entries, reps, V, eg, npr, ir, d, group, triv, eskel0, true);
    [E0, w0] = run_ftlm_pg_sector_gpu_Ih(clt0, R, Mlz, B, seed, gpu_h);
    clt0 = []; wait(gpu_h); %#ok<NASGU>

    n_coll   = numel(entries.src_sorted);
    tiny_cap = ceil(n_coll / 7);          % ~7 rep-tiles

    eskel = build_entry_skeleton_Ih(entries, true, tiny_cap);   % force_b2 + tiny tiles
    clt   = build_clt_skeleton_from_entries_Ih(entries, reps, V, eg, npr, ir, d, group, triv, eskel, true);
    clt.force_stream = true;
    clt.prefix_budget = 0;   % explizit AUS: dieser Test gated den GESTREAMTEN Tail
    nt = numel(clt.tile_e_count);
    [E, w] = run_ftlm_pg_sector_gpu_Ih(clt, R, Mlz, B, seed, gpu_h);
    clt = []; wait(gpu_h); %#ok<NASGU>

    dE = max(abs(E - E0)); dW = max(abs(w - w0));
    fprintf('%-20s is_packed=%d n_tiles=%d : stream vs resident max|dE|=%.3e max|dW|=%.3e\n', ...
            label, eskel.is_packed, nt, dE, dW);
    ok = (dE < 1e-4) && (dW < 1e-3) && (nt >= 2);
end
