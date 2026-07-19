function test_stream_prefix()
%TEST_STREAM_PREFIX  Resident-prefix streaming == resident path, bit-identical.
%   With a >0 prefix budget (clt.prefix_budget), init_skel_stream keeps the
%   LEADING tiles of the entry table permanently on the GPU and streams only
%   the TAIL tiles per SpMV. The prefix is cut at tile boundaries and launched
%   with entry_base = 0, so the per-rep math is unchanged -> results must be
%   bit-identical to the resident path (dE = dW = 0), like TEST_STREAM.
%   Covers the per-entry layouts + the Lever A combination:
%     A: 4x4 C_4v d=4    UNPACKED src + uint8 g (the kagome N=36 layout),
%        sync prefix AND with FTLM_LEVER_A=1 (Lever A then pins only the
%        TAIL -> exercises the s_pin_base_e host-indexing offset);
%     B: icosahedron s=1/2 d=3  PACKED srcg prefix;
%     C: icosahedron s=1 T1g    PACKED srcg + streamed c_idx (c-index prefix).
%   The [prefix] banner is captured via evalc and parsed, so a silently
%   inactive prefix (P=0 or P=all) fails the test instead of passing vacuously.
%
%   See also TEST_STREAM, TEST_STREAM_S1, RUN_FTLM_PG_SECTOR_GPU_IH.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    gpu_h = gpuDevice;

    % Lever A env: force OFF for the sync cases, ON for the combo case; always
    % restore the caller's setting (also on error).
    la_old = getenv('FTLM_LEVER_A');
    restore = onCleanup(@() setenv('FTLM_LEVER_A', la_old));

    % --- Case A: 4x4 C_4v d=4 (unpacked src + uint8 g) ---
    group = square_lattice_spacegroup(4, 4); group.irreps = irreps_from_group(group);
    bonds = adjacency_square_lattice(4, 4);
    cache = enumerate_M_orbits_Ih_gpu(0.5, 0, group);
    entries = collect_clt_entries_Ih(cache.super_reps, bonds, 0.5, 1, group, 'bitmap', 'host');
    p = find(arrayfun(@(z) z.d, group.irreps) == 4, 1);

    setenv('FTLM_LEVER_A', '');
    check_block('4x4-c4v d=4 (sync)',    entries, cache, group.irreps(p).mats, 4, group, gpu_h, false);
    setenv('FTLM_LEVER_A', '1');
    check_block('4x4-c4v d=4 (Lever A)', entries, cache, group.irreps(p).mats, 4, group, gpu_h, true);
    setenv('FTLM_LEVER_A', '');

    % --- Case B: icosahedron s=1/2 d=3 (packed srcg) ---
    g2 = icosahedron_Ih_full(); b2 = adjacency_icosahedron_Ih();
    cache2 = enumerate_M_orbits_Ih_gpu(0.5, 0, g2);
    entries2 = collect_clt_entries_Ih(cache2.super_reps, b2, 0.5, 1, g2, 'bitmap', 'host');
    check_block('icosa s=1/2 T1g (packed)', entries2, cache2, g2.T1g, 3, g2, gpu_h, false);

    % --- Case C: icosahedron s=1 T1g (packed srcg + indexed c) ---
    cache3 = enumerate_M_orbits_Ih_gpu(1.0, 0, g2);
    entries3 = collect_clt_entries_Ih(cache3.super_reps, b2, 1.0, 1.0, g2, 'bitmap', 'host');
    assert(isfield(entries3, 'c_is_indexed') && entries3.c_is_indexed, ...
        'test_stream_prefix: expected indexed c for s=1');
    check_block('icosa s=1 T1g (c_idx)', entries3, cache3, g2.T1g, 3, g2, gpu_h, false);

    fprintf('\nPASS: resident-prefix streaming (sync, Lever A, packed, c_idx) == resident.\n');
end


% ----------------------------------------------------------------
function check_block(label, entries, cache, ir, d, group, gpu_h, want_lever_a)
    % B=2 with R=4 -> 2 Krylov blocks -> the prefix + pinned-tail pointers are
    % reused across separate 'block_lanczos' MEX calls (the N=36 scenario).
    R = 4; Mlz = 50; B = 2; seed = 999;
    [reps, V, eg, npr, ~, triv] = apply_irrep_to_orbits(cache, ir, d, group);

    % Reference: resident path (normal gpuArray borrow).
    eskel0 = build_entry_skeleton_Ih(entries);
    clt0   = build_clt_skeleton_from_entries_Ih(entries, reps, V, eg, npr, ir, d, group, triv, eskel0, true);
    [E0, w0] = run_ftlm_pg_sector_gpu_Ih(clt0, R, Mlz, B, seed, gpu_h);
    clt0 = []; wait(gpu_h); %#ok<NASGU>

    % Streaming with ~7 tiles + an explicit prefix budget of ~3 tiles.
    n_coll   = numel(entries.src_sorted);
    tiny_cap = ceil(n_coll / 7);
    eskel = build_entry_skeleton_Ih(entries, true, tiny_cap);   % force_b2 + tiny tiles
    clt   = build_clt_skeleton_from_entries_Ih(entries, reps, V, eg, npr, ir, d, group, triv, eskel, true);
    clt.force_stream = true;
    nt = numel(clt.tile_e_count);
    assert(nt >= 5, '%s: expected >=5 tiles, got %d', label, nt);
    % Device bytes/entry as streamed (src+g unpacked, srcg packed, +c_idx).
    if isfield(clt, 'srcg') && ~isempty(clt.srcg)
        bpe = 4;
    elseif isa(clt.g_idx, 'uint8')
        bpe = 5;
    else
        bpe = 6;
    end
    if isfield(clt, 'c_idx') && ~isempty(clt.c_idx), bpe = bpe + 1; end
    clt.prefix_budget = double(sum(clt.tile_e_count(1:3))) * bpe;   % exactly 3 tiles

    [out, E, w] = evalc('run_ftlm_pg_sector_gpu_Ih(clt, R, Mlz, B, seed, gpu_h)');
    clt = []; wait(gpu_h); %#ok<NASGU>

    % The prefix must really have been PARTIAL (P >= 1 resident, tail streamed).
    tok = regexp(out, '\[prefix\] (\d+)/(\d+) tiles resident', 'tokens', 'once');
    assert(~isempty(tok), '%s: [prefix] banner missing:\n%s', label, out);
    P = str2double(tok{1});
    assert(P >= 1 && P < nt, '%s: prefix not partial (P=%d of %d)', label, P, nt);
    la_seen = contains(out, '[Lever A] active');
    assert(la_seen == want_lever_a, '%s: Lever A active=%d, expected %d', ...
           label, la_seen, want_lever_a);

    dE = max(abs(E - E0)); dW = max(abs(w - w0));
    fprintf('%-26s n_tiles=%d prefix=%d leverA=%d : vs resident max|dE|=%.3e max|dW|=%.3e\n', ...
            label, nt, P, la_seen, dE, dW);
    assert(dE == 0 && dW == 0, ...
        '%s: prefix streaming differs from resident: dE=%.2e dW=%.2e', label, dE, dW);
end
