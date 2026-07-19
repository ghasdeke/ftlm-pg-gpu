function test_r3_stream()
%TEST_R3_STREAM  R3 pinned-ring mmap streaming == synchronous streaming,
%   bit-identical (dE = dW = 0). R3 (FTLM_R3=1) stages each mmap tile through
%   two OWNED pinned ring buffers (CPU staging thread -> pinned -> async DMA)
%   instead of the pageable synchronous memcpy; tiles, buffers and launch
%   order are IDENTICAL to the sync path, so results must match EXACTLY.
%   This is the permanent suite gate for the wave-B production path (the
%   kernel comment used to claim "test_stream gates it" -- it never did: no
%   suite test set FTLM_R3 before this one existed).
%
%   Cases (each: sync reference vs FTLM_R3=1 on the SAME spilled clt):
%     1a  4x4 C_4v d=4 realified  -- UNPACKED src+g, full streaming;
%     1b  icosahedron s=1 T1g     -- PACKED srcg + streamed c_idx, full stream;
%     2   icosahedron s=1 T1g     -- PARTIAL resident prefix + R3-staged TAIL
%         ([prefix] banner parsed: P in [1, nt-2], since R3 requires >1 tail
%         tile -- a silently full/empty prefix would make the case vacuous);
%     3   4x4 C_4v d=4 realified  -- FTLM_FP16=1 for BOTH runs (the wave-B
%         combination): FP16-sync vs FP16-R3 must also be bit-identical.
%   Every R3 run asserts the '[R3] active' banner (and the sync run its
%   absence), so an R3 setup that silently falls back FAILS instead of
%   passing vacuously. The explicit MEX 'cleanup' between the two runs is
%   MANDATORY: the R3 setup only executes on a FRESH init (!reuse) -- a
%   kept/reused table state would ignore the env change.
%
%   See also TEST_STREAM_MMAP, TEST_STREAM_PREFIX, SPILL_ENTRIES_MMAP,
%            RUN_FTLM_PG_SECTOR_GPU_IH.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    assert(exist('mmap_file', 'file') == 3, 'mmap_file MEX not built.');
    gpu_h = gpuDevice;

    % Env hygiene: R3/Lever A/FP16 are all read at kernel init; force a known
    % state for every case and restore the caller's values (also on error).
    old = {getenv('FTLM_R3'), getenv('FTLM_LEVER_A'), getenv('FTLM_FP16')};
    restore = onCleanup(@() restore_env(old));
    setenv('FTLM_R3', ''); setenv('FTLM_LEVER_A', ''); setenv('FTLM_FP16', '');

    % --- Fixture 1: 4x4 C_4v d=4, REALIFIED (unpacked src+g). Realified so
    %     the FP16 combo case can activate (FTLM_FP16 requires a real clt).
    group = square_lattice_spacegroup(4, 4);
    group.irreps = realify_irreps(irreps_from_group(group), group);
    bonds = adjacency_square_lattice(4, 4);
    cache = enumerate_M_orbits_Ih_gpu(0.5, 0, group);
    entries = collect_clt_entries_Ih(cache.super_reps, bonds, 0.5, 1, group, 'bitmap', 'host');
    p4 = find(arrayfun(@(z) z.d, group.irreps) == 4, 1);
    ir4 = group.irreps(p4).mats;

    % --- Fixture 2: icosahedron s=1 T1g (packed srcg + indexed c).
    g2 = icosahedron_Ih_full(); b2 = adjacency_icosahedron_Ih();
    cache3 = enumerate_M_orbits_Ih_gpu(1.0, 0, g2);
    entries3 = collect_clt_entries_Ih(cache3.super_reps, b2, 1.0, 1.0, g2, 'bitmap', 'host');
    assert(isfield(entries3, 'c_is_indexed') && entries3.c_is_indexed, ...
        'test_r3_stream: expected indexed c for s=1');

    % Case 1: full streaming (prefix explicitly OFF).
    check_r3('4x4-c4v d=4 (full stream)',   entries,  cache,  ir4,    4, group, gpu_h, 0, false);
    check_r3('icosa s=1 T1g (packed+cidx)', entries3, cache3, g2.T1g, 3, g2,    gpu_h, 0, false);
    % Case 2: partial resident prefix, R3 stages only the tail tiles.
    check_r3('icosa s=1 T1g (prefix+tail)', entries3, cache3, g2.T1g, 3, g2,    gpu_h, 3, false);
    % Case 3: FP16 x R3 (wave-B combination), FP16 on for BOTH runs.
    check_r3('4x4-c4v d=4 (FP16 x R3)',     entries,  cache,  ir4,    4, group, gpu_h, 0, true);

    fprintf('\nPASS: R3 pinned-ring mmap streaming == sync (full stream, prefix tail, FP16).\n');
end


% ----------------------------------------------------------------
function check_r3(label, entries, cache, ir, d, group, gpu_h, n_pref_tiles, want_fp16)
    R = 4; Mlz = 50; B = 2; seed = 999;
    [reps, V, eg, npr, ~, triv] = apply_irrep_to_orbits(cache, ir, d, group);
    n_coll = numel(entries.src_sorted);
    tiny   = ceil(n_coll / 7);                        % ~7 rep-tiles
    eskel  = build_entry_skeleton_Ih(entries, true, tiny);
    clt    = build_clt_skeleton_from_entries_Ih(entries, reps, V, eg, npr, ir, d, group, triv, eskel, true);
    clt.force_stream = true;
    nt = numel(clt.tile_e_count);
    % R3 only engages with >1 tail tile; the prefix case additionally needs
    % n_pref_tiles resident + >=2 streamed.
    assert(nt >= n_pref_tiles + 3, '%s: too few tiles (%d) for an R3 tail', label, nt);
    if n_pref_tiles > 0
        % Device bytes/entry as streamed (same derivation as test_stream_prefix).
        if isfield(clt, 'srcg') && ~isempty(clt.srcg)
            bpe = 4;
        elseif isa(clt.g_idx, 'uint8')
            bpe = 5;
        else
            bpe = 6;
        end
        if isfield(clt, 'c_idx') && ~isempty(clt.c_idx), bpe = bpe + 1; end
        clt.prefix_budget = double(sum(clt.tile_e_count(1:n_pref_tiles))) * bpe;
    else
        clt.prefix_budget = 0;   % explicit OFF: this case gates the FULL-streaming tail
    end

    % Spill to disk + memory-map (R3 requires an mmap source: s_h_is_mmap).
    fpath = [tempname, '.entries.bin'];
    [clt, mh] = spill_entries_mmap(clt, fpath);

    if want_fp16, setenv('FTLM_FP16', '1'); end

    % Run 1: synchronous reference (FTLM_R3 unset). Banner ABSENCE is asserted
    % so an env leak from a previous case cannot silently turn this into
    % R3-vs-R3 (vacuous).
    setenv('FTLM_R3', '');
    [out0, E0, w0] = evalc('run_ftlm_pg_sector_gpu_Ih(clt, R, Mlz, B, seed, gpu_h)');
    assert(~contains(out0, '[R3] active'), '%s: R3 active in the sync reference', label);
    if want_fp16
        assert(contains(out0, '[FP16]'), '%s: FP16 not active in the sync run (vacuous combo)', label);
    end
    % MANDATORY between the two runs: the R3 setup lives under `if (!reuse)` --
    % any kept/reusable table state would make the FTLM_R3=1 init skip it.
    cuda_lanczos_clut_block_pg_Ih('cleanup');  wait(gpu_h);

    % Run 2: R3 on the SAME mapped clt.
    setenv('FTLM_R3', '1');
    [out1, E1, w1] = evalc('run_ftlm_pg_sector_gpu_Ih(clt, R, Mlz, B, seed, gpu_h)');
    setenv('FTLM_R3', '');
    assert(contains(out1, '[R3] active'), '%s: R3 did not activate:\n%s', label, out1);
    if want_fp16
        assert(contains(out1, '[FP16]'), '%s: FP16 not active in the R3 run', label);
        setenv('FTLM_FP16', '');
    end
    P = nt;                                            % full-stream cases: no prefix
    if n_pref_tiles > 0
        tok = regexp(out1, '\[prefix\] (\d+)/(\d+) tiles resident', 'tokens', 'once');
        assert(~isempty(tok), '%s: [prefix] banner missing:\n%s', label, out1);
        P = str2double(tok{1});
        assert(P >= 1 && P <= nt - 2, ...
            '%s: prefix not partial with an R3-capable tail (P=%d of %d)', label, P, nt);
    end

    clt = []; wait(gpu_h); %#ok<NASGU>
    cuda_lanczos_clut_block_pg_Ih('cleanup');          % no borrowed ptr before unmap
    mmap_file('close', mh);
    if exist(fpath, 'file'), delete(fpath); end

    dE = max(abs(E1 - E0)); dW = max(abs(w1 - w0));
    fprintf('%-30s n_tiles=%d prefix=%d fp16=%d : R3 vs sync  max|dE|=%.3e max|dW|=%.3e\n', ...
            label, nt, P * (n_pref_tiles > 0), want_fp16, dE, dW);
    assert(dE == 0 && dW == 0, ...
        '%s: R3 streaming differs from sync: dE=%.2e dW=%.2e', label, dE, dW);
end


% ----------------------------------------------------------------
function restore_env(old)
    setenv('FTLM_R3',      old{1});
    setenv('FTLM_LEVER_A', old{2});
    setenv('FTLM_FP16',    old{3});
end
