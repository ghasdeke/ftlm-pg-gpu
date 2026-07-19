function test_keep_table()
%TEST_KEEP_TABLE  Keep-table streaming reuse ('cleanup_keep_table') is
%   bit-identical to fresh per-irrep inits, actually REUSES the kept state,
%   and falls back to a full fresh init on a fingerprint mismatch.
%
%   The production pattern (out-of-core M sector): several irreps stream the
%   SAME per-M entry table; the driver marks clt.keep_table so run_ftlm's
%   end-of-sector cleanup preserves the kernel's irrep-independent table
%   state (tile partition + device tile buffers + resident prefix). The next
%   init_skel_stream fingerprints the incoming table (original host/mmap
%   pointers + full tile partition + n_reps) and skips the table setup on a
%   match -- at dodec s=3/2 on the B200 that skips a ~86-GB per-irrep disk
%   re-read. Here: two irreps of 4x4 C_4v M=0 share one eskel (exactly like
%   the driver), run (a) fresh inits, (b) keep-table, with a PARTIAL resident
%   prefix in both -- E/w must be bit-identical, and 'keep_stats' must report
%   kept=1 after the first sector and reused=1 after the second init. A third
%   init from a DIFFERENT system (icosahedron) must NOT match (reused=0) and
%   must still be correct against its own fresh reference.
%
%   See also TEST_STREAM, TEST_STREAM_PREFIX, RUN_FTLM_PG_SECTOR_GPU_IH.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    gpu_h = gpuDevice;
    R = 4; Mlz = 50; B = 2;

    % --- Shared per-M state (driver pattern): one cache/entries/eskel, two irreps.
    group = square_lattice_spacegroup(4, 4); group.irreps = irreps_from_group(group);
    bonds = adjacency_square_lattice(4, 4);
    cache = enumerate_M_orbits_Ih_gpu(0.5, 0, group);
    entries = collect_clt_entries_Ih(cache.super_reps, bonds, 0.5, 1, group, 'bitmap', 'host');
    n_coll   = numel(entries.src_sorted);
    tiny_cap = ceil(n_coll / 7);                          % ~7 rep-tiles
    eskel    = build_entry_skeleton_Ih(entries, true, tiny_cap);
    % Partial resident prefix (~half the table), same budget in BOTH runs so
    % the keep-run's REUSED prefix faces a fresh-init prefix of equal size.
    bpe = 4 + 2;  if eskel.is_packed, bpe = 4; end        % src+g / packed srcg
    pref_bytes = ceil(n_coll / 2) * bpe;

    ds  = arrayfun(@(z) z.d, group.irreps);
    igs = [find(ds == 4, 1), find(ds == 1, 1)];           % two irreps, d=4 then d=1
    seeds = [4711, 4712];

    clts = cell(1, 2);
    for k = 1:2
        ir = group.irreps(igs(k));
        [reps, V, eg, npr, ~, triv] = apply_irrep_to_orbits(cache, ir.mats, ir.d, group);
        clt = build_clt_skeleton_from_entries_Ih(entries, reps, V, eg, npr, ...
                  ir.mats, ir.d, group, triv, eskel, true);
        clt.force_stream  = true;
        clt.prefix_budget = pref_bytes;
        clts{k} = clt;
    end

    % --- Reference: fresh init + full cleanup per irrep (keep_table unset).
    E_ref = cell(1, 2); w_ref = cell(1, 2);
    for k = 1:2
        [E_ref{k}, w_ref{k}] = run_ftlm_pg_sector_gpu_Ih(clts{k}, R, Mlz, B, seeds(k), gpu_h);
    end

    % --- Keep-table run: same clts, keep_table on; verify kept/reused flags.
    E_kp = cell(1, 2); w_kp = cell(1, 2);
    for k = 1:2
        clts{k}.keep_table = true;
        [E_kp{k}, w_kp{k}] = run_ftlm_pg_sector_gpu_Ih(clts{k}, R, Mlz, B, seeds(k), gpu_h);
        ks = cuda_lanczos_clut_block_pg_Ih('keep_stats');
        assert(ks(2) == 1, 'keep_table: state not kept after sector %d', k);
        assert(ks(1) == double(k == 2), ...
            'keep_table: sector %d reused=%g (expected %d)', k, ks(1), k == 2);
        if k == 2
            assert(ks(3) > 0, 'keep_table: reused init lost the resident prefix');
        end
    end

    for k = 1:2
        dE = max(abs(E_kp{k} - E_ref{k}));  dW = max(abs(w_kp{k} - w_ref{k}));
        fprintf('irrep %d (d=%d): keep-table vs fresh  max|dE|=%.3e max|dW|=%.3e\n', ...
                k, group.irreps(igs(k)).d, dE, dW);
        assert(dE == 0 && dW == 0, 'keep-table reuse is not bit-identical (irrep %d)', k);
    end

    % --- Fingerprint mismatch: a DIFFERENT table right after a keep-cleanup
    %     must NOT reuse (fresh init) and must still be correct.
    g2 = icosahedron_Ih_full(); b2 = adjacency_icosahedron_Ih();
    cache2   = enumerate_M_orbits_Ih_gpu(0.5, 0, g2);
    entries2 = collect_clt_entries_Ih(cache2.super_reps, b2, 0.5, 1, g2, 'bitmap', 'host');
    eskel2   = build_entry_skeleton_Ih(entries2, true, ceil(numel(entries2.src_sorted) / 5));
    [reps2, V2, eg2, npr2, ~, triv2] = apply_irrep_to_orbits(cache2, g2.T1g, 3, g2);
    clt2 = build_clt_skeleton_from_entries_Ih(entries2, reps2, V2, eg2, npr2, ...
               g2.T1g, 3, g2, triv2, eskel2, true);
    clt2.force_stream = true;
    [E_mm, w_mm] = run_ftlm_pg_sector_gpu_Ih(clt2, R, Mlz, B, 4713, gpu_h);   % kept state pending
    ks = cuda_lanczos_clut_block_pg_Ih('keep_stats');
    assert(ks(1) == 0, 'keep_table: foreign table was wrongly fingerprint-matched');
    [E_mm0, w_mm0] = run_ftlm_pg_sector_gpu_Ih(clt2, R, Mlz, B, 4713, gpu_h); % no kept state
    assert(isequal(E_mm, E_mm0) && isequal(w_mm, w_mm0), ...
        'keep_table: mismatch fallback differs from a clean fresh init');

    %% --- Pre-flight reclaim (M=1 H_g postmortem, 2026-07-04): when the VRAM
    %  pre-flight cannot place even ONE Lanczos column while the kernel holds a
    %  KEPT table, run_ftlm must DROP the kept state and re-size before giving
    %  up (the kernel-side self-heal is unreachable from a pre-flight error).
    %  Under a constant fake-VRAM starvation the retry still fails -- but the
    %  kept state must be GONE afterwards, proving the reclaim branch ran.
    clts{1}.keep_table = true;
    run_ftlm_pg_sector_gpu_Ih(clts{1}, R, Mlz, B, seeds(1), gpu_h);
    ks = cuda_lanczos_clut_block_pg_Ih('keep_stats');
    assert(ks(2) == 1, 'reclaim setup: no kept state after keep-run');
    env_guard = onCleanup(@() setenv('FTLM_FAKE_FREE_VRAM_GB', ''));
    setenv('FTLM_FAKE_FREE_VRAM_GB', '1e-6');            % starve the pre-flight
    threw = false;
    try
        run_ftlm_pg_sector_gpu_Ih(clts{2}, R, Mlz, B, seeds(2), gpu_h);
    catch err
        threw = strcmp(err.identifier, 'run_ftlm_pg_sector_gpu_Ih:VRAM');
    end
    setenv('FTLM_FAKE_FREE_VRAM_GB', '');
    clear env_guard;
    assert(threw, 'starved pre-flight did not raise the VRAM error');
    ks = cuda_lanczos_clut_block_pg_Ih('keep_stats');
    assert(ks(2) == 0, 'pre-flight reclaim did NOT drop the kept table before erroring');
    % And after the starvation is lifted, the same clt runs fine (fresh init).
    [E_rc, w_rc] = run_ftlm_pg_sector_gpu_Ih(clts{2}, R, Mlz, B, seeds(2), gpu_h);
    assert(isequal(E_rc, E_ref{2}) && isequal(w_rc, w_ref{2}), ...
        'post-reclaim fresh run differs from the reference');
    fprintf('  pre-flight reclaim: kept table dropped before VRAM error, rerun exact\n');

    %% --- Kept state vs. DEVICE RESET (2026-07-09): the driver resets the
    %  GPU at startup (ftlm_observables_pg_gpu_Ih), destroying the CUDA
    %  context UNDER a kept table left pending by a previous run (exactly
    %  what this test leaves behind by design). The kernel cleanup must
    %  absorb the resulting invalid-pointer frees WITHOUT leaking a soft
    %  CUDA error into the next init's crash guard -- this reproduced as a
    %  bogus "init_skel_ref: CUDA allocation ... (invalid argument)" in the
    %  keep_table -> entries_on_disk suite sequence.
    clt2.keep_table = true;
    run_ftlm_pg_sector_gpu_Ih(clt2, R, Mlz, B, 4713, gpu_h);
    ks = cuda_lanczos_clut_block_pg_Ih('keep_stats');
    assert(ks(2) == 1, 'reset regression: setup left no kept state');
    reset(gpu_h);  gpu_h = gpuDevice;
    % Rebuild the fixture (the reset invalidated its gpuArrays) and rerun:
    % the first kernel init after the reset walks the stale-kept-state
    % cleanup; the run must succeed and match the pre-reset reference.
    eskel2 = build_entry_skeleton_Ih(entries2, true, ceil(numel(entries2.src_sorted) / 5));
    clt2r  = build_clt_skeleton_from_entries_Ih(entries2, reps2, V2, eg2, npr2, ...
                 g2.T1g, 3, g2, triv2, eskel2, true);
    clt2r.force_stream = true;
    [E_pr, w_pr] = run_ftlm_pg_sector_gpu_Ih(clt2r, R, Mlz, B, 4713, gpu_h);
    assert(isequal(E_pr, E_mm) && isequal(w_pr, w_mm), ...
        'post-reset rerun differs from the pre-reset reference');
    fprintf('  device-reset regression: stale kept state absorbed, rerun exact\n');

    %% --- Prefix DROP + RE-GROW (prod_m0 postmortem, 2026-07): the kernel's
    %  self-heal drops the kept resident prefix when an irrep's Lanczos
    %  buffers do not fit; before the re-grow fix that drop was PERMANENT for
    %  every later irrep of the kept sector (full streaming forever). Debug
    %  hook: FTLM_DEBUG_DROP_PREFIX=1 at a REUSE init drops the kept prefix
    %  AND skips the re-grow for that one init (deterministically simulates
    %  the old post-drop state -- no OOM engineering needed); the NEXT reuse
    %  init without the variable must RE-GROW the prefix. keep_stats:
    %  ks = [reused; kept; prefix_entries].
    %  REBUILD the shared-eskel fixture first: the device reset above
    %  invalidated the gpuArrays inside the original clts (entries/cache are
    %  host data and survive; E_ref are host results and stay valid).
    eskel = build_entry_skeleton_Ih(entries, true, tiny_cap);
    for k = 1:2
        ir = group.irreps(igs(k));
        [reps, V, eg, npr, ~, triv] = apply_irrep_to_orbits(cache, ir.mats, ir.d, group);
        clt = build_clt_skeleton_from_entries_Ih(entries, reps, V, eg, npr, ...
                  ir.mats, ir.d, group, triv, eskel, true);
        clt.force_stream  = true;
        clt.prefix_budget = pref_bytes;
        clt.keep_table    = true;
        clts{k} = clt;
    end
    cuda_lanczos_clut_block_pg_Ih('cleanup');           % clean slate after the reset subtest
    dp_guard = onCleanup(@() setenv('FTLM_DEBUG_DROP_PREFIX', ''));
    run_ftlm_pg_sector_gpu_Ih(clts{1}, R, Mlz, B, seeds(1), gpu_h);
    ks = cuda_lanczos_clut_block_pg_Ih('keep_stats');
    assert(ks(2) == 1 && ks(3) > 0, 'drop subtest setup: no kept resident prefix');
    setenv('FTLM_DEBUG_DROP_PREFIX', '1');              % next REUSE init: drop, no re-grow
    [E_dp, w_dp] = run_ftlm_pg_sector_gpu_Ih(clts{2}, R, Mlz, B, seeds(2), gpu_h);
    ks = cuda_lanczos_clut_block_pg_Ih('keep_stats');
    assert(ks(1) == 1, 'drop subtest: debug hook broke the reuse');
    assert(ks(3) == 0, 'debug drop had no effect (hook missing?)');
    % Prefix-vs-full-streaming decomposition is bit-identical by construction
    % (gated in test_stream_prefix), so even the dropped run must match:
    assert(isequal(E_dp, E_ref{2}) && isequal(w_dp, w_ref{2}), ...
        'dropped-prefix run is not bit-identical to the fresh reference');
    setenv('FTLM_DEBUG_DROP_PREFIX', '');
    [E_rg, w_rg] = run_ftlm_pg_sector_gpu_Ih(clts{2}, R, Mlz, B, seeds(2), gpu_h);
    ks = cuda_lanczos_clut_block_pg_Ih('keep_stats');
    assert(ks(1) == 1, 're-grow run did not reuse the kept table');
    assert(ks(3) > 0, 'prefix not re-grown after an earlier drop (re-grow regression)');
    assert(isequal(E_rg, E_ref{2}) && isequal(w_rg, w_ref{2}), ...
        're-grown prefix run is not bit-identical to the fresh reference');
    fprintf('  drop + re-grow: prefix dropped (debug hook), re-grown to %d entries, exact\n', ks(3));

    %% --- B CHANGE under keep (the prod_m0 sequence: small-d irreps at a small
    %  B, then a bigger block at B=4 on the SAME kept table): the reuse must
    %  survive the B change, the tiny test prefix must STAY resident (nothing
    %  here justifies a drop), and the result must be bit-identical to a fresh
    %  run at the SAME B -- the RNG baseline depends on B, so the reference is
    %  computed with the identical block size.
    cuda_lanczos_clut_block_pg_Ih('cleanup');
    B2 = min(4, R);
    cref = clts{2};  cref.keep_table = false;           % fresh same-B reference
    [E_b4, w_b4] = run_ftlm_pg_sector_gpu_Ih(cref, R, Mlz, B2, seeds(2), gpu_h);
    run_ftlm_pg_sector_gpu_Ih(clts{1}, R, Mlz, B, seeds(1), gpu_h);       % keep at B
    [E_kb, w_kb] = run_ftlm_pg_sector_gpu_Ih(clts{2}, R, Mlz, B2, seeds(2), gpu_h);
    ks = cuda_lanczos_clut_block_pg_Ih('keep_stats');
    assert(ks(1) == 1, 'B change broke the keep-table reuse');
    assert(ks(3) > 0, 'B change cost the resident prefix (tiny table must not drop)');
    assert(isequal(E_kb, E_b4) && isequal(w_kb, w_b4), ...
        'B change under keep is not bit-identical to the same-B fresh run');
    cuda_lanczos_clut_block_pg_Ih('cleanup');           % leave no kept state behind
    fprintf('  B change under keep: reuse + prefix intact (B %d -> %d), exact\n', B, B2);

    fprintf('\nPASS: keep-table reuse bit-identical, flags correct, mismatch falls back.\n');
end
