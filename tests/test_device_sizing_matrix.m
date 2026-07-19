function test_device_sizing_matrix()
%TEST_DEVICE_SIZING_MATRIX  Portability matrix: EVERY card size must run.
%
%   User contract (2026-07): "no gambling whether the code runs on a
%   different card" -- on ANY VRAM size the pipeline must complete
%   CORRECTLY (possibly slower via smaller B / streaming / partial
%   prefix), NEVER hard-abort, and carry no hidden per-device
%   calibrations. This test sweeps EMULATED card sizes through the
%   FTLM_FAKE_FREE_VRAM_GB hook (see GPU_FREE_BYTES; run_ftlm reads free
%   VRAM as min(real, fake), so emulation never exceeds the real card):
%
%       sizes  : [4 8 12 20 48 96 180] GB   x   precision {FP32, FP16}
%                (+ a sub-GB clamp ramp 0.5625/0.57/0.6 GB where B_vram
%                 actually BINDS on these tiny fixtures -- see below)
%       fixture RESIDENT   : 4x4 C_4v d=4, default skeleton
%                            (init_skel_ref / resident-B2 chooser side)
%       fixture STREAM-AUTO: same block, force_b2 + tiny tiles +
%                            force_stream + prefix_budget = -1 (AUTO)
%                            (init_skel_stream + kernel AUTO prefix side)
%
%   Assertions per (fixture, precision, size) cell:
%     (1) the run completes WITHOUT error. A hard abort is tolerated ONLY
%         when even B = 1 provably does not fit the emulated card by the
%         driver's own sizing formulas -- and then the error must carry the
%         id 'run_ftlm_pg_sector_gpu_Ih:VRAM' and QUANTIFY the need in GB.
%         (A dedicated sub-B=1 probe at 0.3 GB exercises exactly this
%         contract; none of the swept sizes may abort on these fixtures.)
%     (2) B_used is a power of two (coalesced-gather policy, 2026-07-11)
%         and monotone non-decreasing in the card size.
%     (3) E/w are BIT-IDENTICAL to a reference computed WITHOUT the fake
%         env at the same forced B (one cached reference per
%         fixture x precision x B): the card size may change HOW the
%         table/vectors are placed, never WHAT is computed.
%     (4) the kernel banner '[prefix] device alloc/copy failed' never
%         appears (a silently degraded prefix is a broken sizing formula,
%         not an acceptable fallback), and FP16 cells must show the
%         '[FP16]' banner (non-vacuity).
%
%   Cells whose emulated size exceeds the REAL free VRAM are skipped and
%   logged (the hook clamps via min(), so they would silently test the
%   real card again and corrupt the monotonicity sequence).
%
%   KNOWN LIMIT (documented, not gated here): the kernel's AUTO-prefix
%   budget reads raw cudaMemGetInfo and IGNORES the fake env, so this
%   matrix exercises the MATLAB-side chooser/B sizing under emulation,
%   while the kernel prefix still sizes itself to the real card. Once the
%   kernel honours the hook (inventory finding, effective_free_vram), the
%   same matrix covers both halves without changes here.
%
%   Runtime target: < 6 min on the 20-GB RTX 4000 SFF Ada (sizes > real
%   free VRAM skip there; the fixtures are tiny, so each cell is seconds).
%
%   See also GPU_FREE_BYTES, RUN_FTLM_PG_SECTOR_GPU_IH, TEST_STREAM,
%            TEST_STREAM_PREFIX, TEST_GPU_SIZING_INVARIANTS, RUN_ALL_TESTS.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    gpu_h = gpuDevice;

    % Save/restore BOTH env knobs (also on error).
    prev_fake = getenv('FTLM_FAKE_FREE_VRAM_GB');
    prev_fp16 = getenv('FTLM_FP16');
    restore = onCleanup(@() restore_env(prev_fake, prev_fp16));
    setenv('FTLM_FAKE_FREE_VRAM_GB', '');
    setenv('FTLM_FP16', '');

    %% Fixtures: built ONCE on the REAL card (no fake env) -- the matrix
    %  sweeps ONLY run_ftlm's per-sector sizing, not enumerate/collect.
    %  REALIFIED irreps so the FP16 rows are non-vacuous (FP16 requires a
    %  real clt; the real path is the production default anyway).
    group = square_lattice_spacegroup(4, 4);
    group.irreps = realify_irreps(irreps_from_group(group), group);
    bonds = adjacency_square_lattice(4, 4);
    cache = enumerate_M_orbits_Ih_gpu(0.5, 0, group);
    entries = collect_clt_entries_Ih(cache.super_reps, bonds, 0.5, 1, group, 'bitmap', 'host');
    p  = find(arrayfun(@(z) z.d, group.irreps) == 4, 1);
    ir = group.irreps(p).mats;
    assert(isreal(ir), 'test_device_sizing_matrix: realified C_4v d=4 irrep is not real');
    [reps, V, eg, npr, ~, triv] = apply_irrep_to_orbits(cache, ir, 4, group);

    % Fixture 1: RESIDENT (default skeleton; small system -> init_skel_ref).
    eskel_res = build_entry_skeleton_Ih(entries);
    clt_res   = build_clt_skeleton_from_entries_Ih(entries, reps, V, eg, npr, ...
                                                   ir, 4, group, triv, eskel_res, true);
    assert(isfield(clt_res, 'is_real') && clt_res.is_real, ...
        'test_device_sizing_matrix: resident fixture is not real (FP16 rows vacuous)');

    % Fixture 2: STREAM-AUTO (the test_stream streaming fixture: force_b2 +
    % ~7 tiny rep-tiles + force_stream), but with the PRODUCTION prefix
    % default: prefix_budget = -1 (AUTO) -- unlike test_stream, which pins
    % it to 0 to gate the fully streamed tail.
    tiny_cap = ceil(numel(entries.src_sorted) / 7);
    eskel_st = build_entry_skeleton_Ih(entries, true, tiny_cap);
    clt_st   = build_clt_skeleton_from_entries_Ih(entries, reps, V, eg, npr, ...
                                                  ir, 4, group, triv, eskel_st, true);
    clt_st.force_stream  = true;
    clt_st.prefix_budget = -1;     % AUTO (kernel sizes the resident prefix)
    assert(numel(clt_st.tile_e_count) >= 2, ...
        'test_device_sizing_matrix: streaming fixture has < 2 tiles (chooser untested)');

    fixtures = {clt_res, 'resident'; clt_st, 'stream-auto'};
    precs    = {'',  'fp32'; '1', 'fp16'};
    % [4..180]: the specified card matrix. On these tiny fixtures B is
    % R-clamped there (bytes_per_col ~ 4e6), so the VRAM clamp would stay
    % untested -- the sub-GB CLAMP RAMP prepends cells where B_vram BINDS
    % (with current margins: B ~ 1 / 2 / 4), making the pow2+monotonicity
    % assertion non-trivial. The ramp cells carry no hardcoded expected B
    % (margin changes shift them); each either runs (generic assertions)
    % or refuses quantified via the b1_fits-checked branch.
    sizes_gb = [0.5625 0.57 0.6, 4 8 12 20 48 96 180];

    R = 4;  Mlz = 40;  seed = 777;
    refs = containers.Map('KeyType', 'char', 'ValueType', 'any');

    %% The matrix.
    for f = 1 : size(fixtures, 1)
        clt = fixtures{f, 1};  fname = fixtures{f, 2};
        for q = 1 : size(precs, 1)
            setenv('FTLM_FP16', precs{q, 1});  pname = precs{q, 2};
            B_prev = 0;
            for gb = sizes_gb
                % Live free VRAM (the hook clamps via min(); a fake above
                % the real free would silently re-test the real card and
                % break the monotonicity sequence) -> skip + log.
                real_free = double(gpu_h.AvailableMemory);
                if gb * 1e9 > real_free
                    fprintf('  %-11s %-4s %7g GB : SKIP (real free %.1f GB < emulated)\n', ...
                            fname, pname, gb, real_free / 1e9);
                    continue;
                end
                setenv('FTLM_FAKE_FREE_VRAM_GB', sprintf('%g', gb));
                [E, w, B_used, out, ME] = run_cell(clt, R, Mlz, 0, seed, gpu_h);
                setenv('FTLM_FAKE_FREE_VRAM_GB', '');
                cell_id = sprintf('%s/%s/%gGB', fname, pname, gb);

                % (1) No hard abort -- UNLESS even B=1 provably cannot fit
                % by the driver's own sizing formulas; then the refusal must
                % identify itself and quantify the need in GB. (On these
                % tiny fixtures at >= 4 GB this branch is unreachable; it
                % encodes the contract for future/bigger fixtures.)
                if ~isempty(ME)
                    assert(~b1_fits(clt, gb * 1e9, strcmp(pname, 'fp16')), ...
                        '%s: hard abort although B=1 fits by the sizing formulas: %s', ...
                        cell_id, ME.message);
                    assert(strcmp(ME.identifier, 'run_ftlm_pg_sector_gpu_Ih:VRAM'), ...
                        '%s: legitimate refusal but wrong id "%s"', cell_id, ME.identifier);
                    assert(~isempty(regexp(ME.message, '\d+\.\d+\s*GB', 'once')), ...
                        '%s: refusal does not quantify the need in GB: %s', cell_id, ME.message);
                    fprintf('  %-11s %-4s %7g GB : tolerated quantified refusal (B=1 cannot fit)\n', ...
                            fname, pname, gb);
                    continue;   % no B/bit-identity checks on a refused cell
                end

                % (4) No silent kernel-prefix degradation; FP16 non-vacuous.
                assert(~contains(out, 'device alloc/copy failed'), ...
                    '%s: kernel banner "device alloc/copy failed" (broken prefix sizing)', cell_id);
                if strcmp(pname, 'fp16')
                    assert(contains(out, '[FP16]'), '%s: FTLM_FP16=1 did not activate', cell_id);
                end

                % (2) Power-of-two + monotone non-decreasing in card size.
                assert(B_used >= 1 && bitand(B_used, B_used - 1) == 0, ...
                    '%s: B_used=%d is not a power of two', cell_id, B_used);
                assert(B_used >= B_prev, ...
                    '%s: B_used=%d < %d at the previous (smaller) size -- non-monotone', ...
                    cell_id, B_used, B_prev);
                B_prev = B_used;

                % (3) Bit-identity vs the no-fake reference at the same B
                % (cached per fixture x precision x B; the forced-B reference
                % bypasses only the adaptive CHOICE, not the computation).
                key = sprintf('%s|%s|B%d', fname, pname, B_used);
                if ~refs.isKey(key)
                    [E0, w0, B0, ~, ME0] = run_cell(clt, R, Mlz, B_used, seed, gpu_h);
                    if ~isempty(ME0)   % (no assert: message args evaluate eagerly)
                        error('test_device_sizing_matrix:ref', ...
                            '%s: reference run failed: %s', key, ME0.message);
                    end
                    assert(B0 == B_used, '%s: reference clamped B %d -> %d', key, B_used, B0);
                    refs(key) = {E0, w0};
                end
                ref = refs(key);
                assert(isequal(E, ref{1}) && isequal(w, ref{2}), ...
                    '%s: E/w differ from the same-B reference (card size changed the RESULT)', ...
                    cell_id);
                fprintf('  %-11s %-4s %7g GB : B=%d  bit-identical to reference\n', ...
                        fname, pname, gb, B_used);
            end
        end
        setenv('FTLM_FP16', '');
    end

    %% Sub-B=1 probe (0.3 GB): the ONLY tolerated hard abort, and it must
    %  identify itself and QUANTIFY the need. 0.9*0.3e9 - 0.5e9 < 0, so not
    %  even one Lanczos column fits the driver's budget on either fixture.
    for f = 1 : size(fixtures, 1)
        clt = fixtures{f, 1};  fname = fixtures{f, 2};
        assert(~b1_fits(clt, 0.3e9, false), ...
            'probe mis-sized: B=1 fits a 0.3 GB card on %s', fname);
        setenv('FTLM_FAKE_FREE_VRAM_GB', '0.3');
        [~, ~, ~, ~, ME] = run_cell(clt, R, Mlz, 0, seed, gpu_h);
        setenv('FTLM_FAKE_FREE_VRAM_GB', '');
        assert(~isempty(ME), '%s @0.3 GB: expected a hard VRAM refusal, got none', fname);
        assert(strcmp(ME.identifier, 'run_ftlm_pg_sector_gpu_Ih:VRAM'), ...
            '%s @0.3 GB: wrong error id "%s" (message: %s)', fname, ME.identifier, ME.message);
        assert(~isempty(regexp(ME.message, '\d+\.\d+\s*GB', 'once')), ...
            '%s @0.3 GB: refusal does not quantify the need in GB: %s', fname, ME.message);
        fprintf('  %-11s %-4s  0.3 GB : quantified VRAM refusal (id + GB figures) OK\n', ...
                fname, 'fp32');
    end

    fprintf(['\nPASS: device-sizing matrix -- every emulated card size runs ', ...
             'bit-identically (B power-of-two, monotone; refusal only below B=1, quantified).\n']);
end


% ----------------------------------------------------------------
function [E, w, B_used, out, ME] = run_cell(clt, R, Mlz, B_gpu, seed, gpu_h) %#ok<INUSD> -- args read inside evalc
%   One matrix cell: run_ftlm with captured output AND captured error.
%   evalc keeps the banner text ('[FP16]', '[prefix] ...', '[run_ftlm] ...')
%   for the per-cell asserts; pre-cleared outputs prevent stale carry-over.
    E = [];  w = [];  B_used = NaN;  ME = [];
    out = evalc(['try, [E, w, B_used] = run_ftlm_pg_sector_gpu_Ih(', ...
                 'clt, R, Mlz, B_gpu, seed, gpu_h); catch ME, end']);
end


% ----------------------------------------------------------------
function tf = b1_fits(clt, avail_bytes, fp16)
%   Mirror of run_ftlm's sizing: does B = 1 fit an `avail_bytes` card?
%   Kept formula-parallel on purpose -- if run_ftlm's budget drifts, the
%   probe/abort legitimacy check here must drift WITH it (update both).
    n_basis = double(clt.n_basis);
    if isfield(clt, 'is_real') && clt.is_real
        per_elem = 20;  if fp16, per_elem = 12.5; end
        bytes_per_col = n_basis * per_elem + 4e6;
    else
        bytes_per_col = n_basis * 36 + 4e6;
    end
    % Reserved VRAM: one streaming tile (force_stream fixtures) or the
    % resident B2 table; the tiny resident fixture reserves nothing.
    reserved = 0;
    if isfield(clt, 'force_stream') && clt.force_stream
        if isfield(clt, 'srcg') && ~isempty(clt.srcg)
            bpe = 4;                                   % packed src|g
        elseif isa(clt.g_idx, 'uint8')
            bpe = 5;                                   % src(4) + uint8 g
        else
            bpe = 6;                                   % src(4) + uint16 g
        end
        if isfield(clt, 'c_idx') && ~isempty(clt.c_idx), bpe = bpe + 1; end
        reserved = double(max(clt.tile_e_count)) * bpe;
    end
    tf = (0.90 * (avail_bytes - reserved) - 0.5e9) >= bytes_per_col;
end


% ----------------------------------------------------------------
function restore_env(prev_fake, prev_fp16)
    setenv('FTLM_FAKE_FREE_VRAM_GB', prev_fake);
    setenv('FTLM_FP16', prev_fp16);
end
