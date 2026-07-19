function test_precompute_cache()
%TEST_PRECOMPUTE_CACHE  Verify the opt-in precompute cache (precompute_cache):
%   a cache-HIT (load cache_M+entries_M from disk, skip enumerate+collect) gives
%   BIT-IDENTICAL results to the cache-MISS (first compute), and a tampered
%   config stamp is rejected (recompute). The FTLM seed is deterministic, so
%   miss and hit must agree exactly. Tiny 4x4 C_4v M=0 (same geometry path as
%   the 6x6 N=36 target).
%
%   See also FTLM_OBSERVABLES_PG_GPU_IH, PC_GATHER_STRUCT.
% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    inp  = 'input_square_4x4_c4v_pccache';
    res  = 'ftlm_pg_gpu_Ih_sq4x4sg_s1o2.mat';
    pcf  = 'precompute_sq4x4sg_s1o2_M0.mat';
    ckpt = 'ckpt_ftlm_gpu_sq4x4sg_s1o2.mat';
    files = {res, pcf, ckpt, [pcf '.tmp']};
    scrub(files);
    cleanupObj = onCleanup(@() scrub(files)); %#ok<NASGU>

    % --- Run 1: cache MISS (compute enumerate+collect, write cache) ---
    log1 = evalc('ftlm_observables_pg_gpu_Ih(inp)');
    assert(exist(pcf, 'file') == 2, 'cache file was not written');
    assert(contains(log1, '[precompute-cache] wrote'), 'missing cache-write message');
    assert(~contains(log1, '[precompute-cache] loaded'), 'unexpectedly loaded on first run');
    r1 = load(res, 'all_E', 'all_w', 'all_M');

    % --- Run 2: cache HIT (load cache, skip enumerate+collect) ---
    log2 = evalc('ftlm_observables_pg_gpu_Ih(inp)');
    assert(contains(log2, '[precompute-cache] loaded'), 'cache was NOT used on rerun');
    assert(~contains(log2, '[precompute-cache] wrote'), 'rewrote cache on a hit');
    r2 = load(res, 'all_E', 'all_w', 'all_M');

    assert(numel(r1.all_E) == numel(r2.all_E), 'spectrum length differs (hit vs miss)');
    dE = max(abs(r1.all_E - r2.all_E));
    dW = max(abs(r1.all_w - r2.all_w));
    fprintf('cache-hit vs cache-miss: max|dE|=%.3e  max|dW|=%.3e\n', dE, dW);
    assert(dE == 0 && dW == 0, 'cache HIT not bit-identical to MISS');

    % --- Tamper the stamp -> must be rejected and recomputed (not loaded) ---
    pc = load(pcf); pc.pc_cfg.perms_chk = pc.pc_cfg.perms_chk + 1;
    save(pcf, '-struct', 'pc', '-v7.3', '-nocompression');  clear pc;
    ws = warning('off', 'ftlm_observables_pg_gpu_Ih:pcMismatch');
    log3 = evalc('ftlm_observables_pg_gpu_Ih(inp)');
    warning(ws);
    assert(~contains(log3, '[precompute-cache] loaded'), 'loaded a STALE (tampered) cache!');
    assert(contains(log3, '[precompute-cache] wrote'), 'did not recompute after mismatch');
    r3 = load(res, 'all_E');
    assert(max(abs(r3.all_E - r1.all_E)) == 0, 'recompute-after-mismatch != original');

    fprintf('PASS: precompute_cache (hit==miss bit-identical; stale stamp rejected).\n');
end

% ----------------------------------------------------------------
function scrub(files)
    for i = 1:numel(files)
        if exist(files{i}, 'file') == 2, delete(files{i}); end
    end
end
