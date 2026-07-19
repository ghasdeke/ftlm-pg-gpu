function test_fp16_smoke()
%TEST_FP16_SMOKE  FP16 STORAGE of the real Lanczos vectors (FTLM_FP16=1):
%   default-env bit gate + a u16*W accuracy envelope against the FP32 chain.
%
%   The FP16 lever stores the three real Krylov buffers
%   (v/vp/w) as __half; ALL arithmetic stays fp32 (load = __half2float,
%   store = __float2half_rn). Gates:
%
%   (1) DEFAULT-ENV BIT GATE: only the literal '1' activates the lever --
%       FTLM_FP16 unset and FTLM_FP16='0' must produce bit-identical AL/BE
%       (s_fp16=false takes the plain fp32 template path). The suite-wide
%       "old binary == new binary under default env" statement is carried by
%       run_all_tests staying green; this test pins the env PARSING.
%   (2) TOLERANCE GATE (scientific envelope): per Lanczos step the storage
%       rounding acts like a perturbation dH with ||dH|| <= u16*||H||,
%       u16 = 2^-11 (half has an 11-bit significand). With
%       W = max|AL| + 2*max(BE) -- a Gershgorin bound on spec(T) -- Weyl
%       gives |dRitz| <= c*u16*W; c = 8 covers the accumulation over the
%       reorthogonalisation-free steps. Gated: the per-column ground-state
%       Ritz value E0 over the FULL chain (variationally stable, Weyl
%       applies cleanly), and AL/BE ELEMENTWISE over the first 10 steps
%       only -- raw Lanczos coefficients are FORWARD-UNSTABLE: once the
%       first Ritz pair converges, ANY perturbation (fp16 or not) makes
%       the chains decohere to O(W) while remaining valid Lanczos runs of
%       the same H (measured locally: full-chain |dAL| ~ 2 at |dE0| ~
%       9e-5). The stable observables are the Ritz values/weights, gated
%       here and in (5). BE >= 0 everywhere (ghost-collapse guard).
%   (3) NON-REAL GUARD: FTLM_FP16=1 with a force_complex clt must NOT
%       activate (the kernel requires s_is_real) -> bit-identical to fp32.
%   (4) PATH COVERAGE: resident (init_skel_ref), full streaming AND partial
%       resident prefix (wave B runs prefix+stream) on the unpacked 4x4
%       C_4v layout, plus resident+streamed on the packed-srcg + c_idx
%       icosahedron s=1 layout -- all three OTF dispatch sites carry an
%       s_fp16 branch and each needs a gate.
%   (5) DRIVER SMOKE: run_ftlm under FTLM_FP16=1 -- the beta=0 sum rule
%       sum(w) = D_eff holds to double roundoff (FP16-independent; catches
%       index errors), and min(E) stays inside the same envelope.
%
%   See also TEST_REAL_KERNEL, RUN_FTLM_PG_SECTOR_GPU_IH, RUN_ALL_TESTS.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    gpu_h = gpuDevice;
    assert_kernel_abi();  % direct-MEX caller: bypasses the driver's handshake

    fp_old  = getenv('FTLM_FP16');
    restore = onCleanup(@() setenv('FTLM_FP16', fp_old));
    setenv('FTLM_FP16', '');

    %% Fixture A: 4x4 C_4v d=4 REALIFIED (unpacked src+g; FP16 needs a real clt).
    group = square_lattice_spacegroup(4, 4);
    group.irreps = realify_irreps(irreps_from_group(group), group);
    bonds = adjacency_square_lattice(4, 4);
    cache = enumerate_M_orbits_Ih_gpu(0.5, 0, group);
    entries = collect_clt_entries_Ih(cache.super_reps, bonds, 0.5, 1, group, 'bitmap', 'host');
    p4  = find(arrayfun(@(z) z.d, group.irreps) == 4, 1);
    ir4 = group.irreps(p4).mats;
    assert(isreal(ir4), 'test_fp16_smoke: realified C_4v d=4 irrep is not real');

    check_fp16('4x4-c4v d=4 (unpacked)', entries, cache, ir4, 4, group, 'ref');
    check_fp16('4x4-c4v d=4 (unpacked)', entries, cache, ir4, 4, group, 'stream');
    check_fp16('4x4-c4v d=4 (unpacked)', entries, cache, ir4, 4, group, 'prefix');

    %% Fixture B: icosahedron s=1 T1g realified (PACKED srcg + indexed c).
    gI = icosahedron_Ih_full();  bI = adjacency_icosahedron_Ih();
    irrI = struct('name', 'T1g', 'd', 3, 'mats', gI.T1g);
    irrI = realify_irreps(irrI, gI);
    assert(isreal(irrI.mats), 'test_fp16_smoke: realified I_h T1g is not real');
    cacheC = enumerate_M_orbits_Ih_gpu(1.0, 0, gI);
    entriesC = collect_clt_entries_Ih(cacheC.super_reps, bI, 1.0, 1.0, gI, 'bitmap', 'host');
    assert(isfield(entriesC, 'c_is_indexed') && entriesC.c_is_indexed, ...
        'test_fp16_smoke: expected indexed c for s=1');
    check_fp16('icosa s=1 T1g (c_idx)', entriesC, cacheC, irrI.mats, 3, gI, 'ref');
    check_fp16('icosa s=1 T1g (c_idx)', entriesC, cacheC, irrI.mats, 3, gI, 'stream');

    %% Non-real guard: FTLM_FP16=1 must be INERT on a force_complex clt.
    check_complex_guard('4x4-c4v d=4 force_complex', entries, cache, ir4, 4, group);

    %% Driver smoke: sum rule + E0 envelope through run_ftlm (resident path).
    check_driver_smoke(entries, cache, ir4, 4, group, gpu_h);

    fprintf('\nPASS: FP16 storage -- default env bit-inert, u16*W envelope holds, complex path inert.\n');
end


% ----------------------------------------------------------------
function check_fp16(label, entries, cache, ir, d, group, mode)
    B = 2;  Mlz = 40;
    U16 = 2^-11;  C_ACC = 8;   % see the header: storage-rounding envelope
    [reps, V, eg, npr, ~, triv] = apply_irrep_to_orbits(cache, ir, d, group);
    if strcmp(mode, 'ref')
        eskel = build_entry_skeleton_Ih(entries);
    else
        eskel = build_entry_skeleton_Ih(entries, true, ceil(numel(entries.src_sorted) / 5));
    end
    clt = build_clt_skeleton_from_entries_Ih(entries, reps, V, eg, npr, ir, d, group, triv, eskel, true);
    assert(isfield(clt, 'is_real') && clt.is_real, ...
        'check_fp16(%s): fixture is not real -- FP16 could never activate (vacuous)', label);

    gpurng(31415);
    V0 = gpuArray.randn(clt.n_basis, B, 'single');
    Ve = gpuArray(single([]));                         % real path: empty V_im

    % (1) fp32 reference (env unset) + the '0'-alias bit gate.
    setenv('FTLM_FP16', '');
    out = evalc('kernel_init(clt, B, mode)');
    assert(~contains(out, '[FP16]'), '%s: FP16 active with env unset', label);
    [AL0, BE0] = cuda_lanczos_clut_block_pg_Ih('block_lanczos', V0, Ve, Mlz);
    cuda_lanczos_clut_block_pg_Ih('cleanup');
    setenv('FTLM_FP16', '0');
    out = evalc('kernel_init(clt, B, mode)');
    assert(~contains(out, '[FP16]'), '%s: FP16 active with env=''0''', label);
    [ALz, BEz] = cuda_lanczos_clut_block_pg_Ih('block_lanczos', V0, Ve, Mlz);
    cuda_lanczos_clut_block_pg_Ih('cleanup');
    setenv('FTLM_FP16', '');
    assert(isequal(gather(AL0), gather(ALz)) && isequal(gather(BE0), gather(BEz)), ...
        '%s: FTLM_FP16=''0'' is not bit-identical to unset (default-env gate)', label);

    % (2) FP16 run; the banner assert makes the case non-vacuous.
    setenv('FTLM_FP16', '1');
    out = evalc('kernel_init(clt, B, mode)');
    setenv('FTLM_FP16', '');
    assert(contains(out, '[FP16]'), '%s: FTLM_FP16=1 did not activate', label);
    [AL1, BE1] = cuda_lanczos_clut_block_pg_Ih('block_lanczos', V0, Ve, Mlz);
    cuda_lanczos_clut_block_pg_Ih('cleanup');

    a0 = double(gather(AL0));  b0 = double(gather(BE0));
    a1 = double(gather(AL1));  b1 = double(gather(BE1));
    assert(isequal(size(a1), size(a0)) && isequal(size(b1), size(b0)), ...
        '%s: FP16 chain length differs from fp32 (early breakdown?)', label);
    W   = max(abs(a0(:))) + 2 * max(b0(:));           % Gershgorin bound on spec(T)
    tol = C_ACC * U16 * W;
    assert(all(b1(:) >= 0), '%s: negative beta under FP16 (ghost collapse)', label);

    % Elementwise envelope over an a-priori EARLY window (first 10 steps,
    % before Ritz convergence can decohere the chains -- see header (2)); the
    % full-chain deviation is printed as a diagnostic but NOT gated (Lanczos
    % coefficients are forward-unstable; the gated stable object is E0 below).
    M_CMP = min(10, size(a0, 1));
    dA  = max(max(abs(a1(1:M_CMP, :) - a0(1:M_CMP, :)), [], 1));
    dB  = max(max(abs(b1(1:M_CMP, :) - b0(1:M_CMP, :)), [], 1));
    dAf = max(abs(a1(:) - a0(:)));                    % diagnostic only
    e0_32 = ritz_e0(a0, b0);  e0_16 = ritz_e0(a1, b1);
    dE0 = max(abs(e0_16 - e0_32));
    fprintf('%-26s %-7s: early%2d |dAL|=%.2e |dBE|=%.2e |dE0|=%.2e (tol %.2e, full-chain dAL %.1e)\n', ...
            label, mode, M_CMP, dA, dB, dE0, tol, dAf);
    assert(dA <= tol && dB <= tol, ...
        '%s: early-window AL/BE outside the u16*W envelope (dA=%.2e dB=%.2e tol=%.2e)', ...
        label, dA, dB, tol);
    assert(dE0 <= tol, '%s: E0 outside the u16*W envelope (dE0=%.2e tol=%.2e)', label, dE0, tol);
end


% ----------------------------------------------------------------
function check_complex_guard(label, entries, cache, ir, d, group)
    B = 2;  Mlz = 25;
    [reps, V, eg, npr, ~, triv] = apply_irrep_to_orbits(cache, ir, d, group);
    eskel = build_entry_skeleton_Ih(entries);
    clt_c = build_clt_skeleton_from_entries_Ih(entries, reps, V, eg, npr, ir, d, group, triv, eskel, true, true);
    assert(~clt_c.is_real, 'check_complex_guard(%s): force_complex clt is real', label);

    gpurng(31415);
    V0 = gpuArray.randn(clt_c.n_basis, B, 'single');
    Vz = gpuArray.zeros(clt_c.n_basis, B, 'single');   % complex path: explicit V_im = 0

    setenv('FTLM_FP16', '');
    kernel_init(clt_c, B, 'ref');
    [AL0, BE0] = cuda_lanczos_clut_block_pg_Ih('block_lanczos', V0, Vz, Mlz);
    cuda_lanczos_clut_block_pg_Ih('cleanup');

    setenv('FTLM_FP16', '1');
    out = evalc('kernel_init(clt_c, B, ''ref'')');
    setenv('FTLM_FP16', '');
    assert(~contains(out, '[FP16]'), '%s: FP16 activated on a COMPLEX clt', label);
    [AL1, BE1] = cuda_lanczos_clut_block_pg_Ih('block_lanczos', V0, Vz, Mlz);
    cuda_lanczos_clut_block_pg_Ih('cleanup');
    assert(isequal(gather(AL0), gather(AL1)) && isequal(gather(BE0), gather(BE1)), ...
        '%s: FTLM_FP16=1 changed the complex path (must be inert)', label);
    fprintf('%-26s ref    : FP16 env inert on complex clt (bit-identical)\n', label);
end


% ----------------------------------------------------------------
function check_driver_smoke(entries, cache, ir, d, group, gpu_h)
    R = 4;  Mlz = 40;  B = 2;  seed = 2718;
    U16 = 2^-11;  C_ACC = 8;
    [reps, V, eg, npr, ~, triv] = apply_irrep_to_orbits(cache, ir, d, group);
    eskel = build_entry_skeleton_Ih(entries);
    clt = build_clt_skeleton_from_entries_Ih(entries, reps, V, eg, npr, ir, d, group, triv, eskel, true);
    assert(clt.is_real, 'driver smoke: fixture is not real');
    D_eff = double(clt.n_basis);

    setenv('FTLM_FP16', '');
    [E32, w32] = run_ftlm_pg_sector_gpu_Ih(clt, R, Mlz, B, seed, gpu_h);
    setenv('FTLM_FP16', '1');
    [out, E16, w16] = evalc('run_ftlm_pg_sector_gpu_Ih(clt, R, Mlz, B, seed, gpu_h)');
    setenv('FTLM_FP16', '');
    assert(contains(out, '[FP16]'), 'driver smoke: FP16 did not activate through run_ftlm');

    % beta=0 sum rule: sum_k |q_k(1)|^2 = 1 per column (double eig of T), so
    % sum(w) = D_eff to double roundoff -- INDEPENDENT of the fp16 storage.
    sr32 = abs(sum(w32) - D_eff) / D_eff;
    sr16 = abs(sum(w16) - D_eff) / D_eff;
    W    = max(E32) - min(E32);
    dE0  = abs(min(E16) - min(E32));
    fprintf('driver smoke: sum-rule rel err fp32=%.2e fp16=%.2e, |dE0|=%.2e (tol %.2e)\n', ...
            sr32, sr16, dE0, C_ACC * U16 * W);
    assert(sr32 < 1e-10, 'driver smoke: fp32 run breaks the beta=0 sum rule');
    assert(sr16 < 1e-10, 'driver smoke: FP16 run breaks the beta=0 sum rule (index error?)');
    assert(dE0 <= C_ACC * U16 * W, 'driver smoke: FP16 ground state outside the u16*W envelope');
end


% ----------------------------------------------------------------
function e0 = ritz_e0(AL, BE)
%   Per-column ground-state Ritz value: T = tridiag(BE(1:m-1), AL, BE(1:m-1));
%   BE(m) is the trailing residual beta (not part of T), as in solve_tridiag.
    m  = size(AL, 1);
    e0 = zeros(1, size(AL, 2));
    for b = 1:size(AL, 2)
        T = diag(AL(:, b)) + diag(BE(1:m-1, b), 1) + diag(BE(1:m-1, b), -1);
        e0(b) = min(eig(T));
    end
end


% ----------------------------------------------------------------
function kernel_init(clt, B, mode)
%   Mirror RUN_FTLM_PG_SECTOR_GPU_IH's init dispatch for a direct MEX session
%   (same helper as in TEST_REAL_KERNEL).
    n_basis   = clt.n_basis;
    n_reps    = clt.n_reps;
    d_irrep   = clt.d_irrep;
    n_entries = gather(double(sum(clt.entries_per_rep)));

    if isfield(clt, 'c_idx'),   c_idx_arg   = clt.c_idx;
    else,                       c_idx_arg   = gpuArray(zeros(0, 1, 'uint8'));  end
    if isfield(clt, 'c_table'), c_table_arg = clt.c_table;
    else,                       c_table_arg = gpuArray(zeros(0, 1, 'single')); end
    if isfield(clt, 'srcg'),    srcg_arg    = clt.srcg;
    else,                       srcg_arg    = gpuArray(zeros(0, 1, 'uint32')); end
    if isfield(clt, 'triv'),    triv_arg    = clt.triv;
    else,                       triv_arg    = gpuArray(zeros(0, 1, 'uint8'));  end
    if isfield(clt, 'Qbar_re'), qre_arg = clt.Qbar_re; qim_arg = clt.Qbar_im;
    else, qre_arg = gpuArray(zeros(0,1,'single')); qim_arg = gpuArray(zeros(0,1,'single')); end
    if isfield(clt, 'v_slot'),  vslot_arg   = clt.v_slot;
    else,                       vslot_arg   = gpuArray(zeros(0, 1, 'int32'));  end

    switch mode
        case {'stream', 'prefix'}
            if strcmp(mode, 'prefix')
                % Partial prefix: budget for roughly HALF the entries -> some
                % tiles resident, the tail streamed (both launches exercised).
                if isfield(clt, 'srcg') && ~isempty(clt.srcg)
                    bpe = 4;
                else
                    bpe = 4 + 2;
                    if isfield(clt, 'g_is_u8') && clt.g_is_u8, bpe = 4 + 1; end
                end
                if isfield(clt, 'c_idx') && ~isempty(clt.c_idx), bpe = bpe + 1; end
                pref_budget = floor(n_entries / 2) * bpe;
            else
                pref_budget = 0;                       % full streaming
            end
            cuda_lanczos_clut_block_pg_Ih('init_skel_stream', ...
                clt.diag_vals, clt.rep_offsets, clt.n_per_rep, ...
                clt.entries_per_rep, clt.entry_offsets, ...
                clt.src_idx, clt.g_idx, clt.c_a, ...
                clt.V_re, clt.V_im, clt.rho_re, clt.rho_im, clt.sqrt_eig, ...
                n_basis, n_reps, n_entries, d_irrep, B, clt.c_a_const, ...
                c_idx_arg, c_table_arg, srcg_arg, triv_arg, qre_arg, qim_arg, vslot_arg, ...
                int32(clt.tile_rep_ptr), int64(clt.tile_e_start), int64(clt.tile_e_count), ...
                pref_budget);
        otherwise   % 'ref'
            cuda_lanczos_clut_block_pg_Ih('init_skel_ref', ...
                clt.diag_vals, clt.rep_offsets, clt.n_per_rep, ...
                clt.entries_per_rep, clt.entry_offsets, ...
                clt.src_idx, clt.g_idx, clt.c_a, ...
                clt.V_re, clt.V_im, clt.rho_re, clt.rho_im, clt.sqrt_eig, ...
                n_basis, n_reps, n_entries, d_irrep, B, clt.c_a_const, ...
                c_idx_arg, c_table_arg, srcg_arg, triv_arg, qre_arg, qim_arg, vslot_arg);
    end
end
