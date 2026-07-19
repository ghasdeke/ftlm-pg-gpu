function test_real_kernel()
%TEST_REAL_KERNEL  Real FP32 kernel path == complex kernel at V_im = 0, EXACTLY.
%
%   THE central gate of the real-arithmetic fork (June 2026): with realified
%   (REAL) irreps the complex kernel's imaginary terms are all exact IEEE
%   zeros -- products with 0.0f are +-0.0f and adding/subtracting them never
%   changes a value -- so the real kernel MUST reproduce
%       W_re  ('spmv' mode)          and
%       AL/BE ('block_lanczos' mode, same V0_re + V0_im = 0)
%   with max|diff| == 0. Only after this gate is green does production switch
%   to halved RNG draws (V0_re only -- statistically equivalent new samples).
%
%   Coverage (each case: complex clt via force_complex vs real clt, same
%   entries/eskel/V0):
%     A  4x4 C_4v d=4 realified  -- unpacked src+g, compact-V, Qbar fast path;
%        through ALL four data paths: resident (init_skel_ref), B2
%        (init_skel_b2), streaming (init_skel_stream, multi-tile), and
%        resident-prefix (partial prefix + streamed tail).
%     B  icosahedron s=1/2 T1g realified -- PACKED srcg, I_h named irrep
%        realified via realify_irreps (the Section-5 path).
%     C  icosahedron s=1 T1g realified -- indexed c (c_idx/c_table), resident
%        AND streamed (c_idx tile rides along).
%     D  kagome N=12 spin-flip Gamma- (realified, doubled 2|G| rho tables) --
%        flip preserves realness (+-rho).
%
%   See also BUILD_CLT_SKELETON_FROM_ENTRIES_IH,
%            RUN_FTLM_PG_SECTOR_GPU_IH, TEST_STREAM, TEST_B2.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    gpuDevice;           % fail early without a GPU
    assert_kernel_abi();  % direct-MEX caller: bypasses the driver's handshake
    ok = true;

    %% Case A: 4x4 C_4v d=4 realified (unpacked src+g) -- all four data paths.
    group = square_lattice_spacegroup(4, 4);
    group.irreps = realify_irreps(irreps_from_group(group), group);
    bonds = adjacency_square_lattice(4, 4);
    cache = enumerate_M_orbits_Ih_gpu(0.5, 0, group);
    entries = collect_clt_entries_Ih(cache.super_reps, bonds, 0.5, 1, group, 'bitmap', 'host');
    p4 = find(arrayfun(@(z) z.d, group.irreps) == 4, 1);
    ir4 = group.irreps(p4).mats;
    assert(isreal(ir4), 'test_real_kernel: realified C_4v d=4 irrep is not real');
    % B sweep (K6d, 2026-07): wave B runs blocks up to B=4 (+ MAX_B=8 headroom)
    % but every bit gate here ran B<=2 only -- sweep Case A's four data paths
    % over B. The complex reference is re-run at the SAME B inside check_case,
    % so the gate stays exact; the tiny fixture keeps the extra columns <1 MB.
    for Bx = [1 2 4 8]
        ok = check_case(sprintf('4x4-c4v d=4 resident B=%d', Bx), entries, cache, ir4, 4, group, 'ref',    Bx) && ok;
        ok = check_case(sprintf('4x4-c4v d=4 B2       B=%d', Bx), entries, cache, ir4, 4, group, 'b2',     Bx) && ok;
        ok = check_case(sprintf('4x4-c4v d=4 stream   B=%d', Bx), entries, cache, ir4, 4, group, 'stream', Bx) && ok;
        ok = check_case(sprintf('4x4-c4v d=4 prefix   B=%d', Bx), entries, cache, ir4, 4, group, 'prefix', Bx) && ok;
    end

    %% Case B: icosahedron s=1/2 T1g realified (PACKED srcg; Section-5 I_h path).
    gI = icosahedron_Ih_full();  bI = adjacency_icosahedron_Ih();
    irrI = struct('name', 'T1g', 'd', 3, 'mats', gI.T1g);
    irrI = realify_irreps(irrI, gI);
    assert(isreal(irrI.mats), 'test_real_kernel: realified I_h T1g is not real');
    cacheB = enumerate_M_orbits_Ih_gpu(0.5, 0, gI);
    entriesB = collect_clt_entries_Ih(cacheB.super_reps, bI, 0.5, 1, gI, 'bitmap', 'host');
    ok = check_case('icosa s=1/2 T1g real', entriesB, cacheB, irrI.mats, 3, gI, 'ref') && ok;

    %% Case C: icosahedron s=1 T1g realified (indexed c), resident + streamed.
    cacheC = enumerate_M_orbits_Ih_gpu(1.0, 0, gI);
    entriesC = collect_clt_entries_Ih(cacheC.super_reps, bI, 1.0, 1.0, gI, 'bitmap', 'host');
    assert(isfield(entriesC, 'c_is_indexed') && entriesC.c_is_indexed, ...
        'test_real_kernel: expected indexed c for s=1');
    ok = check_case('icosa s=1 T1g c_idx',  entriesC, cacheC, irrI.mats, 3, gI, 'ref')    && ok;
    ok = check_case('icosa s=1 T1g stream', entriesC, cacheC, irrI.mats, 3, gI, 'stream') && ok;

    %% Case D: kagome N=12 spin-flip Gamma- (doubled 2|G| tables, realified).
    [gk, bk] = kagome_spacegroup(2, 0);
    irrk = realify_irreps(irreps_from_group(gk), gk);
    pdk  = find(arrayfun(@(z) z.d, irrk) >= 2, 1);
    dk   = irrk(pdk).d;
    datk = cat(3, irrk(pdk).mats, -irrk(pdk).mats);    % Gamma- branch: -rho on flips
    assert(isreal(datk), 'test_real_kernel: flip-doubled kagome irrep is not real');
    gks = add_spin_flip_z2(gk);
    cacheD = enumerate_M_orbits_Ih_gpu(0.5, 0, gks);
    entriesD = collect_clt_entries_Ih(cacheD.super_reps, bk, 0.5, 1, gks, 'bitmap', 'host');
    ok = check_case(sprintf('kagome12 flip G-(d=%d)', dk), entriesD, cacheD, datk, dk, gks, 'ref') && ok;

    assert(ok, 'real kernel path differs from complex(V_im=0) somewhere above');
    fprintf('\nPASS: real FP32 kernel == complex kernel at V_im=0 (W_re, AL, BE all exact).\n');
end


% ----------------------------------------------------------------
function ok = check_case(label, entries, cache, ir, d, group, mode, B)
%   Build the SAME block twice -- force_complex (zero-imag arrays, complex
%   kernel) vs auto-real (empty imag arrays, real kernel) -- and compare
%   'spmv' W_re and 'block_lanczos' AL/BE on identical V0_re (V0_im = 0).
%   B is optional (default 2); Case A sweeps it up to MAX_B = 8.
    if nargin < 8, B = 2; end
    Mlz = 25;
    [reps, V, eg, npr, ~, triv] = apply_irrep_to_orbits(cache, ir, d, group);

    if strcmp(mode, 'ref')
        eskel = build_entry_skeleton_Ih(entries);
    else
        n_coll = numel(entries.src_sorted);
        eskel  = build_entry_skeleton_Ih(entries, true, ceil(n_coll / 5));  % force_b2, ~5 tiles
    end
    clt_c = build_clt_skeleton_from_entries_Ih(entries, reps, V, eg, npr, ir, d, group, triv, eskel, true, true);
    clt_r = build_clt_skeleton_from_entries_Ih(entries, reps, V, eg, npr, ir, d, group, triv, eskel, true, false);
    assert(~clt_c.is_real && clt_r.is_real, ...
        'check_case(%s): is_real flags wrong (c=%d r=%d)', label, clt_c.is_real, clt_r.is_real);

    n_basis = clt_c.n_basis;
    gpurng(424242);
    V0 = gpuArray.randn(n_basis, B, 'single');
    Vz = gpuArray.zeros(n_basis, B, 'single');     % complex run: explicit V_im = 0
    Ve = gpuArray(single([]));                     % real run: empty V_im

    kernel_init(clt_c, B, mode);
    [Wc_re, Wc_im] = cuda_lanczos_clut_block_pg_Ih('spmv', V0, Vz);
    [ALc, BEc] = cuda_lanczos_clut_block_pg_Ih('block_lanczos', V0, Vz, Mlz);
    cuda_lanczos_clut_block_pg_Ih('cleanup');

    kernel_init(clt_r, B, mode);
    [Wr_re, Wr_im] = cuda_lanczos_clut_block_pg_Ih('spmv', V0, Ve);
    [ALr, BEr] = cuda_lanczos_clut_block_pg_Ih('block_lanczos', V0, Ve, Mlz);
    cuda_lanczos_clut_block_pg_Ih('cleanup');

    dW  = max(abs(Wc_re(:) - Wr_re(:)));           % the Section-2 SpMV gate
    dWi = max(abs(Wc_im(:)));                      % complex W_im must be exact zeros
    dWri= max(abs(Wr_im(:)));                      % real-path W_im output is zeros
    dA  = max(abs(ALc(:) - ALr(:)));               % the Lanczos-chain gate
    dB  = max(abs(BEc(:) - BEr(:)));
    fprintf('%-26s %-7s packed=%d : |dW|=%.3e |Wim|=%.3e |dAL|=%.3e |dBE|=%.3e\n', ...
            label, mode, eskel.is_packed, dW, max(dWi, dWri), dA, dB);
    ok = (dW == 0) && (dWi == 0) && (dWri == 0) && ...
         isequal(size(ALc), size(ALr)) && (dA == 0) && (dB == 0);
end


% ----------------------------------------------------------------
function kernel_init(clt, B, mode)
%   Mirror RUN_FTLM_PG_SECTOR_GPU_IH's init dispatch for a direct MEX session.
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
        case 'b2'
            cuda_lanczos_clut_block_pg_Ih('init_skel_b2', ...
                clt.diag_vals, clt.rep_offsets, clt.n_per_rep, ...
                clt.entries_per_rep, clt.entry_offsets, ...
                clt.src_idx, clt.g_idx, clt.c_a, ...
                clt.V_re, clt.V_im, clt.rho_re, clt.rho_im, clt.sqrt_eig, ...
                n_basis, n_reps, n_entries, d_irrep, B, clt.c_a_const, ...
                c_idx_arg, c_table_arg, srcg_arg, triv_arg, qre_arg, qim_arg, vslot_arg);
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
