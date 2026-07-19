function test_tiled_spmv()
%TEST_TILED_SPMV  R2-v1: getilter SpMV == Standard-SpMV bis auf FP32-Reorder.
%
%   Der getilte Kernel (BUILD_TILED_ENTRIES + otf_real_tiled_*) berechnet
%   dieselbe Matrix-Vektor-Aktion mit identischer Entry-Arithmetik, aber in
%   (src_tile, tgt)-Summationsreihenfolge statt Bond-Ordnung. Erwartung:
%     - W stimmt elementweise bis auf FP32-Umordnungsrauschen (~1e-6 rel),
%     - block_lanczos liefert endliche AL/BE und (im Treiberkontext) eine
%       exakte Sum Rule.
%   Abgedeckt: UNPACKED src+g (4x4 C_4v d=4, mit Qbar-Trivialpfad) und
%   PACKED srcg + indexiertes c (Ikosaeder s=1 T1g d=3), jeweils Multi-Tile
%   (kleines Fensterbudget) und B=2.
%
%   Siehe auch BUILD_TILED_ENTRIES, TEST_SPLIT_V0, TEST_REAL_KERNEL.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    gpu_h = gpuDevice;   %#ok<NASGU>
    assert_kernel_abi();

    %% A: 4x4 C_4v d=4 (unpacked, realified, Qbar-Trivialpfad aktiv).
    group = square_lattice_spacegroup(4, 4);
    irr_r = realify_irreps(irreps_from_group(group), group);
    bonds = adjacency_square_lattice(4, 4);
    cache = enumerate_M_orbits_Ih_gpu(0.5, 0, group);
    entries = collect_clt_entries_Ih(cache.super_reps, bonds, 0.5, 1, group, 'bitmap', 'host');
    eskel = build_entry_skeleton_Ih(entries);
    pA = find(arrayfun(@(z) z.d, irr_r) == 4, 1);
    run_case('4x4 C4v d=4 (unpacked)', entries, cache, irr_r(pA).mats, 4, group, eskel);

    %% B: Ikosaeder s=1 T1g d=3 (packed srcg + c_idx).
    g2 = icosahedron_Ih_full(); b2 = adjacency_icosahedron_Ih();
    cache2 = enumerate_M_orbits_Ih_gpu(1.0, 0, g2);
    entries2 = collect_clt_entries_Ih(cache2.super_reps, b2, 1.0, 1.0, g2, 'bitmap', 'host');
    assert(entries2.c_is_indexed, 'erwartete indexiertes c fuer s=1');
    eskel2 = build_entry_skeleton_Ih(entries2);
    irrI = struct('name', 'T1g', 'd', 3, 'mats', g2.T1g);
    irrI = realify_irreps(irrI, g2);
    assert(isreal(irrI.mats), 'realified I_h T1g ist nicht reell');
    run_case('icosa s=1 T1g d=3 (packed+c_idx)', entries2, cache2, ...
             irrI.mats, 3, g2, eskel2);

    fprintf('\nPASS: getilter SpMV == Standard (FP32-Reorder-Toleranz), beide Layouts.\n');
end

% ----------------------------------------------------------------
function run_case(label, entries, cache, ir, d, group, eskel)
    % audit K6d: beide Batchbreiten gaten (2 = alt, 4 = Kampagne)
    for B = [2 4]
    [reps, V, eg, npr, ~, triv] = apply_irrep_to_orbits(cache, ir, d, group);
    clt = build_clt_skeleton_from_entries_Ih(entries, reps, V, eg, npr, ...
              ir, d, group, triv, eskel, true);
    nb = clt.n_basis;
    e  = gpuArray(single([]));

    gpurng(77);
    V0 = gpuArray.randn(nb, B, 'single');

    % --- Referenz: Standard-Kernel, ein SpMV.
    init_ref(clt, d, B, [], [], []);
    W_std = cuda_lanczos_clut_block_pg_Ih('spmv', V0, e);
    cuda_lanczos_clut_block_pg_Ih('cleanup');

    % --- Getilt: Entries permutieren, Run-CSR uebergeben (Multi-Tile via
    %     kleines Fensterbudget: ~1/5 der Basis).
    ro = [int64(0); cumsum(int64(gather(clt.n_per_rep)))];
    budget = max(1024, ceil(double(nb) * 4 * B / 5));
    % src/tgt in der NUMMERIERUNG DES KERNELS: aus dem CLT (per-Irrep
    % gefiltert + auf aktive Reps umnummeriert), NICHT aus den rohen Entries.
    epr = double(gather(clt.entries_per_rep));
    ct.tgt = repelem((1:numel(epr))', epr);
    if isfield(clt, 'srcg') && ~isempty(clt.srcg)
        ct.src = double(bitand(gather(clt.srcg), uint32(2^25 - 1)));
    else
        ct.src = double(gather(clt.src_idx));
    end
    assert(numel(ct.src) == numel(ct.tgt), 'CLT src/tgt inkonsistent');
    tl = build_tiled_entries(ct, ro, B, budget);
    fprintf('  [dbg] %s: nb=%d budget=%d n_tiles=%d n_runs=%d\n', ...
            label, nb, budget, tl.n_tiles, tl.n_runs);
    assert(tl.n_tiles >= 2, '%s: erwartet Multi-Tile (Budget zu gross?)', label);
    cltT = permute_clt_entries(clt, tl.perm);
    init_ref(cltT, d, B, gpuArray(int64(tl.run_ptr)), ...
             gpuArray(int32(tl.run_tgt)), tl.tile_run_ptr);
    W_tld = cuda_lanczos_clut_block_pg_Ih('spmv', V0, e);
    % AL/BE-Rauchtest im selben Init (frisches V0, gleiche Statistik-Klasse).
    gpurng(78);
    V1 = gpuArray.randn(nb, B, 'single');
    [AL, BE] = cuda_lanczos_clut_block_pg_Ih('block_lanczos', V1, e, 20);
    assert(all(isfinite(AL(:))) && all(isfinite(BE(:))), ...
        '%s: nicht-endliche AL/BE im getilten Lanczos', label);
    cuda_lanczos_clut_block_pg_Ih('cleanup');

    dW  = max(abs(gather(W_std(:)) - gather(W_tld(:))));
    ref = max(abs(gather(W_std(:))));
    rel = double(dW) / max(double(ref), eps);
    fprintf('  %-32s n_tiles=%d n_runs=%d  max|dW|=%.3e (rel %.3e)\n', ...
            label, tl.n_tiles, tl.n_runs, dW, rel);
    assert(rel < 1e-5, '%s: getilter SpMV weicht zu stark ab (rel %.3e)', label, rel);
end

function cltT = permute_clt_entries(clt, perm)
    cltT = clt;
    if isfield(clt, 'srcg') && ~isempty(clt.srcg)
        cltT.srcg = clt.srcg(perm);
    end
    if ~isempty(clt.src_idx), cltT.src_idx = clt.src_idx(perm); end
    if isfield(clt, 'g_idx') && ~isempty(clt.g_idx), cltT.g_idx = clt.g_idx(perm); end
    if isfield(clt, 'c_a') && numel(clt.c_a) == numel(perm)
        cltT.c_a = clt.c_a(perm);
    end
    if isfield(clt, 'c_idx') && numel(clt.c_idx) == numel(perm)
        cltT.c_idx = clt.c_idx(perm);
    end
end

function init_ref(clt, d, B, run_ptr, run_tgt, tile_run_ptr)
    cuda_lanczos_clut_block_pg_Ih('init_skel_ref', ...
        clt.diag_vals, clt.rep_offsets, clt.n_per_rep, ...
        clt.entries_per_rep, clt.entry_offsets, clt.src_idx, clt.g_idx, clt.c_a, ...
        clt.V_re, clt.V_im, clt.rho_re, clt.rho_im, clt.sqrt_eig, ...
        clt.n_basis, clt.n_reps, double(sum(clt.entries_per_rep)), d, B, clt.c_a_const, ...
        clt.c_idx, clt.c_table, clt.srcg, clt.triv, clt.Qbar_re, clt.Qbar_im, clt.v_slot, ...
        run_ptr, run_tgt, tile_run_ptr);
    end
end
