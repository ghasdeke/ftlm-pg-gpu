function test_split_v0()
%TEST_SPLIT_V0  Chunked V0 upload ('set_v0') == passing V0 directly, EXACTLY.
%
%   The split-V0 path exists for blocks whose n_basis exceeds MATLAB's
%   2^31-1 elements-per-gpuArray cap (dodecahedron s=3/2 M=1..5 d>=4,
%   icosahedron s=5): run_ftlm draws the start vector in chunks and hands
%   them to the kernel's own 64-bit buffer via 'set_v0'; 'block_lanczos'
%   then runs with an EMPTY V0. Since the kernel sees the SAME V0 values,
%   AL/BE must be BIT-IDENTICAL to the direct call. Covered here:
%     A: REAL path (realified irreps -- the production dodec/icosa I_h
%        case), three uneven chunks, to_im = 0.
%     B: COMPLEX path (unrealified C_4v irrep), V0_re AND V0_im chunked
%        (to_im = 0 / 1).
%   Plus guards: out-of-range chunk, B_batch > 1, and the complex-path
%   empty-V0_im misuse (reads NULL without the kernel guard -- that exact
%   bug shipped in this test's first version: it crashed on the RTX 4000
%   and read stable garbage on the B200).
%
%   See also RUN_FTLM_PG_SECTOR_GPU_IH, TEST_B2, TEST_GPU_SIZING_INVARIANTS.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    gpu_h = gpuDevice;   %#ok<NASGU>
    assert_kernel_abi();

    %% Shared small system (same construction as TEST_B2).
    group = square_lattice_spacegroup(4, 4);
    irr_c = irreps_from_group(group);                  % complex (unrealified)
    irr_r = realify_irreps(irr_c, group);              % real (production path)
    bonds = adjacency_square_lattice(4, 4);
    cache = enumerate_M_orbits_Ih_gpu(0.5, 0, group);
    entries = collect_clt_entries_Ih(cache.super_reps, bonds, 0.5, 1, group, 'bitmap', 'host');
    eskel = build_entry_skeleton_Ih(entries);
    d = 4;

    %% A: REAL path -- direct V0 vs three uneven set_v0 chunks.
    clt_r = make_clt(cache, entries, eskel, irr_r, d, group);
    assert(isempty(clt_r.V_im), 'expected a REAL clt after realify_irreps');
    nb = clt_r.n_basis;
    e  = gpuArray(single([]));
    gpurng(11);
    V0 = gpuArray.randn(nb, 1, 'single');

    init_ref(clt_r, d, 1);
    [ALa, BEa] = cuda_lanczos_clut_block_pg_Ih('block_lanczos', V0, e, 40);
    cuda_lanczos_clut_block_pg_Ih('cleanup');

    init_ref(clt_r, d, 1);
    cuts = unique([0, floor(nb * 0.37), floor(nb * 0.71), nb]);
    for k = 1 : numel(cuts) - 1
        cuda_lanczos_clut_block_pg_Ih('set_v0', V0(cuts(k) + 1 : cuts(k + 1)), cuts(k), 0);
    end
    [ALb, BEb] = cuda_lanczos_clut_block_pg_Ih('block_lanczos', e, e, 40);
    assert(isequal(ALa, ALb) && isequal(BEa, BEb), ...
        'REAL split-V0 (set_v0 chunks) differs from the direct V0 call');
    fprintf('  A: real   split-V0 == direct V0 (AL/BE bit-identical, %d steps)\n', size(ALa, 1));

    %% B: COMPLEX path -- V0_re AND V0_im via to_im = 0/1 chunks.
    clt_c = make_clt(cache, entries, eskel, irr_c, d, group);
    assert(~isempty(clt_c.V_im), 'expected a COMPLEX clt from unrealified irreps');
    nbc = clt_c.n_basis;
    gpurng(12);
    W0r = gpuArray.randn(nbc, 1, 'single');
    W0i = gpuArray.randn(nbc, 1, 'single');

    init_ref(clt_c, d, 1);
    [ALc, BEc] = cuda_lanczos_clut_block_pg_Ih('block_lanczos', W0r, W0i, 40);
    cuda_lanczos_clut_block_pg_Ih('cleanup');

    init_ref(clt_c, d, 1);
    cuts = unique([0, floor(nbc * 0.53), nbc]);
    for k = 1 : numel(cuts) - 1
        cuda_lanczos_clut_block_pg_Ih('set_v0', W0r(cuts(k) + 1 : cuts(k + 1)), cuts(k), 0);
        cuda_lanczos_clut_block_pg_Ih('set_v0', W0i(cuts(k) + 1 : cuts(k + 1)), cuts(k), 1);
    end
    [ALd, BEd] = cuda_lanczos_clut_block_pg_Ih('block_lanczos', e, e, 40);
    assert(isequal(ALc, ALd) && isequal(BEc, BEd), ...
        'COMPLEX split-V0 (to_im chunks) differs from the direct V0 call');
    fprintf('  B: complex split-V0 == direct V0 (AL/BE bit-identical, %d steps)\n', size(ALc, 1));

    % Complex misuse guard: non-empty V0_re with EMPTY V0_im must error
    % loudly (kernel guard), never read a NULL V0_im.
    cuda_lanczos_clut_block_pg_Ih('cleanup');
    init_ref(clt_c, d, 1);
    threw = false;
    try, cuda_lanczos_clut_block_pg_Ih('block_lanczos', W0r, e, 4); catch, threw = true; end
    assert(threw, 'complex block_lanczos with empty V0_im did not error');
    cuda_lanczos_clut_block_pg_Ih('cleanup');

    %% C: REAL path, B = 2 (segmented-V0 B-unlock, 2026-07-10):
    %  per-COLUMN set_v0 chunks (5th arg = column) scattered into the
    %  interleaved buffer must reproduce the direct one-shot V0 EXACTLY.
    %  Column slices of ONE draw are used, so this gates the SCATTER, not
    %  any RNG-chunking convention.
    gpurng(13);
    V2 = gpuArray.randn(nb, 2, 'single');
    init_ref(clt_r, d, 2);
    [ALe, BEe] = cuda_lanczos_clut_block_pg_Ih('block_lanczos', V2, e, 40);
    cuda_lanczos_clut_block_pg_Ih('cleanup');

    init_ref(clt_r, d, 2);
    cuts = unique([0, floor(nb * 0.29), floor(nb * 0.83), nb]);
    for b = 0 : 1
        for k = 1 : numel(cuts) - 1
            cuda_lanczos_clut_block_pg_Ih('set_v0', ...
                V2(cuts(k) + 1 : cuts(k + 1), b + 1), cuts(k), 0, b);
        end
    end
    [ALf, BEf] = cuda_lanczos_clut_block_pg_Ih('block_lanczos', e, e, 40);
    assert(isequal(ALe, ALf) && isequal(BEe, BEf), ...
        'REAL segmented-V0 (B=2 per-column scatter) differs from the one-shot V0');
    cuda_lanczos_clut_block_pg_Ih('cleanup');
    fprintf('  C: real   segmented-V0 B=2 == one-shot V0 (AL/BE bit-identical)\n');

    %% C4: dito bei B = 4 (Kampagnen-Batchbreite, audit K6d).
    gpurng(17);
    V4 = gpuArray.randn(nb, 4, 'single');
    init_ref(clt_r, d, 4);
    [AL4, BE4] = cuda_lanczos_clut_block_pg_Ih('block_lanczos', V4, e, 40);
    cuda_lanczos_clut_block_pg_Ih('cleanup');
    init_ref(clt_r, d, 4);
    for b = 0 : 3
        for k = 1 : numel(cuts) - 1
            cuda_lanczos_clut_block_pg_Ih('set_v0', ...
                V4(cuts(k) + 1 : cuts(k + 1), b + 1), cuts(k), 0, b);
        end
    end
    [AL4f, BE4f] = cuda_lanczos_clut_block_pg_Ih('block_lanczos', e, e, 40);
    assert(isequal(AL4, AL4f) && isequal(BE4, BE4f), ...
        'REAL segmented-V0 (B=4 per-column scatter) differs from the one-shot V0');
    cuda_lanczos_clut_block_pg_Ih('cleanup');
    fprintf('  C4: real  segmented-V0 B=4 == one-shot V0 (AL/BE bit-identical)\n');

    %% Guards: out-of-range chunk / column out of range / complex B_batch > 1.
    init_ref(clt_r, d, 1);
    threw = false;
    try, cuda_lanczos_clut_block_pg_Ih('set_v0', V0(1:8), nb - 4, 0); catch, threw = true; end
    assert(threw, 'out-of-range set_v0 chunk did not error');
    cuda_lanczos_clut_block_pg_Ih('cleanup');
    init_ref(clt_r, d, 2);
    threw = false;
    try, cuda_lanczos_clut_block_pg_Ih('set_v0', V0(1:8), 0, 0, 2); catch, threw = true; end
    assert(threw, 'set_v0 with column >= B_batch did not error');
    cuda_lanczos_clut_block_pg_Ih('cleanup');
    init_ref(clt_c, d, 2);
    threw = false;
    try, cuda_lanczos_clut_block_pg_Ih('set_v0', W0r(1:8), 0, 0); catch, threw = true; end
    assert(threw, 'COMPLEX set_v0 with B_batch=2 did not error');
    cuda_lanczos_clut_block_pg_Ih('cleanup');

    fprintf('  PASS: split-V0 upload path exact (real + complex + B=2 segmented) + guarded.\n');
end


% ----------------------------------------------------------------
function clt = make_clt(cache, entries, eskel, irreps, d, group)
    p = find(arrayfun(@(z) z.d, irreps) == d, 1);
    assert(~isempty(p), 'no d=%d irrep found', d);
    ir = irreps(p).mats;
    [reps, V, eg, npr, ~, triv] = apply_irrep_to_orbits(cache, ir, d, group);
    clt = build_clt_skeleton_from_entries_Ih(entries, reps, V, eg, npr, ...
              ir, d, group, triv, eskel, true);
end

function init_ref(clt, d, B)
    cuda_lanczos_clut_block_pg_Ih('init_skel_ref', ...
        clt.diag_vals, clt.rep_offsets, clt.n_per_rep, ...
        clt.entries_per_rep, clt.entry_offsets, clt.src_idx, clt.g_idx, clt.c_a, ...
        clt.V_re, clt.V_im, clt.rho_re, clt.rho_im, clt.sqrt_eig, ...
        clt.n_basis, clt.n_reps, double(sum(clt.entries_per_rep)), d, B, clt.c_a_const, ...
        clt.c_idx, clt.c_table, clt.srcg, clt.triv, clt.Qbar_re, clt.Qbar_im, clt.v_slot);
end
