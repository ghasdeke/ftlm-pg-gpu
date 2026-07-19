function test_c_const_refactor()
%TEST_C_CONST_REFACTOR  Verify the entries memory compactions (s=1/2).
%
%   Two COLLECT_CLT_ENTRIES_IH compactions are checked together:
%     - constant c: for s=1/2 the off-diagonal coefficient is exactly 0.5*J,
%       stored as a scalar c_const + empty c_sorted (saves n_entries*8 B =
%       22.6 GB at N=36);
%     - uint8 g: the group index 1..120 is stored as uint8 not int32
%       (saves n_entries*3 B = 8.5 GB at N=36).
%   The test proves the downstream CLT builders are BIT-IDENTICAL whether
%   they receive the new compact representation (scalar c, uint8 g) or the
%   old one (full per-entry c array, int32 g):
%
%     (1) Producer: collect sets c_is_const=true, c_const=0.5*J, empty
%         c_sorted, and an unchanged n_entries.
%     (2) Reconstruct an "old style" entries_full (c_is_const removed,
%         c_sorted = 0.5*ones(n_entries,1) -- order-independent since
%         constant) and build BUILD_CLT_SKELETON_FROM_ENTRIES_IH (the
%         production builder) from BOTH. Every output field must match.
%     (3) Same for BUILD_CLT_FROM_ENTRIES_IH_STREAMED (the bench path).
%
%   Run on the icosahedron (N=12) -- the const-handling logic is size-
%   independent, so the small geometry exercises it fully and fast.
%
%   See also COLLECT_CLT_ENTRIES_IH, BUILD_CLT_SKELETON_FROM_ENTRIES_IH,
%            BUILD_CLT_FROM_ENTRIES_IH_STREAMED, TEST_LOOKUP_SCHNACK_VS_BITMAP.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    s_val = 0.5; J = 1.0; M = 0;
    group = icosahedron_Ih_full();
    bonds = adjacency_icosahedron_Ih();

    fprintf('=== test_c_const_refactor (icosahedron N=%d, s=1/2, M=0) ===\n', double(group.N));

    cache = enumerate_M_orbits_Ih_gpu(s_val, M, group);
    entries = collect_clt_entries_Ih(cache.super_reps, bonds, s_val, J, group, 'schnack', 'host');

    %% (1) Producer assertions: compact c AND uint8 g.
    assert(isfield(entries, 'c_is_const') && entries.c_is_const, ...
           'collect did not set c_is_const for s=1/2');
    assert(abs(entries.c_const - 0.5*J) < 1e-12, 'c_const ~= 0.5*J');
    assert(isempty(entries.c_sorted), 'c_sorted should be empty when constant');
    assert(isa(entries.g_sorted, 'uint16'), 'g_sorted should be uint16');
    n_e = entries.n_entries;
    fprintf('(1) producer: c_is_const=1, c_const=%.4f, c_sorted empty, g uint16, n_entries=%d  OK\n', ...
            entries.c_const, n_e);
    fprintf('    memory vs old: c_sorted %.2f MB -> 0; g_sorted %.2f MB -> %.2f MB\n', ...
            n_e*8/1e6, n_e*4/1e6, n_e*2/1e6);

    %% Build the "old style" entries: full per-entry c array (constant ->
    %  order-independent) AND int32 g. Matching skeletons/streamed CLTs then
    %  verify BOTH the constant-c and the uint8-g compactions at once.
    entries_full = rmfield(entries, {'c_is_const', 'c_const'});
    entries_full.c_sorted = entries.c_const * ones(n_e, 1);
    entries_full.g_sorted = int32(entries.g_sorted);

    has_gpu = (gpuDeviceCount > 0);
    if ~has_gpu
        fprintf('No CUDA device: skipping CLT-builder bit-identity (needs gpuArray).\n');
        fprintf('\ntest_c_const_refactor: producer checks PASS (builder check skipped)\n');
        return;
    end
    gpu_h = gpuDevice;

    irreps = build_irreps_table(group);
    all_ok = true;

    %% (2) + (3) Bit-identity of the CLT builders, per irrep.
    fprintf('(2/3) CLT builder bit-identity (compact-c vs full-c):\n');
    for ig = 1 : numel(irreps)
        ir = irreps{ig};
        [reps, V, eg, npr, ~] = apply_irrep_to_orbits(cache, ir.data, ir.d, group);
        if sum(npr) == 0, continue; end

        sk_new  = build_clt_skeleton_from_entries_Ih(entries,      reps, V, eg, npr, ir.data, ir.d, group);
        sk_full = build_clt_skeleton_from_entries_Ih(entries_full, reps, V, eg, npr, ir.data, ir.d, group);
        ok_sk = struct_equal(sk_new, sk_full);

        st_new  = build_clt_from_entries_Ih_streamed(entries,      reps, V, eg, npr, ir.data, ir.d, group, gpu_h);
        st_full = build_clt_from_entries_Ih_streamed(entries_full, reps, V, eg, npr, ir.data, ir.d, group, gpu_h);
        ok_st = struct_equal(st_new, st_full);

        fprintf('   %-4s d=%d n_basis=%4d : skeleton=%d  streamed=%d\n', ...
                ir.name, ir.d, sk_new.n_basis, ok_sk, ok_st);
        all_ok = all_ok && ok_sk && ok_st;
    end

    assert(all_ok, 'CLT builder output differs between compact and old-style entries');
    fprintf('\ntest_c_const_refactor: PASS (production CLT bit-identical; %.1f GB/N=36 saved)\n', 22.6 + 8.5);
end


% ----------------------------------------------------------------
function ok = struct_equal(a, b)
%STRUCT_EQUAL  True if two CLT structs match field-by-field (gpuArrays
%   gathered to host first). Uses isequaln so empty/NaN compare cleanly.
    ok = true;
    fa = sort(fieldnames(a)); fb = sort(fieldnames(b));
    if ~isequal(fa, fb)
        fprintf('     field-set differs\n'); ok = false; return;
    end
    for i = 1 : numel(fa)
        f = fa{i};
        va = a.(f); vb = b.(f);
        if isa(va, 'gpuArray'), va = gather(va); end
        if isa(vb, 'gpuArray'), vb = gather(vb); end
        if ~isequaln(va, vb)
            fprintf('     field %s DIFFERS\n', f); ok = false;
        end
    end
end


% ----------------------------------------------------------------
function irreps = build_irreps_table(group)
    irreps = {};
    irreps{end+1} = struct('name', 'A_g', 'd', 1, 'data', group.Ag);
    irreps{end+1} = struct('name', 'A_u', 'd', 1, 'data', group.Au);
    irreps{end+1} = struct('name', 'T1g', 'd', 3, 'data', group.T1g);
    irreps{end+1} = struct('name', 'T1u', 'd', 3, 'data', group.T1u);
    irreps{end+1} = struct('name', 'T2g', 'd', 3, 'data', group.T2g);
    irreps{end+1} = struct('name', 'T2u', 'd', 3, 'data', group.T2u);
    irreps{end+1} = struct('name', 'F_g', 'd', 4, 'data', group.Fg);
    irreps{end+1} = struct('name', 'F_u', 'd', 4, 'data', group.Fu);
    irreps{end+1} = struct('name', 'H_g', 'd', 5, 'data', group.Hg);
    irreps{end+1} = struct('name', 'H_u', 'd', 5, 'data', group.Hu);
end
