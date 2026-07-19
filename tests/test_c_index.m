function test_c_index()
%TEST_C_INDEX  Verify the uint8 c-index (s>1/2) against the per-entry c_a path.
%
%   On the s=3/2 icosahedron (M=0), c is NOT constant, so COLLECT stores a
%   uint8 c_idx + a tiny c_table. This test:
%     (1) checks the producer emits the indexed representation, and that
%         c_table(c_idx) reconstructs a finite per-entry coefficient;
%     (2) runs the GPU block-Lanczos on each irrep TWICE through the SAME
%         recompiled kernel -- once with the indexed skeleton (kernel reads
%         c = c_table(c_idx)) and once with a reconstructed full-c skeleton
%         (kernel reads the per-entry c_a single) -- and checks the Ritz
%         values + FTLM weights agree. They must, since c_table(c_idx) is
%         the exact single the per-entry path stores.
%
%   See also COLLECT_CLT_ENTRIES_IH, BUILD_CLT_SKELETON_FROM_ENTRIES_IH,
%            RUN_FTLM_PG_SECTOR_GPU_IH, TEST_C_CONST_REFACTOR.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    s_val = 1.5; J = 1.0; M = 0; R = 2; M_lz = 30;
    group = icosahedron_Ih_full();
    bonds = adjacency_icosahedron_Ih();
    fprintf('=== test_c_index (icosahedron N=%d, s=3/2, M=0) ===\n', double(group.N));

    cache   = enumerate_M_orbits_Ih_gpu(s_val, M, group);
    entries = collect_clt_entries_Ih(cache.super_reps, bonds, s_val, J, group, 'schnack', 'host');

    %% (1) producer
    assert(isfield(entries,'c_is_indexed') && entries.c_is_indexed, 'c_is_indexed not set');
    assert(isempty(entries.c_sorted), 'c_sorted must be empty when indexed');
    assert(~any(entries.c_idx == 0), 'c_idx has a 0 (unmapped) entry');
    c_recon = entries.c_table(entries.c_idx);
    assert(all(isfinite(c_recon)) && all(c_recon > 0), 'reconstructed c not positive-finite');
    fprintf('(1) producer: c_is_indexed=1, |c_table|=%d, n_entries=%d  OK\n', ...
            numel(entries.c_table), entries.n_entries);

    %% full-c entries (reconstruct the per-entry double from the index)
    entries_full = rmfield(entries, {'c_table', 'c_idx'});
    entries_full.c_is_indexed = false;
    entries_full.c_sorted     = double(entries.c_table(entries.c_idx));

    if gpuDeviceCount == 0 || exist('cuda_lanczos_clut_block_pg_Ih','file') ~= 3
        fprintf('No GPU/kernel: producer check only.\n\ntest_c_index: producer PASS\n');
        return;
    end
    gpu_h = gpuDevice;
    irreps = build_irreps_table(group);

    %% (2) A/B Lanczos: indexed vs per-entry c, same kernel
    fprintf('(2) Lanczos index-c vs per-entry-c (same kernel):\n');
    all_ok = true;
    for ig = 1 : numel(irreps)
        ir = irreps{ig};
        [reps, V, eg, npr, ~] = apply_irrep_to_orbits(cache, ir.data, ir.d, group);
        if sum(npr) == 0, continue; end

        sk_i = build_clt_skeleton_from_entries_Ih(entries,      reps, V, eg, npr, ir.data, ir.d, group);
        seed = 7777;
        [E_i, w_i] = run_ftlm_pg_sector_gpu_Ih(sk_i, R, M_lz, 0, seed, gpu_h);
        sk_i = []; wait(gpu_h); %#ok<NASGU>

        sk_f = build_clt_skeleton_from_entries_Ih(entries_full, reps, V, eg, npr, ir.data, ir.d, group);
        [E_f, w_f] = run_ftlm_pg_sector_gpu_Ih(sk_f, R, M_lz, 0, seed, gpu_h);
        sk_f = []; wait(gpu_h); %#ok<NASGU>

        dE = max(abs(E_i - E_f)) / max(max(abs(E_f)), 1e-30);
        dw = max(abs(w_i - w_f)) / max(max(abs(w_f)), 1e-30);
        exact = isequal(E_i, E_f) && isequal(w_i, w_f);
        ok = (dE < 1e-5) && (dw < 1e-5);
        fprintf('   %-4s n_basis=%5d : relErr E=%.1e w=%.1e  exact=%d  %s\n', ...
                ir.name, sum(npr), dE, dw, exact, tern(ok,'OK','FAIL'));
        all_ok = all_ok && ok;
    end

    assert(all_ok, 'c-index Lanczos differs from per-entry c_a');
    fprintf('\ntest_c_index: PASS\n');
end


function irreps = build_irreps_table(group)
    irreps = {};
    irreps{end+1} = struct('name','A_g','d',1,'data',group.Ag);
    irreps{end+1} = struct('name','A_u','d',1,'data',group.Au);
    irreps{end+1} = struct('name','T1g','d',3,'data',group.T1g);
    irreps{end+1} = struct('name','T1u','d',3,'data',group.T1u);
    irreps{end+1} = struct('name','T2g','d',3,'data',group.T2g);
    irreps{end+1} = struct('name','T2u','d',3,'data',group.T2u);
    irreps{end+1} = struct('name','F_g','d',4,'data',group.Fg);
    irreps{end+1} = struct('name','F_u','d',4,'data',group.Fu);
    irreps{end+1} = struct('name','H_g','d',5,'data',group.Hg);
    irreps{end+1} = struct('name','H_u','d',5,'data',group.Hu);
end

function s = tern(c, a, b)
    if c, s = a; else, s = b; end
end
