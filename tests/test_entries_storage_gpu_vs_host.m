function test_entries_storage_gpu_vs_host()
%TEST_ENTRIES_STORAGE_GPU_VS_HOST  Regression for Stufe 6b.
%
%   For several (geometry, s, M) configurations, runs the per-M
%   precompute pipeline twice:
%
%     (1) entries_storage = 'host'  (legacy default)
%     (2) entries_storage = 'gpu'   (Stufe 6b production path for N >= 32)
%
%   and verifies that the resulting CLT skeleton produced by
%   BUILD_CLT_SKELETON_FROM_ENTRIES_IH is identical (the entry
%   indexing must match exactly; floating-point values must agree
%   to machine precision in their respective types).
%
%   The decisive test: if BUILD_CLT_SKELETON_FROM_ENTRIES_IH outputs
%   identical CLT structs for both storage modes, every downstream
%   step (CUDA init_skel_ref, SpMV, Lanczos) consumes identical
%   inputs and therefore produces identical spectra.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    fprintf('=== Regression: entries_storage gpu vs host ===\n\n');

    overall = true;

    cases = {
        struct('geom','icosahedron',       's',0.5, 'M',0)
        struct('geom','icosahedron',       's',1.0, 'M',0)
        struct('geom','icosahedron',       's',2.0, 'M',0)
        struct('geom','icosidodecahedron', 's',0.5, 'M',0)
    };

    for k = 1 : numel(cases)
        c = cases{k};
        fprintf('--- %s, s=%g, M=%d ---\n', c.geom, c.s, c.M);
        ok = run_one_case(c);
        overall = overall && ok;
    end

    fprintf('\n=========================================\n');
    fprintf('OVERALL Stufe 6b entries storage: %s\n', tern(overall, 'PASS', 'FAIL'));
end


% ----------------------------------------------------------------
function ok = run_one_case(c)
    switch c.geom
        case 'icosahedron'
            group = icosahedron_Ih_full();
            bonds = adjacency_icosahedron_Ih();
        case 'icosidodecahedron'
            group = icosidodecahedron_Ih_full();
            bonds = adjacency_icosidodecahedron_Ih();
        otherwise
            error('unknown geometry');
    end
    J = 1.0;

    cache_M = enumerate_M_orbits_Ih_gpu(c.s, c.M, group);
    n_reps  = numel(cache_M.super_reps);
    fprintf('  n_super_reps = %d\n', n_reps);
    if n_reps == 0
        fprintf('  empty sector, skipping.\n');
        ok = true;
        return;
    end

    %% Build entries via both pipelines (using bitmap lookup; the
    %  lookup_method choice is orthogonal to entries_storage).
    t0 = tic;
    entries_host = collect_clt_entries_Ih(cache_M.super_reps, bonds, c.s, J, group, ...
                                          'bitmap', 'host');
    t_host = toc(t0);
    t0 = tic;
    entries_gpu  = collect_clt_entries_Ih(cache_M.super_reps, bonds, c.s, J, group, ...
                                          'bitmap', 'gpu');
    t_gpu = toc(t0);
    fprintf('  collect: host %.3f s, gpu %.3f s\n', t_host, t_gpu);

    %% (1) Compare entries arrays directly. gather() the GPU ones for
    %  comparison.
    ok_src = isequal(entries_host.src_sorted, gather(entries_gpu.src_sorted));
    ok_tgt = isequal(entries_host.tgt_sorted, gather(entries_gpu.tgt_sorted));
    ok_g   = isequal(entries_host.g_sorted,   gather(entries_gpu.g_sorted));
    ok_c   = isequal(entries_host.c_sorted,   gather(entries_gpu.c_sorted));
    fprintf('  entries.src_sorted (host vs gpu) : %s\n', tern(ok_src, 'OK', 'FAIL'));
    fprintf('  entries.tgt_sorted (host vs gpu) : %s\n', tern(ok_tgt, 'OK', 'FAIL'));
    fprintf('  entries.g_sorted   (host vs gpu) : %s\n', tern(ok_g,   'OK', 'FAIL'));
    fprintf('  entries.c_sorted   (host vs gpu) : %s\n', tern(ok_c,   'OK', 'FAIL'));

    ok = ok_src && ok_tgt && ok_g && ok_c;
    if ~ok, return; end

    %% (2) For each irrep, build the CLT skeleton both ways and verify
    %  that the resulting struct is identical.
    irreps = build_irreps_table(group);
    for ig = 1 : numel(irreps)
        ir = irreps{ig};
        [reps, V_per_rep, eig_per_rep, n_per_rep, ~] = ...
            apply_irrep_to_orbits(cache_M, ir.data, ir.d, group);
        if isempty(reps), continue; end

        skel_h = build_clt_skeleton_from_entries_Ih(entries_host, reps, ...
            V_per_rep, eig_per_rep, n_per_rep, ir.data, ir.d, group);
        skel_g = build_clt_skeleton_from_entries_Ih(entries_gpu, reps, ...
            V_per_rep, eig_per_rep, n_per_rep, ir.data, ir.d, group);

        % Both skel.src_idx, etc. are gpuArrays (Stufe 5a output);
        % gather() and compare.
        ok_sk_src = isequal(gather(skel_h.src_idx),  gather(skel_g.src_idx));
        ok_sk_g   = isequal(gather(skel_h.g_idx),    gather(skel_g.g_idx));
        ok_sk_ca  = isequal(gather(skel_h.c_a),      gather(skel_g.c_a));
        ok_sk_n   = (skel_h.n_basis == skel_g.n_basis) && ...
                    (skel_h.n_reps  == skel_g.n_reps);
        ok_sk_diag = max(abs(gather(skel_h.diag_vals) - gather(skel_g.diag_vals))) < 1e-12;
        % V_im is EMPTY on the real FP32 path (real irrep data, e.g. the d=1
        % A_g/A_u characters) -- guard the max() so empty==empty passes.
        ok_sk_V   = max([0; abs(gather(skel_h.V_re(:)) - gather(skel_g.V_re(:)))]) < 1e-6 && ...
                    isequal(size(skel_h.V_im), size(skel_g.V_im)) && ...
                    max([0; abs(gather(skel_h.V_im(:)) - gather(skel_g.V_im(:)))]) < 1e-6;

        ok_ir = ok_sk_src && ok_sk_g && ok_sk_ca && ok_sk_n && ok_sk_diag && ok_sk_V;
        fprintf('    %-4s (d=%d): skel host vs gpu : %s\n', ...
                ir.name, ir.d, tern(ok_ir, 'OK', 'FAIL'));
        ok = ok && ok_ir;
    end
end


% ----------------------------------------------------------------
function irreps = build_irreps_table(group)
    irreps = {};
    irreps{end+1} = struct('name', 'A_g', 'd', 1, 'data', group.Ag);
    irreps{end+1} = struct('name', 'A_u', 'd', 1, 'data', group.Au);
    irreps{end+1} = struct('name', 'T1g', 'd', 3, 'data', group.T1g);
    irreps{end+1} = struct('name', 'F_g', 'd', 4, 'data', group.Fg);
    irreps{end+1} = struct('name', 'H_g', 'd', 5, 'data', group.Hg);
end

function s = tern(c, a, b)
    if c, s = a; else, s = b; end
end
