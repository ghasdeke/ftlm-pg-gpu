function test_collect_gpu_native_vs_host()
%TEST_COLLECT_GPU_NATIVE_VS_HOST  Regression for Phase B.2.1.
%
%   Compares the GPU-native collect (COLLECT_CLT_ENTRIES_IH_GPU) against
%   the host reference (COLLECT_CLT_ENTRIES_IH with lookup_method =
%   'schnack', entries_storage = 'host') and verifies bit-equivalent
%   output for several (geometry, s, M) configurations.
%
%   Cases:
%     icosahedron s=1/2 M=0    -- smallest valid case
%     icosahedron s=2  M=0     -- higher spin
%     icosidodecahedron s=1/2 M=0  -- production target
%
%   For each case the test confirms:
%     entries.src_sorted, tgt_sorted, g_sorted   identical (int32)
%     entries.c_sorted                            identical (double, exact)
%     entries.diag_vals                           identical to <1e-12
%     entries.n_entries                           identical
%
%   Wall time per case is reported so the GPU-native overhead is visible.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    fprintf('=== Regression: collect_clt_entries_Ih_gpu vs host ===\n\n');

    cases = {
        struct('geom','icosahedron',       's',0.5, 'M',0)
        struct('geom','icosahedron',       's',2.0, 'M',0)
        struct('geom','icosidodecahedron', 's',0.5, 'M',0)
    };

    overall = true;
    for k = 1 : numel(cases)
        c = cases{k};
        fprintf('--- %s, s=%g, M=%d ---\n', c.geom, c.s, c.M);
        ok = run_one(c);
        overall = overall && ok;
    end

    fprintf('\n=========================================\n');
    fprintf('OVERALL Phase B.2.1: %s\n', tern(overall, 'PASS', 'FAIL'));
end


% ----------------------------------------------------------------
function ok = run_one(c)
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
        fprintf('  empty sector, skip\n');
        ok = true; return;
    end

    t0 = tic;
    entries_host = collect_clt_entries_Ih(cache_M.super_reps, bonds, c.s, J, group, ...
                                          'schnack', 'host');
    t_host = toc(t0);

    t0 = tic;
    entries_gpu  = collect_clt_entries_Ih_gpu(cache_M.super_reps, bonds, c.s, J, group);
    t_gpu = toc(t0);

    fprintf('  collect: host %.3f s, gpu_native %.3f s\n', t_host, t_gpu);

    %% Compare entries (gather GPU sides).
    src_h = entries_host.src_sorted;        src_g = gather(entries_gpu.src_sorted);
    tgt_h = entries_host.tgt_sorted;        tgt_g = gather(entries_gpu.tgt_sorted);
    g_h   = entries_host.g_sorted;          g_g   = gather(entries_gpu.g_sorted);
    c_h   = entries_host.c_sorted;          c_g   = gather(entries_gpu.c_sorted);

    ok_n   = entries_host.n_entries == entries_gpu.n_entries;
    ok_src = isequal(src_h, src_g);
    ok_tgt = isequal(tgt_h, tgt_g);
    ok_g   = isequal(g_h, g_g);
    ok_c   = isequal(c_h, c_g);
    ok_d   = max(abs(entries_host.diag_vals - entries_gpu.diag_vals)) < 1e-12;

    fprintf('    n_entries identical (%d)   : %s\n', entries_host.n_entries, tern(ok_n,'OK','FAIL'));
    fprintf('    src_sorted identical       : %s\n', tern(ok_src,'OK','FAIL'));
    fprintf('    tgt_sorted identical       : %s\n', tern(ok_tgt,'OK','FAIL'));
    fprintf('    g_sorted identical         : %s\n', tern(ok_g,  'OK','FAIL'));
    fprintf('    c_sorted identical         : %s\n', tern(ok_c,  'OK','FAIL'));
    fprintf('    diag_vals ok to 1e-12      : %s\n', tern(ok_d,  'OK','FAIL'));

    ok = ok_n && ok_src && ok_tgt && ok_g && ok_c && ok_d;
    fprintf('  case result: %s\n\n', tern(ok,'PASS','FAIL'));
end


function s = tern(c, a, b)
    if c, s = a; else, s = b; end
end
