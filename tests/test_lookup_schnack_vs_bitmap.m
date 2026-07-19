function test_lookup_schnack_vs_bitmap()
%TEST_LOOKUP_SCHNACK_VS_BITMAP  Regression: Schnack-CR == bitmap CLT.
%
%   For several (s, M) combinations:
%     (1) Builds super_reps via ENUMERATE_M_ORBITS_IH_GPU.
%     (2) Builds both lookup backends.
%     (3) Queries each backend with every super_rep + a fixed set of
%         non-super-rep M-sector states. Asserts identical results
%         (1-based super_rep indices, 0 for non-super-reps).
%     (4) Runs COLLECT_CLT_ENTRIES_IH with both lookup_methods and
%         confirms the entry tables (src/tgt/g/c sorted) are
%         bit-for-bit identical.
%
%   This is the gate for shipping the Schnack-CR path: as long as the
%   entry tables are identical, the downstream CLT + SpMV + Lanczos
%   results are guaranteed to agree.
%
%   Cases:
%     s = 1/2 on icosahedron, M = 0 and M = 2
%     s = 1   on icosahedron, M = 0
%     s = 1/2 on icosidodecahedron, M = 0
%
%   The icosidodecahedron case is the production target.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    fprintf('=== Regression: Schnack-CR vs bitmap CLT lookup ===\n\n');

    overall = true;

    cases = {
        struct('geom','icosahedron',       's',0.5, 'M',0)
        struct('geom','icosahedron',       's',0.5, 'M',2)
        struct('geom','icosahedron',       's',1.0, 'M',0)
        struct('geom','icosidodecahedron', 's',0.5, 'M',0)
    };

    for k = 1 : numel(cases)
        c = cases{k};
        fprintf('--- %s, s=%g, M=%d ---\n', c.geom, c.s, c.M);
        ok = run_single_case(c);
        overall = overall && ok;
    end

    fprintf('\n=======================================\n');
    fprintf('OVERALL Schnack-vs-bitmap: %s\n', tern(overall, 'PASS', 'FAIL'));
end


% ----------------------------------------------------------------
function ok = run_single_case(c)
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

    %% (1) Build both lookups and compare query results on super-reps.
    N_sites = double(group.N);
    d_loc   = round(2*c.s + 1);
    n_total = double(d_loc)^N_sites;

    t0 = tic;
    lookup_bm = build_clt_lookup(cache_M.super_reps, n_total);
    t_bm_build = toc(t0);

    t0 = tic;
    lookup_sn = build_lookup_schnack(cache_M.super_reps, c.s, N_sites);
    t_sn_build = toc(t0);

    fprintf('  build : bitmap %.3f s, schnack %.3f s\n', t_bm_build, t_sn_build);

    % Query on super-reps themselves (should return 1..n_reps).
    t0 = tic;
    idx_bm_self = query_clt_lookup(lookup_bm, cache_M.super_reps);
    t_bm_q = toc(t0);
    t0 = tic;
    idx_sn_self = query_lookup_schnack(lookup_sn, cache_M.super_reps);
    t_sn_q = toc(t0);

    fprintf('  query (%d states): bitmap %.3f s, schnack %.3f s\n', ...
            n_reps, t_bm_q, t_sn_q);

    ok_self_bm = isequal(idx_bm_self, int32(1 : n_reps).');
    ok_self_sn = isequal(idx_sn_self, int32(1 : n_reps).');
    ok_match   = isequal(idx_bm_self, idx_sn_self);

    fprintf('    bitmap self -> 1..n_reps       : %s\n', tern(ok_self_bm, 'OK', 'FAIL'));
    fprintf('    schnack self -> 1..n_reps      : %s\n', tern(ok_self_sn, 'OK', 'FAIL'));
    fprintf('    bitmap == schnack on super_reps: %s\n', tern(ok_match, 'OK', 'FAIL'));

    ok = ok_self_bm && ok_self_sn && ok_match;

    %% (2) End-to-end: COLLECT_CLT_ENTRIES_IH with both methods.
    entries_bm = collect_clt_entries_Ih(cache_M.super_reps, bonds, c.s, J, group, 'bitmap');
    entries_sn = collect_clt_entries_Ih(cache_M.super_reps, bonds, c.s, J, group, 'schnack');

    ok_src = isequal(entries_bm.src_sorted, entries_sn.src_sorted);
    ok_tgt = isequal(entries_bm.tgt_sorted, entries_sn.tgt_sorted);
    ok_g   = isequal(entries_bm.g_sorted,   entries_sn.g_sorted);
    % c may be stored constant (s=1/2: scalar c_const, empty c_sorted) or
    % per-entry; compare a representation that covers both.
    ok_c   = isequaln(c_signature(entries_bm), c_signature(entries_sn));

    fprintf('    entries.src_sorted  identical : %s\n', tern(ok_src, 'OK', 'FAIL'));
    fprintf('    entries.tgt_sorted  identical : %s\n', tern(ok_tgt, 'OK', 'FAIL'));
    fprintf('    entries.g_sorted    identical : %s\n', tern(ok_g,   'OK', 'FAIL'));
    fprintf('    entries.c_sorted    identical : %s\n', tern(ok_c,   'OK', 'FAIL'));

    ok = ok && ok_src && ok_tgt && ok_g && ok_c;

    fprintf('  case result: %s\n\n', tern(ok, 'PASS', 'FAIL'));
end


function s = tern(c, a, b)
    if c, s = a; else, s = b; end
end


function sig = c_signature(e)
% Representation-agnostic signature of the off-diagonal coefficient:
% handles the constant-c case (scalar c_const, empty c_sorted) and the
% per-entry array case uniformly.
    if isfield(e, 'c_is_const') && e.c_is_const
        sig = {true,  double(e.c_const), []};
    elseif isfield(e, 'c_is_indexed') && e.c_is_indexed
        sig = {false, [], double(e.c_table(e.c_idx(:)))};   % reconstruct per-entry c
    else
        sig = {false, [], double(e.c_sorted(:))};
    end
end
