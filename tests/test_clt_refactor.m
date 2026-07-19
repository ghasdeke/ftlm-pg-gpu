function test_clt_refactor()
%TEST_CLT_REFACTOR  Regression: collect + build_from_entries == build_clt_pg_Ih.
%
%   For a handful of (s, M, Gamma) cases this test builds the CLT both
%       (a) via the monolithic BUILD_CLT_PG_IH (reference)
%       (b) via the two-phase COLLECT_CLT_ENTRIES_IH + BUILD_CLT_FROM_ENTRIES_IH
%   and checks that every CLT field matches. Index fields must be
%   identical; M_tens is compared in Frobenius norm with a tight FP-double
%   tolerance.
%
%   Also reports wall times so the Phase 1 amortisation effect (collect
%   called ONCE per M and reused across irreps) is directly visible.
%
%   Test cases:
%       s = 1/2 (M = 0): all 10 irreps
%       s = 1   (M = 0): A_g, T1g, H_g (covers d = 1, 3, 5)
%
%   The s = 2 case is left for the full pipeline benchmark
%   (bench_s2_M0_breakdown) where the wall-time picture is meaningful.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    fprintf('=== Regression: CLT refactor (two-phase vs monolithic) ===\n\n');

    group = icosahedron_Ih_full();
    bonds = adjacency_icosahedron_Ih();
    J = 1.0;
    tol_M = 1e-12;

    cases = {
        struct('s', 0.5, 'M', 0, 'names', {{'A_g','A_u','T1g','T1u','T2g','T2u','F_g','F_u','H_g','H_u'}})
        struct('s', 1.0, 'M', 0, 'names', {{'A_g','T1g','H_g'}})
    };

    overall = true;
    for k = 1 : numel(cases)
        c = cases{k};
        fprintf('--- s = %g, M = %d ---\n', c.s, c.M);

        cache_M = enumerate_M_orbits_Ih_gpu(c.s, c.M, group);

        % Two-phase: collect entries ONCE for this M.
        t0 = tic;
        entries = collect_clt_entries_Ih(cache_M.super_reps, bonds, c.s, J, group);
        t_collect = toc(t0);
        fprintf('  collect_clt_entries_Ih (once)  : %.3f s   (n_entries = %d)\n', ...
                t_collect, entries.n_entries);

        irreps = build_irreps_table(group);

        t_mono_sum = 0;
        t_phase2_sum = 0;
        n_checked = 0;

        for ig = 1 : numel(irreps)
            ir = irreps{ig};
            if ~any(strcmp(c.names, ir.name)), continue; end

            [reps, V_per_rep, eig_per_rep, n_per_rep, ~] = ...
                apply_irrep_to_orbits(cache_M, ir.data, ir.d, group);

            if sum(n_per_rep) == 0
                fprintf('    %-4s: empty, skip\n', ir.name);
                continue;
            end

            % Reference: monolithic build
            t0 = tic;
            clt_mono = build_clt_pg_Ih(reps, V_per_rep, eig_per_rep, ...
                n_per_rep, bonds, c.s, J, ir.data, ir.d, group);
            t_mono = toc(t0);
            t_mono_sum = t_mono_sum + t_mono;

            % New: phase 2 only
            t0 = tic;
            clt_new = build_clt_from_entries_Ih(entries, reps, V_per_rep, eig_per_rep, ...
                n_per_rep, ir.data, ir.d, group);
            t_phase2 = toc(t0);
            t_phase2_sum = t_phase2_sum + t_phase2;

            % Also check the GPU pagemtimes variant (Option A) for
            % bit-for-bit equality (within FP-double tolerance).
            clt_gpu = build_clt_from_entries_Ih_gpu(entries, reps, V_per_rep, ...
                eig_per_rep, n_per_rep, ir.data, ir.d, group);
            dM_gpu = norm(clt_mono.M(:) - clt_gpu.M(:)) / max(norm(clt_mono.M(:)), 1e-20);
            ok_gpu = dM_gpu < 1e-10;     % cuBLAS may reorder accumulation
            overall = overall && ok_gpu;

            % Field-by-field check
            ok_basis = clt_mono.n_basis == clt_new.n_basis;
            ok_reps  = clt_mono.n_reps  == clt_new.n_reps;
            ok_d     = clt_mono.d_irrep == clt_new.d_irrep;
            ok_ro    = isequal(clt_mono.rep_offsets, clt_new.rep_offsets);
            ok_npr   = isequal(clt_mono.n_per_rep, clt_new.n_per_rep);
            ok_dv    = max(abs(clt_mono.diag_vals - clt_new.diag_vals)) < 1e-13;
            ok_epr   = isequal(clt_mono.entries_per_rep, clt_new.entries_per_rep);
            ok_eo    = isequal(clt_mono.entry_offsets, clt_new.entry_offsets);
            ok_src   = isequal(clt_mono.src_idx, clt_new.src_idx);

            dM = norm(clt_mono.M(:) - clt_new.M(:)) / max(norm(clt_mono.M(:)), 1e-20);
            ok_M = dM < tol_M;

            ok = ok_basis && ok_reps && ok_d && ok_ro && ok_npr && ok_dv ...
                 && ok_epr && ok_eo && ok_src && ok_M;
            overall = overall && ok;
            n_checked = n_checked + 1;

            fprintf('    %-4s (d=%d) n_basis=%5d: mono=%.3fs, phase2=%.3fs, ||dM||/||M||(cpu)=%.1e (gpu)=%.1e   [%s/%s]\n', ...
                ir.name, ir.d, clt_mono.n_basis, t_mono, t_phase2, dM, dM_gpu, ...
                tern(ok, 'OK', 'FAIL'), tern(ok_gpu, 'OK', 'FAIL'));
        end

        fprintf('\n  TOTAL monolithic (%d irreps)  : %.3f s\n', n_checked, t_mono_sum);
        fprintf('  TOTAL two-phase   (collect+phase2): %.3f s   (= %.3f + %.3f)\n', ...
                t_collect + t_phase2_sum, t_collect, t_phase2_sum);
        fprintf('  saved by amortising Phase 1 : %.3f s   (x%.2f speedup)\n\n', ...
                t_mono_sum - (t_collect + t_phase2_sum), ...
                t_mono_sum / max(t_collect + t_phase2_sum, 1e-9));
    end

    fprintf('===============================================\n');
    fprintf('OVERALL: %s\n', tern(overall, 'PASS', 'FAIL'));
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

function s = tern(c, a, b)
    if c, s = a; else, s = b; end
end
