function test_milestone_G_SpMV_CLT_Ih()
%TEST_MILESTONE_G_SPMV_CLT_IH  CLT-based gather SpMV vs matrix-free SpMV.
%
%   For each (M, Gamma) block of the s=1/2 Heisenberg icosahedron, the
%   test compares two matrix-free SpMV implementations:
%
%       Y_mf  = spmv_pg_Ih_matlab(...)            (source scatter)
%       Y_clt = spmv_pg_Ih_clt_matlab(clt, X)     (output gather, CLT)
%
%   for a random block X of shape [n_basis x B_block]. The two patterns
%   visit the same matrix elements but in different summation orders;
%   the result must agree to a few ULP times the operator norm.
%
%   Because SPMV_PG_IH_MATLAB is itself bit-exact against
%   BUILD_HEISENBERG_SPARSE_IH_GAMMA2 (test_milestone_G_SpMV_Ih), this
%   test transitively verifies that the CLT-based path is correct,
%   which in turn fixes the data layout that the CUDA kernel will
%   consume.
%
%   No GPU, no MEX. Pure MATLAB. Run from mit_pg/:
%       >> test_milestone_G_SpMV_CLT_Ih

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    fprintf('=== Milestone G SpMV CLT: gather (CLT) vs scatter (matrix-free) ===\n\n');

    group = icosahedron_Ih_full();
    bonds = adjacency_icosahedron_Ih();
    s_val = 0.5; J = 1.0;
    B_block = 4;
    tol_rel = 1e-12;

    irreps = build_irreps_table(group);

    rng(27182);
    overall = true;
    for M = 0 : 3
        fprintf('--- M = %d ---\n', M);
        for k = 1 : numel(irreps)
            ir = irreps{k};
            [super_reps, V_per_rep, eig_per_rep, n_per_rep, ~] = ...
                enumerate_sector_with_Ih_gamma2(s_val, M, ir.data, ir.d, group);
            n_basis = double(sum(n_per_rep));
            if n_basis == 0
                fprintf('    %-4s (d=%d): empty\n', ir.name, ir.d);
                continue;
            end

            % Build the CLT
            clt = build_clt_pg_Ih(super_reps, V_per_rep, eig_per_rep, ...
                n_per_rep, bonds, s_val, J, ir.data, ir.d, group);

            X = randn(n_basis, B_block) + 1i * randn(n_basis, B_block);

            Y_mf = spmv_pg_Ih_matlab(super_reps, V_per_rep, ...
                eig_per_rep, n_per_rep, bonds, s_val, J, ...
                ir.data, ir.d, group, X);

            Y_clt = spmv_pg_Ih_clt_matlab(clt, X);

            nrm = norm(Y_mf, 'fro');
            err_abs = norm(Y_mf - Y_clt, 'fro');
            if nrm < 1e-15
                err_rel = err_abs;
            else
                err_rel = err_abs / nrm;
            end
            ok = err_rel < tol_rel;
            overall = overall && ok;
            n_entries = double(sum(clt.entries_per_rep));
            fprintf('    %-4s (d=%d): n_basis = %3d, n_clt_entries = %4d, ||Y_mf - Y_clt||_rel = %.2e  [%s]\n', ...
                ir.name, ir.d, n_basis, n_entries, err_rel, ternary(ok, 'OK', 'FAIL'));
        end
    end

    fprintf('\n=========================================================\n');
    fprintf('OVERALL Milestone G SpMV CLT: %s\n', ternary(overall, 'PASS', 'FAIL'));
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

function s = ternary(cond, a, b)
    if cond, s = a; else, s = b; end
end
