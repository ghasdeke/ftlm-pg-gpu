function test_milestone_G_SpMV_Ih()
%TEST_MILESTONE_G_SPMV_IH  Matrix-free I_h SpMV vs explicit sparse H.
%
%   For each (M, Gamma) block of the s=1/2 Heisenberg icosahedron, the
%   test compares:
%
%       Y_sparse = H_block * X        (built via the gamma.2 sparse
%                                       builder, then explicit multiply)
%       Y_spmv   = spmv_pg_Ih_matlab(X)   (matrix-free)
%
%   for a random block X of shape [n_basis x B]. The matrix-free SpMV
%   must reproduce the sparse-H result to machine precision relative
%   to ||Y_sparse||.
%
%   This validates the per-thread CUDA logic before any GPU code is
%   written. Once the matrix-free path is bit-exact in MATLAB, the
%   CUDA kernel just has to mirror it.
%
%   No GPU, no MEX. Pure MATLAB. Run from mit_pg/:
%       >> test_milestone_G_SpMV_Ih

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    fprintf('=== Milestone G SpMV: matrix-free I_h SpMV vs sparse H ===\n\n');

    group = icosahedron_Ih_full();
    bonds = adjacency_icosahedron_Ih();
    s_val = 0.5; J = 1.0;
    B_block = 4;     % block-Lanczos column count for the test
    tol_rel = 1e-12;

    irreps = build_irreps_table(group);

    rng(31415);
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

            % Build sparse H (reference)
            H_block = build_heisenberg_sparse_Ih_gamma2(super_reps, ...
                V_per_rep, eig_per_rep, n_per_rep, bonds, s_val, J, ...
                ir.data, ir.d, group);

            % Random block input. Use complex inputs unconditionally so
            % the comparison exercises both real and imag arithmetic in
            % the SpMV when irrep is genuinely complex; for real-H irreps
            % (A_g, A_u) Y is still real but accepting complex X is the
            % standard convention.
            X = randn(n_basis, B_block) + 1i * randn(n_basis, B_block);

            % Reference and matrix-free SpMV
            Y_sparse = H_block * X;
            Y_spmv   = spmv_pg_Ih_matlab(super_reps, V_per_rep, ...
                eig_per_rep, n_per_rep, bonds, s_val, J, ...
                ir.data, ir.d, group, X);

            % Compare relative to ||Y_sparse||_F
            nrm_sparse = norm(Y_sparse, 'fro');
            err_abs    = norm(Y_sparse - Y_spmv, 'fro');
            if nrm_sparse < 1e-15
                err_rel = err_abs;        % degenerate zero block
            else
                err_rel = err_abs / nrm_sparse;
            end
            ok = err_rel < tol_rel;
            overall = overall && ok;
            fprintf('    %-4s (d=%d): n_basis = %3d, ||Y_sparse - Y_spmv||_rel = %.2e  [%s]\n', ...
                ir.name, ir.d, n_basis, err_rel, ternary(ok, 'OK', 'FAIL'));
        end
    end

    fprintf('\n=========================================================\n');
    fprintf('OVERALL Milestone G SpMV: %s\n', ternary(overall, 'PASS', 'FAIL'));
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
