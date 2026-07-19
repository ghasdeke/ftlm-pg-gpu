function test_milestone_G_gamma2()
%TEST_MILESTONE_G_GAMMA2  Scalable column-picking construction for all 10
%                        I_h irreps on the s=1/2 icosahedron, verified
%                        against the Phase gamma.1 Frobenius reference.
%
%   For each M sector and each I_h irrep Gamma:
%     1. enumerate_sector_with_Ih_gamma2 returns the super-reps with
%        n_Gamma(r) > 0, plus the d_Gamma x n_Gamma(r) column-basis
%        matrix V_r and eigenvalues lambda_r,k.
%     2. build_heisenberg_sparse_Ih_gamma2 builds the sparse Hamiltonian
%        on the (M, Gamma) basis using the matrix-element formula.
%     3. We diagonalise the block, replicate each eigenvalue d_Gamma
%        times (once per partner row of Gamma), and aggregate across
%        all 10 irreps. The result must match the direct M-sector
%        spectrum to machine precision.
%
%   Reference test: TEST_MILESTONE_G_GAMMA (Phase gamma.1) builds the
%   same blocks via the Frobenius projector and does NOT scale beyond
%   dim_M ~ few thousand because it diagonalises dim_M x dim_M
%   projectors. This Phase gamma.2 construction does scale, and it is
%   the construction that an FTLM Lanczos kernel will use inside the
%   I_h-adapted SpMV.
%
%   Pure MATLAB, no GPU. Run from mit_pg/:
%       >> test_milestone_G_gamma2

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    fprintf('=== Milestone G Phase gamma.2: scalable construction, all 10 I_h irreps ===\n\n');

    group = icosahedron_Ih_full();
    bonds = adjacency_icosahedron_Ih();
    s_val = 0.5; J = 1.0;

    irreps = build_irreps_table(group);

    overall = true;
    for M = 0 : 3
        fprintf('--- M = %d ---\n', M);
        pass = run_one(M, s_val, J, bonds, group, irreps);
        overall = overall && pass;
    end
    fprintf('\n=========================================================\n');
    fprintf('OVERALL Phase gamma.2: %s\n', ternary(overall, 'PASS', 'FAIL'));
end

% ----------------------------------------------------------------
function pass = run_one(M_target, s_val, J, bonds, group, irreps)
    d_loc = round(2*s_val + 1);
    N_sites = 12;
    n_total = double(d_loc)^N_sites;

    [states_M, dim_M] = enumerate_M_basis(N_sites, s_val, d_loc, M_target);
    if dim_M == 0
        fprintf('  empty\n'); pass = true; return;
    end
    fprintf('  dim_M = %d\n', dim_M);

    H_M = build_H_M(states_M, bonds, s_val, J, N_sites, n_total);
    E_full = sort(real(eig(H_M)));

    all_E = [];
    dim_check = 0;
    for k = 1 : numel(irreps)
        ir = irreps{k};
        [super_reps, V_per_rep, eig_per_rep, n_per_rep, ~] = ...
            enumerate_sector_with_Ih_gamma2(s_val, M_target, ir.data, ...
                                             ir.d, group);
        n_basis = sum(n_per_rep);
        if n_basis == 0
            fprintf('    %-4s (d=%d): block dim = 0\n', ir.name, ir.d);
            continue;
        end

        H_block = build_heisenberg_sparse_Ih_gamma2(super_reps, ...
            V_per_rep, eig_per_rep, n_per_rep, bonds, s_val, J, ...
            ir.data, ir.d, group);
        H_block = full(H_block);

        herm_err = max(abs(H_block(:) - reshape(H_block', [], 1)));
        H_block = 0.5 * (H_block + H_block');
        E_block = sort(real(eig(H_block)));

        % Each block eigenvalue is replicated d_Gamma times in the full
        % Hilbert space (one copy per partner row of the irrep).
        for e_i = 1 : numel(E_block)
            all_E = [all_E; repmat(E_block(e_i), ir.d, 1)]; %#ok<AGROW>
        end
        dim_check = dim_check + double(n_basis) * ir.d;

        fprintf('    %-4s (d=%d): block dim = %3d x d=%d -> %3d, herm_err = %.2e\n', ...
            ir.name, ir.d, double(n_basis), ir.d, ...
            double(n_basis) * ir.d, herm_err);
    end
    all_E = sort(all_E);

    tol = 1e-10 * abs(J) * N_sites;
    pass_dim = (dim_check == dim_M);
    if pass_dim
        err = max(abs(all_E - E_full));
    else
        err = inf;
    end
    fprintf('    Aggregated = %d (expected %d)  [%s]\n', dim_check, dim_M, ...
        ternary(pass_dim, 'OK', 'FAIL'));
    fprintf('    max|dE| aggregated vs full = %.2e  [%s]\n', err, ...
        ternary(err < tol, 'OK', 'FAIL'));
    pass = pass_dim && (err < tol);
end

% ----------------------------------------------------------------
function irreps = build_irreps_table(group)
    irreps = {};
    irreps{end+1} = struct('name', 'A_g',  'd', 1, 'data', group.Ag);
    irreps{end+1} = struct('name', 'A_u',  'd', 1, 'data', group.Au);
    irreps{end+1} = struct('name', 'T1g',  'd', 3, 'data', group.T1g);
    irreps{end+1} = struct('name', 'T1u',  'd', 3, 'data', group.T1u);
    irreps{end+1} = struct('name', 'T2g',  'd', 3, 'data', group.T2g);
    irreps{end+1} = struct('name', 'T2u',  'd', 3, 'data', group.T2u);
    irreps{end+1} = struct('name', 'F_g',  'd', 4, 'data', group.Fg);
    irreps{end+1} = struct('name', 'F_u',  'd', 4, 'data', group.Fu);
    irreps{end+1} = struct('name', 'H_g',  'd', 5, 'data', group.Hg);
    irreps{end+1} = struct('name', 'H_u',  'd', 5, 'data', group.Hu);
end

% ----------------------------------------------------------------
function [basis, dim] = enumerate_M_basis(N_sites, s_val, d_loc, M_target)
    n_total = int64(d_loc)^int64(N_sites);
    if s_val == 0.5
        k_up = N_sites/2 + M_target;
        if k_up < 0 || k_up > N_sites || abs(k_up - round(k_up)) > 0.01
            basis = int64([]); dim = 0; return;
        end
        k_up = round(k_up);
        all_states = int64(0 : double(n_total) - 1)';
        pop = zeros(numel(all_states), 1);
        for b = 0 : N_sites - 1
            pop = pop + double(bitand(bitshift(all_states, -b), int64(1)));
        end
        basis = all_states(pop == k_up);
    else
        all_states = int64(0 : double(n_total) - 1)';
        Mv = zeros(numel(all_states), 1);
        tmp = all_states;
        for kk = 1 : N_sites
            dg = double(mod(tmp, int64(d_loc)));
            Mv = Mv + dg - s_val;
            tmp = (tmp - int64(dg)) / int64(d_loc);
        end
        Mv = round(Mv * 2) / 2;
        basis = all_states(Mv == M_target);
    end
    dim = numel(basis);
end

% ----------------------------------------------------------------
function H = build_H_M(basis, bonds, s_val, J, N_sites, n_total)
    d_loc = round(2*s_val + 1);
    dim = numel(basis);
    n_b = size(bonds, 1);
    powers = int64(d_loc).^int64((0 : N_sites - 1)');
    lookup = zeros(n_total, 1, 'int32');
    lookup(double(basis) + 1) = int32(1 : dim);

    mi = zeros(dim, N_sites);
    tmp = basis;
    for site = 1 : N_sites
        dg = double(mod(tmp, int64(d_loc)));
        mi(:, site) = dg - s_val;
        tmp = (tmp - int64(dg)) / int64(d_loc);
    end

    diag_vals = zeros(dim, 1);
    for b = 1 : n_b
        diag_vals = diag_vals + J * mi(:, bonds(b,1)) .* mi(:, bonds(b,2));
    end

    cap = dim * (1 + 2*n_b);
    rows = zeros(cap, 1); cols = zeros(cap, 1); vals = zeros(cap, 1);
    rows(1:dim) = (1:dim)'; cols(1:dim) = (1:dim)'; vals(1:dim) = diag_vals;
    nz = dim;
    for b = 1 : n_b
        si = bonds(b,1); sj = bonds(b,2);
        can = (mi(:,si) < s_val - 1e-10) & (mi(:,sj) > -s_val + 1e-10);
        idx_from = find(can);
        if ~isempty(idx_from)
            m_i = mi(idx_from, si); m_j = mi(idx_from, sj);
            c = 0.5*J * sqrt(s_val*(s_val+1) - m_i.*(m_i+1)) .* ...
                          sqrt(s_val*(s_val+1) - m_j.*(m_j-1));
            new_states = basis(idx_from) + powers(si) - powers(sj);
            ns1 = double(new_states) + 1;
            ok = ns1 >= 1 & ns1 <= n_total;
            ni = zeros(numel(idx_from), 1, 'int32');
            ni(ok) = lookup(ns1(ok));
            v = ni > 0; nv = sum(v);
            rows(nz+1:nz+nv) = double(ni(v));
            cols(nz+1:nz+nv) = double(idx_from(v));
            vals(nz+1:nz+nv) = c(v);
            nz = nz + nv;
        end
        can = (mi(:,si) > -s_val + 1e-10) & (mi(:,sj) < s_val - 1e-10);
        idx_from = find(can);
        if ~isempty(idx_from)
            m_i = mi(idx_from, si); m_j = mi(idx_from, sj);
            c = 0.5*J * sqrt(s_val*(s_val+1) - m_i.*(m_i-1)) .* ...
                          sqrt(s_val*(s_val+1) - m_j.*(m_j+1));
            new_states = basis(idx_from) - powers(si) + powers(sj);
            ns1 = double(new_states) + 1;
            ok = ns1 >= 1 & ns1 <= n_total;
            ni = zeros(numel(idx_from), 1, 'int32');
            ni(ok) = lookup(ns1(ok));
            v = ni > 0; nv = sum(v);
            rows(nz+1:nz+nv) = double(ni(v));
            cols(nz+1:nz+nv) = double(idx_from(v));
            vals(nz+1:nz+nv) = c(v);
            nz = nz + nv;
        end
    end
    H = full(sparse(rows(1:nz), cols(1:nz), vals(1:nz), dim, dim));
    H = 0.5 * (H + H');
end

function s = ternary(cond, a, b)
    if cond, s = a; else, s = b; end
end
