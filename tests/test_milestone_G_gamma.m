function test_milestone_G_gamma()
%TEST_MILESTONE_G_GAMMA  All ten I_h irreps via Frobenius-projector ED.
%
%   For each (M, Gamma) block of the spin-1/2 Heisenberg icosahedron
%   the test:
%     1. Builds the projector
%           P_Gamma = (d_Gamma / |G|) * sum_g chi_Gamma(g)^* rho_perm(g)
%        on the M-sector basis (dim = dim_M).
%     2. Diagonalizes P_Gamma to get a basis U_Gamma of its rank-r
%        subspace (rank equals the multiplicity n_Gamma times the irrep
%        dimension d_Gamma).
%     3. Restricts H_M to this subspace: H_block = U_Gamma' H_M U_Gamma.
%     4. Diagonalizes H_block.
%
%   The concatenated and sorted spectra across all ten irreps must
%   reproduce the direct M-sector spectrum to machine precision.
%
%   This is Phase gamma.1 of Milestone G: a correctness reference for
%   the full I_h decomposition including the non-Abelian 3D, 4D, and
%   5D irreps. The construction does not scale beyond dim_M ~ few
%   thousand because it diagonalizes dim_M x dim_M projectors. Phase
%   gamma.2 will provide the scalable super-rep + column-picking
%   construction usable inside the FTLM Lanczos loop.
%
%   Pure MATLAB, no GPU. Run from mit_pg/:
%       >> test_milestone_G_gamma

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    fprintf('=== Milestone G Phase gamma.1: all 10 I_h irreps on s=1/2 icosahedron ===\n\n');

    group = icosahedron_Ih_full();
    bonds = adjacency_icosahedron_Ih();
    s_val = 0.5; J = 1.0;

    %% Characters from the group struct
    irreps = build_irreps_table(group);

    overall = true;
    for M = 0 : 3
        fprintf('--- M = %d ---\n', M);
        pass = run_one(M, s_val, J, bonds, group, irreps);
        overall = overall && pass;
    end
    fprintf('\n=========================================================\n');
    fprintf('OVERALL Phase gamma.1: %s\n', ternary(overall, 'PASS', 'FAIL'));
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
        P = build_projector_M(states_M, ir.chars, ir.d, group, s_val);
        P = 0.5 * (P + P');
        [V, D] = eig(P);
        eigs_P = real(diag(D));
        plus_idx = find(eigs_P > 0.5);
        n_basis = numel(plus_idx);
        dim_check = dim_check + n_basis;
        if n_basis == 0
            fprintf('    %-4s (d=%d): block dim = 0\n', ir.name, ir.d);
            continue;
        end
        U = V(:, plus_idx);
        H_block = U' * H_M * U;
        H_block = 0.5 * (H_block + H_block');
        E_block = sort(real(eig(H_block)));
        all_E = [all_E; E_block]; %#ok<AGROW>
        fprintf('    %-4s (d=%d): block dim = %d\n', ir.name, ir.d, n_basis);
    end
    all_E = sort(all_E);

    tol = 1e-10 * abs(J) * N_sites;
    pass_dim = (dim_check == dim_M);
    if pass_dim
        err = max(abs(all_E - E_full));
    else
        err = inf;
    end
    fprintf('  Total block dims = %d (expected %d)  [%s]\n', dim_check, dim_M, ternary(pass_dim, 'OK', 'FAIL'));
    fprintf('  max|dE| aggregated vs full = %.2e  [%s]\n', err, ternary(err < tol, 'OK', 'FAIL'));
    pass = pass_dim && (err < tol);
end

% ----------------------------------------------------------------
function irreps = build_irreps_table(group)
    % chars(g) = trace(rho_Gamma(g))
    function ch = ch_scalar(v)
        ch = double(v);
    end
    function ch = ch_matrix(M)
        K = size(M, 3);
        ch = zeros(K, 1);
        for k = 1 : K, ch(k) = trace(M(:, :, k)); end
    end
    irreps = {};
    irreps{end+1} = struct('name', 'A_g',  'd', 1, 'chars', ch_scalar(group.Ag));
    irreps{end+1} = struct('name', 'A_u',  'd', 1, 'chars', ch_scalar(group.Au));
    irreps{end+1} = struct('name', 'T1g',  'd', 3, 'chars', ch_matrix(group.T1g));
    irreps{end+1} = struct('name', 'T1u',  'd', 3, 'chars', ch_matrix(group.T1u));
    irreps{end+1} = struct('name', 'T2g',  'd', 3, 'chars', ch_matrix(group.T2g));
    irreps{end+1} = struct('name', 'T2u',  'd', 3, 'chars', ch_matrix(group.T2u));
    irreps{end+1} = struct('name', 'F_g',  'd', 4, 'chars', ch_matrix(group.Fg));
    irreps{end+1} = struct('name', 'F_u',  'd', 4, 'chars', ch_matrix(group.Fu));
    irreps{end+1} = struct('name', 'H_g',  'd', 5, 'chars', ch_matrix(group.Hg));
    irreps{end+1} = struct('name', 'H_u',  'd', 5, 'chars', ch_matrix(group.Hu));
end

% ----------------------------------------------------------------
function P = build_projector_M(states_M, chars, d_irrep, group, s_val)
    d_loc = round(2*s_val + 1);
    N_sites = 12;
    dim_M = numel(states_M);
    idx_map = zeros(double(d_loc)^N_sites, 1, 'int32');
    idx_map(double(states_M) + 1) = int32(1 : dim_M);
    P = zeros(dim_M, dim_M);
    for g = 1 : group.order
        ch_star = conj(chars(g));
        if abs(ch_star) < 1e-14, continue; end
        perm_g = double(group.perms(g, :));
        states_g = apply_perm_to_state(perm_g, states_M, d_loc, N_sites);
        for i = 1 : dim_M
            j = idx_map(double(states_g(i)) + 1);
            if j > 0
                P(j, i) = P(j, i) + ch_star;
            end
        end
    end
    P = (d_irrep / group.order) * P;
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
