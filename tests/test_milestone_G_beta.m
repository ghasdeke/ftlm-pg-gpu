function test_milestone_G_beta()
%TEST_MILESTONE_G_BETA  Verify the 1D-irrep I_h pipeline on s=1/2 icosahedron.
%
%   For each M sector of the spin-1/2 Heisenberg icosahedron the test
%   compares:
%
%   (a) Reference: full M-sector ED diagonalized after projection onto
%       the A_g and A_u subspaces via the Frobenius projector
%       P_Gamma = (1/|G|) sum_g chi_Gamma(g)^* rho(g).
%
%   (b) Our construction: ENUMERATE_SECTOR_WITH_IH gives super-rep
%       lists for A_g and A_u; BUILD_HEISENBERG_SPARSE_IH builds the
%       (M, Gamma) sparse Hamiltonian. Diagonalize each.
%
%   Sanity checks:
%       - Dimensions match: |reps_Ag| = rank(P_Ag), |reps_Au| = rank(P_Au).
%       - Spectra agree to machine precision (~1e-12) per sub-block.
%
%   This is the Phase beta deliverable of Milestone G: the 1D-irrep
%   pipeline working end-to-end on the icosahedron at ED level. The
%   FTLM / GPU integration and the higher-dimensional irreps come in
%   Phase gamma and later.
%
%   Pure MATLAB, no GPU required. Run from mit_pg/:
%       >> test_milestone_G_beta

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    fprintf('=== Milestone G Phase beta: A_g + A_u on s=1/2 icosahedron ===\n\n');

    group = icosahedron_Ih_full();
    bonds = adjacency_icosahedron_Ih();

    s_val = 0.5;
    J = 1.0;
    M_list = 6 : -1 : 0;

    overall = true;
    for M = M_list
        fprintf('--- M = %d ---\n', M);
        pass = run_one(M, s_val, J, bonds, group);
        overall = overall && pass;
    end
    fprintf('\n=========================================================\n');
    fprintf('OVERALL Phase beta: %s\n', ternary(overall, 'PASS', 'FAIL'));
end

% ----------------------------------------------------------------
function pass = run_one(M_target, s_val, J, bonds, group)
    d_loc   = round(2*s_val + 1);
    N_sites = 12;
    n_total = double(d_loc)^N_sites;

    %% Build M-sector
    [states_M, dim_M] = enumerate_M_basis_test(N_sites, s_val, d_loc, M_target);
    if dim_M == 0
        fprintf('  empty\n');
        pass = true; return;
    end

    H_M = build_H_M_full(states_M, bonds, s_val, J, N_sites, n_total);

    %% Build projectors P_Ag and P_Au, compute rank and projected spectra.
    irrep_Ag = ones(group.order, 1);
    irrep_Au = double(group.det);

    [P_Ag, ranks_Ag, E_Ag_ref] = project_M_to_irrep(H_M, states_M, ...
                                  irrep_Ag, group, s_val);
    [P_Au, ranks_Au, E_Au_ref] = project_M_to_irrep(H_M, states_M, ...
                                  irrep_Au, group, s_val);

    fprintf('  dim_M = %4d, rank(P_Ag) = %3d, rank(P_Au) = %3d\n', ...
        dim_M, ranks_Ag, ranks_Au);

    %% Our pipeline
    [reps_Ag, L_Ag, ~, ~] = enumerate_sector_with_Ih(s_val, M_target, ...
                              irrep_Ag, group);
    [reps_Au, L_Au, ~, ~] = enumerate_sector_with_Ih(s_val, M_target, ...
                              irrep_Au, group);

    if numel(reps_Ag) ~= ranks_Ag || numel(reps_Au) ~= ranks_Au
        fprintf('  DIM MISMATCH: A_g %d/%d, A_u %d/%d  FAIL\n', ...
            numel(reps_Ag), ranks_Ag, numel(reps_Au), ranks_Au);
        pass = false; return;
    end

    H_Ag = build_heisenberg_sparse_Ih(reps_Ag, L_Ag, bonds, s_val, J, ...
                                       irrep_Ag, group);
    H_Au = build_heisenberg_sparse_Ih(reps_Au, L_Au, bonds, s_val, J, ...
                                       irrep_Au, group);
    H_Ag = full(H_Ag); H_Ag = 0.5 * (H_Ag + H_Ag');
    H_Au = full(H_Au); H_Au = 0.5 * (H_Au + H_Au');

    E_Ag = sort(real(eig(H_Ag)));
    E_Au = sort(real(eig(H_Au)));

    tol = 1e-10 * abs(J) * N_sites;
    if isempty(E_Ag)
        err_Ag = 0;
    else
        err_Ag = max(abs(E_Ag - E_Ag_ref));
    end
    if isempty(E_Au)
        err_Au = 0;
    else
        err_Au = max(abs(E_Au - E_Au_ref));
    end

    pass_Ag = err_Ag < tol;
    pass_Au = err_Au < tol;
    pass = pass_Ag && pass_Au;
    fprintf('  A_g: dim=%3d  max|dE|=%.2e [%s]\n', numel(E_Ag), err_Ag, ternary(pass_Ag, 'OK', 'FAIL'));
    fprintf('  A_u: dim=%3d  max|dE|=%.2e [%s]\n', numel(E_Au), err_Au, ternary(pass_Au, 'OK', 'FAIL'));
end

% ----------------------------------------------------------------
function [P, rk, E_proj] = project_M_to_irrep(H_M, states_M, irrep_chars, ...
                                                group, s_val)
    d_loc = round(2*s_val + 1);
    N_sites = 12;
    dim_M = numel(states_M);
    idx_map = zeros(double(d_loc)^N_sites, 1, 'int32');
    idx_map(double(states_M) + 1) = int32(1 : dim_M);

    P = zeros(dim_M, dim_M);
    for g = 1 : group.order
        ch = irrep_chars(g);
        if abs(ch) < 1e-14, continue; end
        perm_g = double(group.perms(g, :));
        states_g = apply_perm_to_state(perm_g, states_M, d_loc, N_sites);
        for i = 1 : dim_M
            j = idx_map(double(states_g(i)) + 1);
            if j > 0
                P(j, i) = P(j, i) + ch;
            end
        end
    end
    P = P / double(group.order);

    [V, D] = eig(0.5 * (P + P'));
    eigs_P = real(diag(D));
    plus_idx = find(eigs_P > 0.5);
    rk = numel(plus_idx);
    if rk == 0
        E_proj = [];
        return;
    end
    U = V(:, plus_idx);
    H_block = U' * H_M * U;
    H_block = 0.5 * (H_block + H_block');
    E_proj = sort(real(eig(H_block)));
end

% ----------------------------------------------------------------
function [basis, dim] = enumerate_M_basis_test(N_sites, s_val, d_loc, M_target)
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
function H = build_H_M_full(basis, bonds, s_val, J, N_sites, n_total)
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
        diag_vals = diag_vals + J * mi(:, bonds(b, 1)) .* mi(:, bonds(b, 2));
    end

    cap = dim * (1 + 2 * n_b);
    rows = zeros(cap, 1); cols = zeros(cap, 1); vals = zeros(cap, 1);
    rows(1 : dim) = (1 : dim)';
    cols(1 : dim) = (1 : dim)';
    vals(1 : dim) = diag_vals;
    nz = dim;

    for b = 1 : n_b
        si = bonds(b, 1); sj = bonds(b, 2);
        can = (mi(:, si) < s_val - 1e-10) & (mi(:, sj) > -s_val + 1e-10);
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
            rows(nz+1 : nz+nv) = double(ni(v));
            cols(nz+1 : nz+nv) = double(idx_from(v));
            vals(nz+1 : nz+nv) = c(v);
            nz = nz + nv;
        end
        can = (mi(:, si) > -s_val + 1e-10) & (mi(:, sj) < s_val - 1e-10);
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
            rows(nz+1 : nz+nv) = double(ni(v));
            cols(nz+1 : nz+nv) = double(idx_from(v));
            vals(nz+1 : nz+nv) = c(v);
            nz = nz + nv;
        end
    end

    H = sparse(rows(1 : nz), cols(1 : nz), vals(1 : nz), dim, dim);
    H = full(0.5 * (H + H'));
end

function s = ternary(cond, a, b)
    if cond, s = a; else, s = b; end
end
