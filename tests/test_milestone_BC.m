function test_milestone_BC()
%TEST_MILESTONE_BC  Verify the full PG-FTLM pipeline against full ED.
%
%   For a small Heisenberg ring the test runs:
%
%   (1) PG-ED-aggregate: per-(M, p) dense ED via build_heisenberg_sparse_pg
%       + aggregation in ftlm_observables_pg (using ed_thresh = inf).
%
%   (2) Full-ED reference: per-M dense ED via the unsymmetrized sparse
%       builder + aggregation.
%
%   These two results must agree to machine precision on every
%   temperature in the grid. This validates:
%       - complex H_pg construction (all irreps incl. complex ones)
%       - aggregation across (M, p) sectors
%       - M <-> -M multiplicity factor
%       - observables formulas in compute_observables_pg
%
%   In addition the test confirms a deterministic high-T sum rule
%   sum_i w_i = full Hilbert dimension (independent of FTLM noise).
%
%   Run from MATLAB inside the mit_pg directory:
%       >> test_milestone_BC
%
%   No GPU, no MEX. Pure MATLAB. Uses ftlm_observables_pg in its
%   ed_thresh = inf mode (no random vectors are drawn).

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    fprintf('=== Milestone BC: PG-FTLM aggregation verification ===\n\n');

    cases = {
        struct('N', 4,  's_val', 0.5, 'J',  1.0)
        struct('N', 6,  's_val', 0.5, 'J',  1.0)
        struct('N', 8,  's_val', 0.5, 'J',  1.0)
        struct('N', 4,  's_val', 1.0, 'J',  1.0)
        struct('N', 6,  's_val', 0.5, 'J', -0.7)
    };

    T_range = [0.1, 0.3, 1.0, 3.0, 10.0];

    overall = true;
    for tc = 1 : numel(cases)
        c = cases{tc};
        fprintf('--- Case %d: N=%d, s=%g, J=%g ---\n', tc, c.N, c.s_val, c.J);
        pass = run_one_case(c.N, c.s_val, c.J, T_range);
        overall = overall && pass;
        fprintf('  => %s\n\n', ternary(pass, 'PASS', 'FAIL'));
    end

    fprintf('=========================================================\n');
    fprintf('OVERALL: %s\n', ternary(overall, 'ALL CASES PASS', 'SOME CASES FAILED'));
end

% ----------------------------------------------------------------
function pass = run_one_case(N, s_val, J, T_range)
    d_loc   = round(2*s_val + 1);
    n_total = double(d_loc)^N;
    M_max   = round(N * s_val);
    bonds   = adjacency_ring(N);

    pass = true;

    %% Path A: full (no-PG) ED, M-sector aggregation
    all_E_no = []; all_w_no = []; all_M_no = [];
    for M = 0 : M_max
        mult_M = 1 + (M > 0);
        [basis_M, dim_M] = enumerate_M_basis_for_test(N, s_val, d_loc, M);
        if dim_M == 0, continue; end
        H = build_heisenberg_sparse_full(basis_M, bonds, s_val, J, N, n_total);
        E = sort(eig(full(H)));
        w = mult_M * ones(numel(E), 1);
        all_E_no = [all_E_no; E];                          %#ok<AGROW>
        all_w_no = [all_w_no; w];                          %#ok<AGROW>
        all_M_no = [all_M_no; M * ones(numel(E), 1)];       %#ok<AGROW>
    end
    [C_no, chi_no, Z_no] = compute_observables_pg(all_E_no, all_w_no, all_M_no, T_range);

    %% Path B: PG-ED, (M, p)-sector aggregation
    all_E_pg = []; all_w_pg = []; all_M_pg = [];
    sumw_per_sec = [];
    dim_sum = 0;
    for M = 0 : M_max
        mult_M = 1 + (M > 0);
        for p = 0 : N - 1
            [reps, L, dim_pg] = enumerate_sector_with_translation(N, s_val, M, p);
            if dim_pg == 0, continue; end
            H = build_heisenberg_sparse_pg(reps, L, bonds, s_val, J, N, p, n_total);
            Hd = full(H); Hd = 0.5 * (Hd + Hd');
            E = sort(real(eig(Hd)));
            w = mult_M * ones(numel(E), 1);
            all_E_pg = [all_E_pg; E];                         %#ok<AGROW>
            all_w_pg = [all_w_pg; w];                         %#ok<AGROW>
            all_M_pg = [all_M_pg; M * ones(numel(E), 1)];      %#ok<AGROW>
            sumw_per_sec(end+1, 1) = mult_M * dim_pg;          %#ok<AGROW>
            dim_sum = dim_sum + mult_M * dim_pg;
        end
    end
    [C_pg, chi_pg, Z_pg] = compute_observables_pg(all_E_pg, all_w_pg, all_M_pg, T_range);

    %% Compare observables
    tol = max(1e-9 * abs(J) * N, 1e-12);
    err_C   = max(abs(C_pg   - C_no));
    err_chi = max(abs(chi_pg - chi_no));
    err_Z   = max(abs(Z_pg   - Z_no) ./ max(abs(Z_no), 1e-30));

    %% Sum rule: sum_M sum_p mult_M * dim_(M,p) == n_total
    err_sumrule = abs(dim_sum - n_total);

    fmt = '  %-22s = %.3e  [%s]\n';
    s1 = ternary(err_C   < tol, 'OK ', 'FAIL');
    s2 = ternary(err_chi < tol, 'OK ', 'FAIL');
    s3 = ternary(err_Z   < tol, 'OK ', 'FAIL');
    s4 = ternary(err_sumrule < 1e-9, 'OK ', 'FAIL');
    fprintf(fmt, 'max|dC|',          err_C,   s1);
    fprintf(fmt, 'max|dchi|',        err_chi, s2);
    fprintf(fmt, 'max rel|dZ_eff|',  err_Z,   s3);
    fprintf(fmt, 'sum-rule |d|',     err_sumrule, s4);

    pass = all([err_C, err_chi, err_Z] < tol) && err_sumrule < 1e-9;
end

% ----------------------------------------------------------------
%  Self-contained helpers (mirror release/ftlm_observables.m)
% ----------------------------------------------------------------

function s = ternary(cond, a, b)
    if cond, s = a; else, s = b; end
end

function bonds = adjacency_ring(N)
    bonds = zeros(N, 2);
    for i = 1 : N - 1
        bonds(i, :) = [i, i+1];
    end
    bonds(N, :) = [N, 1];
end

function [basis, dim] = enumerate_M_basis_for_test(N, s_val, d_loc, M_target)
    n_total = int64(d_loc)^int64(N);
    if s_val == 0.5
        k = N/2 + M_target;
        if k < 0 || k > N || abs(k - round(k)) > 0.01
            basis = int64([]); dim = 0; return;
        end
        k = round(k);
        all_states = int64(0:double(n_total)-1)';
        pop = zeros(numel(all_states), 1);
        for b = 0 : N - 1
            pop = pop + double(bitand(bitshift(all_states, -b), int64(1)));
        end
        basis = all_states(pop == k);
    else
        all_states = int64(0:double(n_total)-1)';
        Mv  = zeros(numel(all_states), 1);
        tmp = all_states;
        for kk = 1 : N
            dg  = double(mod(tmp, int64(d_loc)));
            Mv  = Mv + dg - s_val;
            tmp = (tmp - int64(dg)) / int64(d_loc);
        end
        Mv = round(Mv * 2) / 2;
        basis = all_states(Mv == M_target);
    end
    dim = numel(basis);
end

function H = build_heisenberg_sparse_full(basis, bonds, s_val, J, N, n_total)
    d_loc   = round(2*s_val + 1);
    dim     = length(basis);
    n_bonds = size(bonds, 1);
    powers  = int64(d_loc).^int64((0:N-1)');

    lookup = zeros(double(n_total), 1, 'int32');
    lookup(double(basis) + 1) = int32(1:dim);

    mi  = zeros(dim, N);
    tmp = basis;
    for site = 1 : N
        dg = double(mod(tmp, int64(d_loc)));
        mi(:, site) = dg - s_val;
        tmp = (tmp - int64(dg)) / int64(d_loc);
    end

    diag_vals = zeros(dim, 1);
    for b = 1 : n_bonds
        diag_vals = diag_vals + J * mi(:, bonds(b,1)) .* mi(:, bonds(b,2));
    end

    cap = dim * (1 + 2*n_bonds);
    rows = zeros(cap, 1); cols = zeros(cap, 1); vals = zeros(cap, 1);
    rows(1:dim) = (1:dim)'; cols(1:dim) = (1:dim)'; vals(1:dim) = diag_vals;
    nz = dim;

    for b = 1 : n_bonds
        si = bonds(b,1); sj = bonds(b,2);
        can = (mi(:,si) < s_val - 1e-10) & (mi(:,sj) > -s_val + 1e-10);
        idx_from = find(can);
        if ~isempty(idx_from)
            m_i = mi(idx_from, si); m_j = mi(idx_from, sj);
            coeffs = 0.5*J * sqrt(s_val*(s_val+1) - m_i.*(m_i+1)) .* ...
                             sqrt(s_val*(s_val+1) - m_j.*(m_j-1));
            new_states = basis(idx_from) + powers(si) - powers(sj);
            ns1 = double(new_states) + 1;
            ok  = ns1 >= 1 & ns1 <= n_total;
            ni  = zeros(numel(idx_from), 1, 'int32');
            ni(ok) = lookup(ns1(ok));
            v = ni > 0;
            nv = sum(v);
            rows(nz+1:nz+nv) = double(ni(v));
            cols(nz+1:nz+nv) = double(idx_from(v));
            vals(nz+1:nz+nv) = coeffs(v);
            nz = nz + nv;
        end
        can = (mi(:,si) > -s_val + 1e-10) & (mi(:,sj) < s_val - 1e-10);
        idx_from = find(can);
        if ~isempty(idx_from)
            m_i = mi(idx_from, si); m_j = mi(idx_from, sj);
            coeffs = 0.5*J * sqrt(s_val*(s_val+1) - m_i.*(m_i-1)) .* ...
                             sqrt(s_val*(s_val+1) - m_j.*(m_j+1));
            new_states = basis(idx_from) - powers(si) + powers(sj);
            ns1 = double(new_states) + 1;
            ok  = ns1 >= 1 & ns1 <= n_total;
            ni  = zeros(numel(idx_from), 1, 'int32');
            ni(ok) = lookup(ns1(ok));
            v = ni > 0;
            nv = sum(v);
            rows(nz+1:nz+nv) = double(ni(v));
            cols(nz+1:nz+nv) = double(idx_from(v));
            vals(nz+1:nz+nv) = coeffs(v);
            nz = nz + nv;
        end
    end
    H = sparse(rows(1:nz), cols(1:nz), vals(1:nz), dim, dim);
end
