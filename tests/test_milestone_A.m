function test_milestone_A()
%TEST_MILESTONE_A  Verify the rep-basis Heisenberg builder against full ED.
%
%   For each (N, s) test case on a ring with periodic boundary:
%       1. Build the full M-sector Hamiltonian without symmetry adaptation
%          (sparse, identical convention to release/ftlm_observables.m).
%       2. For each irrep p = 0..N-1, build the (M, p)-sector Hamiltonian
%          via build_heisenberg_sparse_pg.
%       3. Aggregate the (M, p) spectra and compare to the full M-sector
%          spectrum (sorted, pointwise).
%
%   PASS criterion (per sector): max |E_full - E_pg_aggregate| < tol.
%   This validates jointly:
%       - rep enumeration
%       - orbit-length bookkeeping
%       - C_N compatibility filter
%       - matrix-element formula with phase and sqrt(L_r / L_a) factor.
%
%   Run from MATLAB inside the mit_pg directory:
%       >> test_milestone_A
%
%   No GPU, no MEX. Pure MATLAB.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    fprintf('=== Milestone A: rep-basis ED verification on spin rings ===\n\n');

    cases = {
        struct('N', 4,  's_val', 0.5, 'J',  1.0)
        struct('N', 6,  's_val', 0.5, 'J',  1.0)
        struct('N', 8,  's_val', 0.5, 'J',  1.0)
        struct('N', 4,  's_val', 1.0, 'J',  1.0)
        struct('N', 10, 's_val', 0.5, 'J',  1.0)
        struct('N', 4,  's_val', 0.5, 'J', -0.7)
    };

    overall = true;
    for tc = 1 : numel(cases)
        c = cases{tc};
        fprintf('--- Case %d: N=%d, s=%g, J=%g ---\n', tc, c.N, c.s_val, c.J);
        pass = run_one_case(c.N, c.s_val, c.J);
        overall = overall && pass;
        fprintf('  => %s\n\n', ternary(pass, 'PASS', 'FAIL'));
    end

    fprintf('=========================================================\n');
    fprintf('OVERALL: %s\n', ternary(overall, 'ALL CASES PASS', 'SOME CASES FAILED'));
end

% ----------------------------------------------------------------
function pass = run_one_case(N, s_val, J)
    d_loc   = round(2*s_val + 1);
    n_total = double(d_loc)^N;
    M_max   = round(N * s_val);
    bonds   = adjacency_ring(N);
    tol     = max(1e-10 * abs(J) * N, 1e-12);

    pass = true;

    for M = -M_max : M_max
        [basis_full, dim_full] = enumerate_M_basis_test(N, s_val, d_loc, M);
        if dim_full == 0, continue; end

        % Reference: full M-sector spectrum.
        H_full = build_heisenberg_sparse_full(basis_full, bonds, ...
                                              s_val, J, N, n_total);
        E_full = sort(eig(full(H_full)));

        % Aggregate from (M, p) sectors.
        E_agg = [];
        per_sector_dims = '';
        for p = 0 : N - 1
            [reps, L, dim_pg] = enumerate_sector_with_translation( ...
                                    N, s_val, M, p);
            if dim_pg == 0, continue; end
            H_pg = build_heisenberg_sparse_pg(reps, L, bonds, ...
                                              s_val, J, N, p, n_total);
            % Hermitize against round-off (eig of dense complex).
            Hd = full(H_pg);
            Hd = 0.5 * (Hd + Hd');
            E_pg = sort(real(eig(Hd)));
            E_agg = [E_agg; E_pg]; %#ok<AGROW>
            per_sector_dims = [per_sector_dims, ...
                sprintf('p=%d:%d ', p, dim_pg)]; %#ok<AGROW>
        end
        E_agg = sort(E_agg);

        if numel(E_agg) ~= numel(E_full)
            fprintf('  M=%+d: DIM MISMATCH full=%d, agg=%d\n', ...
                M, numel(E_full), numel(E_agg));
            pass = false;
            continue;
        end
        diff = max(abs(E_agg - E_full));
        status = ternary(diff < tol, 'OK ', 'FAIL');
        fprintf('  M=%+d: dim_full=%4d, [%s] max|dE|=%.2e [%s]\n', ...
            M, dim_full, strtrim(per_sector_dims), diff, status);
        if diff > tol, pass = false; end
    end
end

% ----------------------------------------------------------------
%  Local helpers (kept self-contained so the test does not depend
%  on release/ ftlm_observables.m being on the MATLAB path).
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

function [basis, dim] = enumerate_M_basis_test(N, s_val, d_loc, M_target)
    n_total = int64(d_loc)^int64(N);
    if s_val == 0.5
        k = N/2 + M_target;
        if k < 0 || k > N || abs(k - round(k)) > 0.01
            basis = int64([]); dim = 0; return;
        end
        k = round(k);
        all_states = int64(0:double(n_total)-1)';
        % popcount via bitand chain (fine for small N).
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
% Reference: unsymmetrized M-sector Heisenberg sparse matrix. Mirrors
% build_heisenberg_sparse in release/ftlm_observables.m.
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
    rows = zeros(cap, 1);
    cols = zeros(cap, 1);
    vals = zeros(cap, 1);
    rows(1:dim) = (1:dim)';
    cols(1:dim) = (1:dim)';
    vals(1:dim) = diag_vals;
    nz = dim;

    for b = 1 : n_bonds
        si = bonds(b,1); sj = bonds(b,2);

        % S+_i S-_j
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

        % S-_i S+_j
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
