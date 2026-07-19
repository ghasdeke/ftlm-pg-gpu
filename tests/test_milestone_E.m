function test_milestone_E()
%TEST_MILESTONE_E  Verify the spin-parity reduction of (M=0, p) sectors.
%
%   For each test case the script verifies:
%
%   1. P is unitary and squares to identity in the rep basis.
%   2. P commutes with H_pg.
%   3. D_plus + D_minus = dim of the sector.
%   4. Diagonalizing P on the rep basis produces eigenvalues ±1 in the
%      counts predicted by parity_action_pg.
%   5. Projecting H_pg onto the +/- subspaces and concatenating the
%      spectra reproduces the full (M=0, p) sector spectrum to machine
%      precision.
%
%   This validates parity_action_pg + apply_parity_pg and guarantees that
%   the parity-projected FTLM Lanczos chain in the driver stays exactly
%   within the projected subspace.
%
%   Pure MATLAB, no GPU required.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    fprintf('=== Milestone E: spin-parity reduction verification ===\n\n');

    cases = {
        struct('N', 4,  's_val', 0.5, 'J',  1.0)
        struct('N', 6,  's_val', 0.5, 'J',  1.0)
        struct('N', 8,  's_val', 0.5, 'J',  1.0)
        struct('N', 4,  's_val', 1.0, 'J',  1.0)
        struct('N', 10, 's_val', 0.5, 'J',  1.0)
    };

    overall = true;
    for tc = 1 : numel(cases)
        c = cases{tc};
        fprintf('Case %d: N=%d, s=%g, J=%g\n', tc, c.N, c.s_val, c.J);
        pass = run_one(c.N, c.s_val, c.J);
        overall = overall && pass;
        fprintf('  => %s\n\n', ternary(pass, 'PASS', 'FAIL'));
    end
    fprintf('=========================================================\n');
    fprintf('OVERALL Milestone E: %s\n', ternary(overall, 'PASS', 'FAIL'));
end

% ----------------------------------------------------------------
function pass = run_one(N, s_val, J)
    d_loc   = round(2*s_val + 1);
    n_total = double(d_loc)^N;
    bonds   = adjacency_ring(N);
    tol     = 1e-9 * abs(J) * N;
    pass    = true;

    for p = 0 : N - 1
        [reps, L, dim] = enumerate_sector_with_translation(N, s_val, 0, p);
        if dim == 0, continue; end

        H  = full(build_heisenberg_sparse_pg(reps, L, bonds, s_val, J, N, p, n_total));
        H  = 0.5 * (H + H');
        E_full = sort(real(eig(H)));

        [P_idx, P_phase, D_plus, D_minus] = parity_action_pg( ...
            reps, L, N, s_val, p);

        % Build dense P as a permutation+phase matrix in the rep basis.
        P_mat = zeros(dim, dim);
        if all(imag(P_phase) == 0)
            P_mat = real(P_mat);
        else
            P_mat = complex(P_mat);
        end
        for t = 1 : dim
            P_mat(P_idx(t), t) = P_phase(t);
        end

        % Checks 1+2
        err_invol = max(max(abs(P_mat * P_mat - eye(dim))));
        err_comm  = max(max(abs(P_mat * H - H * P_mat)));
        % Check 3
        dim_ok    = (D_plus + D_minus == dim);
        % Check 4: dimensions of +/- eigenspaces of P
        E_P = sort(real(eig(0.5*(P_mat + P_mat'))));
        n_plus_obs  = sum(abs(E_P - 1) < 1e-8);
        n_minus_obs = sum(abs(E_P + 1) < 1e-8);
        dims_ok = (n_plus_obs == D_plus) && (n_minus_obs == D_minus);
        % Check 5: project H to +/- subspaces, compare aggregated spectrum
        [V_P, ~] = eig(0.5*(P_mat + P_mat'));
        E_P_diag = diag(V_P' * P_mat * V_P);
        plus_idx  = find(real(E_P_diag) >  0.5);
        minus_idx = find(real(E_P_diag) < -0.5);
        V_plus  = V_P(:, plus_idx);
        V_minus = V_P(:, minus_idx);
        H_plus  = V_plus'  * H * V_plus;   H_plus  = 0.5*(H_plus  + H_plus');
        H_minus = V_minus' * H * V_minus;  H_minus = 0.5*(H_minus + H_minus');
        E_plus  = real(eig(H_plus));
        E_minus = real(eig(H_minus));
        E_agg   = sort([E_plus; E_minus]);
        err_spec = max(abs(E_agg - E_full));

        ok = (err_invol < tol) && (err_comm < tol) && dim_ok && ...
             dims_ok && (err_spec < tol);
        if ~ok
            fprintf(['  p=%2d: dim=%d, D+=%d, D-=%d, invol=%.1e, ', ...
                     '[P,H]=%.1e, eig-dims=(%d,%d), dE=%.1e  FAIL\n'], ...
                p, dim, D_plus, D_minus, err_invol, err_comm, ...
                n_plus_obs, n_minus_obs, err_spec);
            pass = false;
        end
    end
    if pass
        fprintf('  All (M=0, p) sectors PASS (machine precision)\n');
    end
end

% ----------------------------------------------------------------
function bonds = adjacency_ring(N)
    bonds = zeros(N, 2);
    for i = 1 : N - 1, bonds(i, :) = [i, i+1]; end
    bonds(N, :) = [N, 1];
end

function s = ternary(cond, a, b)
    if cond, s = a; else, s = b; end
end
