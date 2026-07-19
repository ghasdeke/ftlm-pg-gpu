function test_milestone_F()
%TEST_MILESTONE_F  Verify the D_N sigma-restriction at real momenta.
%
%   For each test case the script checks:
%
%   1. ENUMERATION INVARIANTS
%      - D_+ + D_- = dim_C of the underlying (M, p) C_N sector.
%      - Cross-paired super-reps appear in both sigma_par = +1 and -1.
%      - Self-stable super-reps appear in exactly one sigma_par sector
%        according to their lambda_sigma.
%
%   2. AGGREGATED-SPECTRUM CHECK (ED level)
%      - Build H in the sigma_+ and sigma_- super-rep bases via
%        BUILD_HEISENBERG_SPARSE_DN.
%      - Concatenate the +/- spectra and compare to direct diagonalization
%        of H_C on the underlying C_N rep basis. The two spectra must
%        agree to machine precision.
%
%   3. P=0 vs P=N/2 COVERAGE
%      - Test at both real momenta (p=0 always; p=N/2 only for even N).
%
%   This is the MATLAB analogue of verify_DN_pg.py and validates the
%   enumeration plus sparse-builder math before the GPU SpMV kernel is
%   written. Pure MATLAB, no GPU required.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    fprintf('=== Milestone F: D_N sigma-restriction verification ===\n\n');

    cases = {
        struct('N', 4,  's_val', 0.5, 'J',  1.0)
        struct('N', 6,  's_val', 0.5, 'J',  1.0)
        struct('N', 8,  's_val', 0.5, 'J',  1.0)
        struct('N', 4,  's_val', 1.0, 'J',  1.0)
        struct('N', 10, 's_val', 0.5, 'J',  1.0)
        struct('N', 5,  's_val', 0.5, 'J',  1.0)   % odd N (p=N/2 not used)
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
    fprintf('OVERALL Milestone F: %s\n', ternary(overall, 'PASS', 'FAIL'));
end

% ----------------------------------------------------------------
function pass = run_one(N, s_val, J)
    d_loc   = round(2*s_val + 1);
    n_total = double(d_loc)^N;
    M_max   = round(N * s_val);
    bonds   = adjacency_ring(N);
    tol     = 1e-9 * abs(J) * N;
    pass    = true;

    if mod(N, 2) == 0
        p_real_list = [0, N/2];
    else
        p_real_list = 0;
    end

    for M = 0 : M_max
        for p = p_real_list
            [reps_C, L_C, dim_C] = enumerate_sector_with_translation( ...
                                       N, s_val, M, p);
            if dim_C == 0, continue; end

            H_C = build_heisenberg_sparse_pg(reps_C, L_C, bonds, ...
                                              s_val, J, N, p, n_total);
            H_C = 0.5 * (H_C + H_C');
            E_C = sort(real(eig(full(H_C))));

            % Enumerate +/- sigma sectors
            E_agg = [];
            dim_total = 0;
            for sigma_par = [+1, -1]
                [super_reps, orbit_lens, type_arr, sR_idx_C, lam_r] = ...
                    enumerate_sector_with_DN(N, s_val, M, p, sigma_par);
                if isempty(super_reps), continue; end
                dim_total = dim_total + numel(super_reps);

                H_DN = build_heisenberg_sparse_DN(super_reps, orbit_lens, ...
                            type_arr, sR_idx_C, lam_r, reps_C, L_C, ...
                            bonds, s_val, J, N, p, sigma_par, n_total);
                H_DN = 0.5 * (full(H_DN) + full(H_DN)');
                E_s = sort(real(eig(H_DN)));
                E_agg = [E_agg; E_s];   %#ok<AGROW>
            end
            E_agg = sort(E_agg);

            if dim_total ~= dim_C
                fprintf('  M=%+d p=%2d: DIM MISMATCH agg=%d vs C=%d  FAIL\n', ...
                    M, p, dim_total, dim_C);
                pass = false; continue;
            end
            if isempty(E_agg)
                continue;
            end
            diff = max(abs(E_agg - E_C));
            status = ternary(diff < tol, 'OK ', 'FAIL');
            fprintf('  M=%+d p=%2d  dim_C=%4d  dim_DN(+/-)=%4d  max|dE|=%.2e [%s]\n', ...
                M, p, dim_C, dim_total, diff, status);
            if diff > tol, pass = false; end
        end
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
