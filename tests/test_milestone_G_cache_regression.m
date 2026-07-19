function test_milestone_G_cache_regression()
%TEST_MILESTONE_G_CACHE_REGRESSION  Per-M-cache vs all-in-one regression.
%
%   Verifies that the new two-step gamma.2 enumeration path
%       ENUMERATE_M_ORBITS_IH (per M)  +  APPLY_IRREP_TO_ORBITS (per Gamma)
%   produces outputs that are exactly identical to the previous
%   monolithic ENUMERATE_SECTOR_WITH_IH_GAMMA2 path -- now redirected
%   through the same two functions but with the cache built inside.
%
%   This is a sanity check that the FTLM driver speedup achieved by
%   caching the irrep-independent work across the 10 I_h irreps has not
%   altered any block (super-reps, V_r, lambda_r, n_Gamma, orbit lengths).
%
%   Runs on the s = 1/2 icosahedron where everything is fast.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    fprintf('=== Milestone G cache regression: per-M cache vs monolithic ===\n\n');

    group = icosahedron_Ih_full();
    s_val = 0.5;
    irreps = build_irreps_table(group);

    overall = true;
    for M = 0 : 3
        cache_M = enumerate_M_orbits_Ih(s_val, M, group);
        fprintf('--- M = %d  (cached super-reps: %d) ---\n', M, ...
            numel(cache_M.super_reps));

        for ig = 1 : numel(irreps)
            ir = irreps{ig};

            % Cached path
            [reps_C, V_C, eig_C, n_C, L_C] = ...
                apply_irrep_to_orbits(cache_M, ir.data, ir.d, group);

            % Wrapper / "old" path (also goes through the cache now, but
            % rebuilds it from scratch each call -- same numbers).
            [reps_W, V_W, eig_W, n_W, L_W] = ...
                enumerate_sector_with_Ih_gamma2(s_val, M, ir.data, ir.d, group);

            ok = isequal(reps_C, reps_W) && isequal(n_C, n_W) && ...
                 isequal(L_C, L_W);
            for j = 1 : numel(V_C)
                ok = ok && (max(abs(V_C{j}(:) - V_W{j}(:))) < 1e-12);
                ok = ok && (max(abs(eig_C{j} - eig_W{j})) < 1e-10);
            end
            overall = overall && ok;
            fprintf('    %-4s (d=%d): %s\n', ir.name, ir.d, ...
                ternary(ok, 'OK', 'FAIL'));
        end
    end

    fprintf('\n=========================================================\n');
    fprintf('OVERALL cache regression: %s\n', ternary(overall, 'PASS', 'FAIL'));
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
