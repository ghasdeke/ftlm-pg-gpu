function validate_triangular12()
%VALIDATE_TRIANGULAR12  Verify the triangular space-group pipeline on N=12 vs ED.
%   Runs the symmetry-adapted EXACT diagonalisation over the triangular torus
%   space group on the N=12 (a,b)=(2,2) triangular cluster (s=1/2 Heisenberg)
%   and checks:
%     (1) sum of (mult_M * d_Gamma * n_basis) weights == 2^12 (the space-group
%         block decomposition tiles the full Hilbert space);
%     (2) C_symED(T) == C_full-ED(T) to ~1e-10.
%
%   This passes whether the site action realises the full C_6v space group or
%   only a faithful quotient -- either is a genuine symmetry group of H.
%
%   See also TRIANGULAR_SPACEGROUP, IRREPS_FROM_GROUP, ED_FULL_HEISENBERG,
%            FTLM_OBSERVABLES_PG_IH, VALIDATE_KAGOME12.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    [g, bonds] = triangular_spacegroup(2, 2);
    N = g.N; s_val = 0.5; J = 1.0;
    assert(N == 12, 'expected N=12');

    fprintf('\n>>> Symmetry-adapted exact run (triangular N=12, %s space group |G|=%d)\n', ...
        g.point_group, g.order);
    r = ftlm_observables_pg_Ih('input_triangular12_ED.m');
    T = r.T_range(:)';

    fprintf('\n>>> Independent full ED (dim %d)\n', 2^N);
    E = ed_full_heisenberg(bonds, N, s_val, J);
    C = compute_observables_pg(E, ones(numel(E),1), zeros(numel(E),1), T);

    nz = C > 1e-9*max(C);
    errC = max(abs(r.C_T(nz) - C(nz)) ./ C(nz));
    sumw_err = abs(r.sum_w - 2^N) / 2^N;
    fprintf('\n=== CHECKS (triangular N=12 space group) ===\n');
    fprintf('  sum-rule: sum_w = %.6g, expected %d, rel.err = %.2e\n', r.sum_w, 2^N, sumw_err);
    fprintf('  C_symED vs full ED: max rel.err = %.3e\n', errC);
    assert(sumw_err < 1e-10, 'sum-rule violated: %.2e', sumw_err);
    assert(errC < 1e-8, 'triangular space-group spectrum != full ED: %.2e', errC);
    fprintf('  PASS: triangular space-group decomposition reproduces full ED.\n');
end
