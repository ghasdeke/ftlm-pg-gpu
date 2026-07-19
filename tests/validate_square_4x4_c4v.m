function validate_square_4x4_c4v()
%VALIDATE_SQUARE_4X4_C4V  Verify the 4x4 square-lattice SPACE-GROUP pipeline.
%
%   Runs the symmetry-adapted EXACT diagonalisation over the full C_4v space
%   group (order 128) on the 4x4 s=1/2 Heisenberg AFM and checks it against
%   the independent full ED (ED_FULL_HEISENBERG):
%     (1) sum-rule: sum of (mult_M * d_Gamma * n_basis) weights == 2^16, i.e.
%         the space-group block decomposition tiles the full Hilbert space
%         (a strong test that the irreps + dimensions + multiplicities are right);
%     (2) C_symED(T) == C_full-ED(T) to ~1e-10.
%
%   See also SQUARE_LATTICE_SPACEGROUP, IRREPS_FROM_GROUP, ED_FULL_HEISENBERG,
%            FTLM_OBSERVABLES_PG_IH, VALIDATE_SQUARE_4X4.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    Lx = 4; Ly = 4; N = 16; s_val = 0.5; J = 1.0;

    fprintf('\n>>> Symmetry-adapted exact run (C_4v space group, order 128)\n');
    r = ftlm_observables_pg_Ih('input_square_4x4_c4v_ED.m');
    T = r.T_range(:)';

    fprintf('\n>>> Independent full ED (dim %d)\n', 2^N);
    E_all = ed_full_heisenberg(adjacency_square_lattice(Lx,Ly), N, s_val, J);
    C_ED  = compute_observables_pg(E_all, ones(numel(E_all),1), zeros(numel(E_all),1), T);

    nz       = C_ED > 1e-9*max(C_ED);
    err_C    = max(abs(r.C_T(nz) - C_ED(nz)) ./ C_ED(nz));
    sumw_err = abs(r.sum_w - 2^N) / 2^N;

    fprintf('\n=== CHECKS (space group) ===\n');
    fprintf('  sum-rule: sum_w = %.6g, expected %d, rel.err = %.2e\n', r.sum_w, 2^N, sumw_err);
    fprintf('  C_symED vs full ED: max rel.err = %.3e\n', err_C);

    assert(sumw_err < 1e-10, 'sum-rule violated (space-group irreps/dims wrong?): %.2e', sumw_err);
    assert(err_C    < 1e-8,  'space-group spectrum != full ED: %.2e', err_C);
    fprintf('  PASS: C_4v space-group decomposition reproduces full ED exactly.\n');
end
