function validate_cuboctahedron12()
%VALIDATE_CUBOCTAHEDRON12  Verify the cuboctahedron O_h pipeline vs full ED.
%   Three layers (mirrors VALIDATE_KAGOME12 / VALIDATE_DODECAHEDRON20):
%     (1) provider structure: |G| = 48, 24 four-coordinated edges, bond set
%         O_h-invariant, 10 generic irreps with sum d^2 = 48 and max d = 3,
%         ALL realified to real orthogonal form (O_h is ambivalent, FS=+1);
%     (2) sum rule: the (M, Gamma) block decomposition tiles the full
%         Hilbert space, sum of weights == 2^12 exactly;
%     (3) physics: C_symED(T) == C_full-ED(T) (independent dense 4096 ED).
%
%   See also CUBOCTAHEDRON_OH, IRREPS_FROM_GROUP, REALIFY_IRREPS,
%            ED_FULL_HEISENBERG, FTLM_OBSERVABLES_PG_IH.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    N = 12; s_val = 0.5; J = 1.0;

    %% (1) Provider structure.
    [group, bonds] = cuboctahedron_Oh();      % asserts |G|=48, 24 edges, deg 4,
                                              % bond invariance internally
    irr = irreps_from_group(group);
    irr = realify_irreps(irr, group);
    ds  = arrayfun(@(z) z.d, irr);
    assert(numel(irr) == 10, 'expected 10 O_h irreps, got %d', numel(irr));
    assert(sum(ds.^2) == group.order, 'sum d^2 = %d != |G| = %d', sum(ds.^2), group.order);
    assert(max(ds) == 3, 'max O_h irrep dim %d != 3', max(ds));
    for k = 1 : numel(irr)
        assert(isreal(irr(k).mats), 'O_h irrep %d not realified (FS=+1 expected)', k);
    end
    fprintf('  provider: |G|=48, 24 edges (deg 4), 10 real irreps, sum d^2 = 48  [OK]\n');

    %% (2+3) Symmetry-adapted exact run vs independent full ED.
    fprintf('\n>>> Symmetry-adapted exact run (cuboctahedron N=12, O_h)\n');
    r = ftlm_observables_pg_Ih('input_cuboctahedron12_ED.m');
    T = r.T_range(:)';

    fprintf('\n>>> Independent full ED (dim %d)\n', 2^N);
    E = ed_full_heisenberg(bonds, N, s_val, J);
    C = compute_observables_pg(E, ones(numel(E), 1), zeros(numel(E), 1), T);

    nz = C > 1e-9 * max(C);
    errC = max(abs(r.C_T(nz) - C(nz)) ./ C(nz));
    sumw_err = abs(r.sum_w - 2^N) / 2^N;
    fprintf('\n=== CHECKS (cuboctahedron N=12, O_h) ===\n');
    fprintf('  sum-rule: sum_w = %.6g, expected %d, rel.err = %.2e\n', r.sum_w, 2^N, sumw_err);
    fprintf('  C_symED vs full ED: max rel.err = %.3e\n', errC);
    fprintf('  E0 = %.6f J (frustrated cuboctahedron, N=12 s=1/2)\n', min(E));
    assert(sumw_err < 1e-10, 'sum-rule violated: %.2e', sumw_err);
    assert(errC < 1e-8, 'cuboctahedron O_h spectrum != full ED: %.2e', errC);
    fprintf('  PASS: cuboctahedron O_h decomposition reproduces full ED.\n');
end
