function validate_square_4x4()
%VALIDATE_SQUARE_4X4  Heat capacity of the 4x4 Heisenberg torus: FTLM vs ED.
%
%   Validates the square-lattice translation-symmetry FTLM pipeline on the
%   4x4 (N=16) spin-1/2 Heisenberg antiferromagnet:
%
%     1. INDEPENDENT full ED (ED_FULL_HEISENBERG): every eigenvalue, no
%        symmetry beyond total Sz -> the reference heat capacity C_ED(T).
%     2. Symmetry-adapted EXACT run (driver, ed_thresh = inf): the C_4 x C_4
%        block decomposition diagonalised exactly -> C_symED(T). Must equal
%        C_ED(T) to ~1e-10 (validates provider + generalised driver), and
%        the FTLM weights must sum to 2^16 (no states lost/double-counted).
%     3. FTLM run (driver, finite R, M_lz) -> C_FTLM(T).
%
%   Produces a plot comparing C_FTLM(T) against the full-ED reference and
%   saves it to validate_square_4x4_C.png (+ the numbers to
%   validate_square_4x4.mat).
%
%   Run:  validate_square_4x4
%
%   See also ED_FULL_HEISENBERG, FTLM_OBSERVABLES_PG_IH,
%            COMPUTE_OBSERVABLES_PG, SQUARE_LATTICE_TRANSLATION_GROUP.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    Lx = 4;  Ly = 4;  N = Lx * Ly;  s_val = 0.5;  J = 1.0;

    %% (3) FTLM run (defines the canonical T grid).
    fprintf('\n>>> FTLM run (finite R, M_lz)\n');
    r_ftlm = ftlm_observables_pg_Ih('input_square_4x4_s12.m');
    T      = r_ftlm.T_range(:)';

    %% (2) Symmetry-adapted exact run.
    fprintf('\n>>> Symmetry-adapted exact run (ed_thresh = inf)\n');
    r_sym = ftlm_observables_pg_Ih('input_square_4x4_s12_ED.m');
    assert(isequal(r_sym.T_range(:)', T), 'T grids of the two driver runs differ.');

    %% (1) Independent full ED reference.
    fprintf('\n>>> Independent full ED (%d sites, dim = %d)\n', N, 2^N);
    t_ed  = tic;
    E_all = ed_full_heisenberg(adjacency_square_lattice(Lx, Ly), N, s_val, J);
    fprintf('    full ED done in %.1f s (%d eigenvalues)\n', toc(t_ed), numel(E_all));
    C_ED  = compute_observables_pg(E_all, ones(numel(E_all), 1), ...
                                   zeros(numel(E_all), 1), T);

    %% Quantitative checks.
    nz = C_ED > 1e-9 * max(C_ED);
    err_sym  = max(abs(r_sym.C_T(nz)  - C_ED(nz)) ./ C_ED(nz));
    err_ftlm = max(abs(r_ftlm.C_T(nz) - C_ED(nz)) ./ C_ED(nz));
    sumw_err = abs(r_sym.sum_w - 2^N) / 2^N;

    fprintf('\n=== CHECKS ===\n');
    fprintf('  sum-rule (symED): sum_w = %.6g, expected %d, rel.err = %.2e\n', ...
        r_sym.sum_w, 2^N, sumw_err);
    fprintf('  C_symED vs full ED : max rel.err = %.3e  (must be ~1e-10)\n', err_sym);
    fprintf('  C_FTLM  vs full ED : max rel.err = %.3e  (FTLM statistical error)\n', err_ftlm);

    assert(sumw_err < 1e-10, 'sum-rule violated: states lost or double-counted.');
    assert(err_sym  < 1e-8,  'symmetry-adapted spectrum does NOT match full ED (err %.2e).', err_sym);
    fprintf('  PASS: symmetry-adapted ED reproduces full ED.\n');

    %% Plot: C(T) FTLM vs full ED.
    fig = figure('Visible', 'off', 'Position', [100 100 760 520]);
    semilogx(T, C_ED, 'k-', 'LineWidth', 2, 'DisplayName', 'full ED (reference)'); hold on;
    semilogx(T, r_ftlm.C_T, 'ro', 'MarkerSize', 6, 'LineWidth', 1.2, ...
        'DisplayName', sprintf('FTLM (R=%d, M_{lz}=%d)', r_ftlm.R, r_ftlm.M_lz));
    grid on; box on;
    xlabel('temperature  T / J');
    ylabel('specific heat  C(T)  [k_B]');
    title(sprintf('4x4 Heisenberg AFM (s=1/2): FTLM vs full ED  (max rel.err %.1e)', err_ftlm));
    legend('Location', 'northeast');
    png = 'validate_square_4x4_C.png';
    exportgraphics(fig, png, 'Resolution', 150);
    close(fig);
    fprintf('\nPlot saved to: %s\n', png);

    %% Save numbers.
    save('validate_square_4x4.mat', 'T', 'C_ED', 'r_ftlm', 'r_sym', ...
        'err_sym', 'err_ftlm', 'sumw_err', '-v7.3');
    fprintf('Numbers saved to: validate_square_4x4.mat\n');
    fprintf('\nALL DONE.\n');
end
