function plot_CT_chiT_compare(matA, matB, labA, labB, tmax, npts, outpng)
%PLOT_CT_CHIT_COMPARE  Overlay C(T) and chi(T) (stacked panels) for two FTLM
%   results on a LINEAR grid linspace(0, tmax, npts), recomputed from each
%   result's all_E/all_w/all_M (no re-run). The T=0 point is evaluated at a
%   tiny epsilon (beta=1/T -> Inf at T=0). Use to compare R values / methods.
% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    if nargin < 5 || isempty(tmax), tmax = 4;    end
    if nargin < 6 || isempty(npts), npts = 1000; end
    if nargin < 7 || isempty(outpng), outpng = fullfile('figures', 'icosido_s12_CT_chiT_compare.png'); end

    [TA, CA, XA, c0A, x0A] = eval_mat(matA, tmax, npts);
    [TB, CB, XB, c0B, x0B] = eval_mat(matB, tmax, npts);

    fig = figure('Visible', 'off', 'Position', [100 100 760 780], 'Color', 'w');
    try, theme(fig, 'light'); catch, end
    tl  = tiledlayout(fig, 2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

    nexttile;
    plot(TA, CA, '-', 'LineWidth', 1.3, 'Color', [0.6 0.78 0.92]); hold on;
    plot(TB, CB, '-', 'LineWidth', 1.8, 'Color', [0 0.30 0.62]);
    grid on; box on; xlim([0 tmax]);
    xlabel('T / J'); ylabel('C(T) / k_B');
    legend({sprintf('%s  (C(0+)=%.3f)', labA, c0A), sprintf('%s  (C(0+)=%.3f)', labB, c0B)}, ...
        'Location', 'southeast');
    title('Heat capacity');

    nexttile;
    plot(TA, XA, '-', 'LineWidth', 1.3, 'Color', [0.96 0.70 0.55]); hold on;
    plot(TB, XB, '-', 'LineWidth', 1.8, 'Color', [0.75 0.22 0.0]);
    grid on; box on; xlim([0 tmax]);
    xlabel('T / J'); ylabel('\chi(T)  (1/J)');
    legend({labA, labB}, 'Location', 'northeast');
    title('Susceptibility');

    title(tl, 's=1/2 icosidodecahedron (N=30), full M sweep -- FTLM statistics R', ...
        'FontWeight', 'bold');

    outdir = fileparts(outpng);
    if ~isempty(outdir) && ~exist(outdir, 'dir'), mkdir(outdir); end
    exportgraphics(fig, outpng, 'Resolution', 150);
    close(fig);
    fprintf('saved %s\n  %s: C(0+)=%.4f chi(0+)=%.4f | %s: C(0+)=%.4f chi(0+)=%.4f\n', ...
        outpng, labA, c0A, x0A, labB, c0B, x0B);
end

% ----------------------------------------------------------------
function [T, C, X, c0, x0] = eval_mat(matfile, tmax, npts)
    S = load(matfile, 'all_E', 'all_w', 'all_M');
    T = linspace(0, tmax, npts)';
    Tcalc = T;  if T(1) == 0, Tcalc(1) = T(2) * 1e-3; end
    [C, X] = compute_observables_pg(S.all_E, S.all_w, S.all_M, Tcalc);
    C = C(:);  X = X(:);  c0 = C(1);  x0 = X(1);
end
