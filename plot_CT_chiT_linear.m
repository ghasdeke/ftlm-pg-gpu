function plot_CT_chiT_linear(matfile, tmax, npts, outpng)
%PLOT_CT_CHIT_LINEAR  Plot C(T) and chi(T) (stacked panels) on a LINEAR T axis.
%
%   plot_CT_chiT_linear(matfile, tmax, npts, outpng)
%     matfile : FTLM result .mat (needs T_range/C_T/chi_T, or all_E/all_w/all_M
%               when npts is given). Default the icosido full-sweep result.
%     tmax    : upper T limit on the linear axis (default 10).
%     npts    : if given, RE-EVALUATE the observables on a fresh grid
%               linspace(0, tmax, npts) from the stored all_E/all_w/all_M
%               (no re-run). The grid starts AT T=0; the T=0 point is evaluated
%               at a tiny epsilon to show the genuine T->0+ limit (beta=1/T would
%               be Inf at T=0 -> NaN). If empty, use the stored C_T/chi_T.
%     outpng  : output PNG path (default under figures/).
% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    if nargin < 1 || isempty(matfile), matfile = 'ftlm_pg_gpu_Ih_icosido_s1o2.mat'; end
    if nargin < 2 || isempty(tmax),    tmax    = 10; end
    if nargin < 3,                     npts    = []; end
    [~, stem] = fileparts(matfile);
    if nargin < 4 || isempty(outpng)
        if isempty(npts)
            outpng = fullfile('figures', [stem '_CT_chiT_lin.png']);
        else
            outpng = fullfile('figures', sprintf('%s_CT_chiT_0to%g_%dpts.png', stem, tmax, npts));
        end
    end

    if isempty(npts)
        S = load(matfile, 'T_range', 'C_T', 'chi_T');
        T = S.T_range(:);  C = S.C_T(:);  X = S.chi_T(:);
    else
        S = load(matfile, 'all_E', 'all_w', 'all_M');
        T = linspace(0, tmax, npts)';
        Tcalc = T;
        if T(1) == 0, Tcalc(1) = T(2) * 1e-3; end     % T->0+ limit (avoid 1/0 -> NaN)
        [C, X] = compute_observables_pg(S.all_E, S.all_w, S.all_M, Tcalc);
        C = C(:);  X = X(:);
    end

    in = T <= tmax;
    [cmax, ic] = max(C(in));  [xmax, ix] = max(X(in));
    Ti = T(in);  c0 = C(find(in, 1));  x0 = X(find(in, 1));
    big = (numel(T) > 120);
    mk = {'-o', 'MarkerSize', 3.5};  if big, mk = {'-', 'LineWidth', 1.6}; else, mk = [mk, {'LineWidth', 1.6}]; end

    fig = figure('Visible', 'off', 'Position', [100 100 720 760], 'Color', 'w');
    try, theme(fig, 'light'); catch, end
    tl  = tiledlayout(fig, 2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

    nexttile;
    plot(T, C, mk{:}, 'Color', [0 0.45 0.74]); hold on;
    plot(Ti(ic), cmax, 'k^', 'MarkerFaceColor', 'k', 'MarkerSize', 7);
    grid on; box on; xlim([0 tmax]); ylim([0 1.1*max(cmax, eps)]);
    xlabel('T / J'); ylabel('C(T) / k_B');
    title(sprintf('Heat capacity  (peak %.3f k_B at T=%.3f J;  C(T\\rightarrow0)=%.3f)', cmax, Ti(ic), c0));

    nexttile;
    plot(T, X, mk{:}, 'Color', [0.85 0.33 0.10]); hold on;
    plot(Ti(ix), xmax, 'k^', 'MarkerFaceColor', 'k', 'MarkerSize', 7);
    grid on; box on; xlim([0 tmax]); ylim([0 1.1*max(xmax, eps)]);
    xlabel('T / J'); ylabel('\chi(T)  (1/J)');
    title(sprintf('Susceptibility  (peak %.3f at T=%.3f J;  \\chi(T\\rightarrow0)=%.3f)', xmax, Ti(ix), x0));

    title(tl, sprintf('s=1/2 Heisenberg icosidodecahedron (N=30), full M sweep'), 'FontWeight', 'bold');

    outdir = fileparts(outpng);
    if ~isempty(outdir) && ~exist(outdir, 'dir'), mkdir(outdir); end
    exportgraphics(fig, outpng, 'Resolution', 150);
    close(fig);
    fprintf('saved %s  (T=%.4f..%.4f, %d pts; C(0+)=%.4f chi(0+)=%.4f)\n', ...
        outpng, T(1), T(end), numel(T), c0, x0);
end
