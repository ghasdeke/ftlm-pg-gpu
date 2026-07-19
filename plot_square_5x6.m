function plot_square_5x6(matfile, outpng)
%PLOT_SQUARE_5X6  C(T) [and chi(T)] of a 5x6 FTLM run on a LINEAR T axis.
%
%   PLOT_SQUARE_5X6(MATFILE, OUTPNG) loads a saved FTLM result and plots the
%   specific heat C(T) and (if meaningful) the magnetic susceptibility
%   chi(T) = beta*<M^2> versus temperature on a LINEAR T axis.
%
%   If the run was restricted to M = 0 (only_M0), chi(T) is identically zero
%   by construction (no magnetisation fluctuations) and C(T) is the specific
%   heat of the M = 0 SUBSPACE only -- not the full thermodynamics. This is
%   detected automatically: a single, clearly labelled C(T) panel is drawn.
%   For a full-M run both C(T) and chi(T) are drawn.
%
%   The x-range is clipped to the informative window (the T-grid is
%   logarithmic up to ~31.6 J, so a full linear axis would crush the peak).

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    if nargin < 1 || isempty(matfile), matfile = 'ftlm_pg_gpu_Ih_sq5x6_s1o2.mat'; end
    if nargin < 2 || isempty(outpng),  outpng  = 'square_5x6_CT_chiT.png';        end

    d   = load(matfile);
    T   = d.T_range(:);
    C   = d.C_T(:);
    chi = d.chi_T(:);
    N   = 30;

    chi_is_zero = max(abs(chi)) < 1e-9 * max(abs(C));   % M=0-only -> chi == 0

    %% Data-driven linear x-limit from the C(T) decay ONLY (floored at 8 J).
    %  chi(T) has a slow 1/T Curie tail that would otherwise stretch the axis
    %  and crush the low-T peak region; C(T) decays fast and sets a good window
    %  that still shows the chi peak + its initial descent.
    iC   = find(C > 0.02 * max(C), 1, 'last');
    Tmax = min(max(8, T(iC)), max(T));

    red  = [0.85 0.10 0.10];
    blue = [0.10 0.25 0.80];

    if chi_is_zero
        fig = figure('Visible', 'off', 'Position', [100 100 820 470]);
        plot(T, C, '-o', 'Color', red, 'MarkerFaceColor', red, 'MarkerSize', 4, 'LineWidth', 1.3);
        grid on; box on; xlim([0 Tmax]);
        xlabel('temperature  T / J');
        ylabel('specific heat  C(T)  [k_B]');
        title({sprintf('5\\times6 Heisenberg AFM (s=1/2, N=%d): C(T) of the M=0 SUBSPACE', N), ...
               '(M=0-only run -- \chi(T)\equiv0; full thermodynamics needs all M sectors)'});
    else
        fig = figure('Visible', 'off', 'Position', [100 100 820 760]);
        subplot(2,1,1);
        plot(T, C, '-o', 'Color', red, 'MarkerFaceColor', red, 'MarkerSize', 4, 'LineWidth', 1.3);
        grid on; box on; xlim([0 Tmax]);
        xlabel('temperature  T / J'); ylabel('specific heat  C(T)  [k_B]');
        title(sprintf('5\\times6 Heisenberg AFM (s=1/2, N=%d), full M sweep, C_5\\timesC_6', N));
        subplot(2,1,2);
        plot(T, chi, '-o', 'Color', blue, 'MarkerFaceColor', blue, 'MarkerSize', 4, 'LineWidth', 1.3);
        grid on; box on; xlim([0 Tmax]);
        xlabel('temperature  T / J');
        ylabel('susceptibility  \chi(T)=\beta\langle M_z^2\rangle  [(g\mu_B)^2/k_B]');
    end

    exportgraphics(fig, outpng, 'Resolution', 150);
    close(fig);

    [Cmax, iCm] = max(C);
    fprintf('C(T): peak %.3f k_B at T = %.3f J  (= %.4f k_B/site)\n', Cmax, T(iCm), Cmax/N);
    if chi_is_zero
        fprintf('chi(T): identically 0 (M=0-only run).\n');
    else
        [chimax, ichm] = max(chi);
        fprintf('chi(T): peak %.4f at T = %.3f J\n', chimax, T(ichm));
    end
    fprintf('x-axis clipped to [0, %.2f] J (grid to %.1f J). Saved: %s\n', Tmax, max(T), outpng);
end
