function fig1_validation()
%FIG1_VALIDATION  Paper Fig 1: symmetry-adapted FTLM/ED vs independent full ED.
%
%   Regenerates the method-validation figure across three lattice geometries,
%   each at a size where an independent brute-force full ED (every eigenvalue,
%   total-Sz only) is affordable as the ground truth:
%
%       (a) square 4x4   (N=16, C_4 x C_4 translations)  -- also overlays the
%           genuine finite-R FTLM curve (the largest block, C(16,8)=12870,
%           is well above ed_thresh so this exercises real FTLM statistics);
%       (b) kagome  N=12 (a,b)=(2,0), C_6v space group, |G|=48  -- frustrated;
%       (c) triangular N=12 (a,b)=(2,2), C_6v space group, |G|=144 -- frustrated.
%
%   For each geometry the symmetry-adapted EXACT run (driver, ed_thresh=inf)
%   must reproduce the full-ED heat capacity to ~1e-10 (it does, ~1e-13),
%   proving the provider + generic irrep extractor + driver tile the Hilbert
%   space correctly. The 4x4 panel additionally shows that finite-R FTLM
%   tracks ED. Produces figures/fig1_validation.png and .mat.
%
%   Run:  setup_paths; fig1_validation
%
%   See also VALIDATE_SQUARE_4X4, VALIDATE_KAGOME12, VALIDATE_TRIANGULAR12,
%            ED_FULL_HEISENBERG, COMPUTE_OBSERVABLES_PG.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    s_val = 0.5;  J = 1.0;
    P = struct('tag', {}, 'title', {}, 'T', {}, 'C_ED', {}, 'C_sym', {}, ...
               'C_ftlm', {}, 'err_sym', {}, 'err_ftlm', {}, 'sumw_err', {}, ...
               'R', {}, 'M_lz', {});

    %% ---- (a) square 4x4 : full ED + symED + finite-R FTLM -----------------
    fprintf('\n===== (a) square 4x4 =====\n');
    r_ftlm = ftlm_observables_pg_Ih('input_square_4x4_s12.m');
    T      = r_ftlm.T_range(:)';
    r_sym  = ftlm_observables_pg_Ih('input_square_4x4_s12_ED.m');
    assert(isequal(r_sym.T_range(:)', T), 'square: T grids differ');
    E      = ed_full_heisenberg(adjacency_square_lattice(4, 4), 16, s_val, J);
    C_ED   = compute_observables_pg(E, ones(numel(E),1), zeros(numel(E),1), T);
    P(end+1) = pack('sq4x4', 'square 4\times4  (N=16)', T, C_ED, r_sym.C_T, ...
                    r_ftlm.C_T, r_sym.sum_w, 2^16, r_ftlm.R, r_ftlm.M_lz);

    %% ---- (b) kagome N=12 : full ED + symED --------------------------------
    fprintf('\n===== (b) kagome N=12 (2,0) =====\n');
    [~, bonds_k] = kagome_spacegroup(2, 0);
    r_k = ftlm_observables_pg_Ih('input_kagome12_ED.m');
    Tk  = r_k.T_range(:)';
    Ek  = ed_full_heisenberg(bonds_k, 12, s_val, J);
    Ck  = compute_observables_pg(Ek, ones(numel(Ek),1), zeros(numel(Ek),1), Tk);
    P(end+1) = pack('kag12', 'kagome  N=12  (frustrated)', Tk, Ck, r_k.C_T, ...
                    [], r_k.sum_w, 2^12, r_k.R, r_k.M_lz);

    %% ---- (c) triangular N=12 : full ED + symED ----------------------------
    fprintf('\n===== (c) triangular N=12 (2,2) =====\n');
    [gt, bonds_t] = triangular_spacegroup(2, 2);
    r_t = ftlm_observables_pg_Ih('input_triangular12_ED.m');
    Tt  = r_t.T_range(:)';
    Et  = ed_full_heisenberg(bonds_t, gt.N, s_val, J);
    Ct  = compute_observables_pg(Et, ones(numel(Et),1), zeros(numel(Et),1), Tt);
    P(end+1) = pack('tri12', 'triangular  N=12  (frustrated)', Tt, Ct, r_t.C_T, ...
                    [], r_t.sum_w, 2^12, r_t.R, r_t.M_lz);

    %% ---- combined figure --------------------------------------------------
    fig = figure('Visible', 'off', 'Position', [80 80 1500 460], 'Color', 'w');
    try, theme(fig, 'light'); catch, end          % force light theme (R2025+)
    for i = 1:numel(P)
        ax = subplot(1, 3, i);
        set(ax, 'Color', 'w', 'XColor', 'k', 'YColor', 'k', ...
            'GridColor', [0.2 0.2 0.2], 'GridAlpha', 0.25);
        semilogx(P(i).T, P(i).C_ED, 'k-', 'LineWidth', 2, ...
                 'DisplayName', 'full ED (reference)'); hold on;
        semilogx(P(i).T, P(i).C_sym, 'bo', 'MarkerSize', 5, 'LineWidth', 1.0, ...
                 'DisplayName', 'sym-adapted ED');
        if ~isempty(P(i).C_ftlm)
            semilogx(P(i).T, P(i).C_ftlm, 'r^', 'MarkerSize', 5, 'LineWidth', 1.0, ...
                'DisplayName', sprintf('FTLM (R=%d, M_{lz}=%d)', P(i).R, P(i).M_lz));
        end
        grid on; box on;
        xlabel('T / J'); ylabel('C(T)  [k_B]');
        title(P(i).title, 'Interpreter', 'tex');
        legend('Location', 'northeast', 'FontSize', 8);
        txt = sprintf('symED vs ED: %.1e\n\\Sigma w err: %.1e', ...
                      P(i).err_sym, P(i).sumw_err);
        if ~isempty(P(i).C_ftlm)
            txt = sprintf('%s\nFTLM vs ED: %.1e', txt, P(i).err_ftlm);
        end
        yl = ylim; xl = xlim;
        text(xl(1)*1.3, yl(2)*0.62, txt, 'FontSize', 8, 'Interpreter', 'tex', ...
             'BackgroundColor', [1 1 1 0.6], 'EdgeColor', [0.7 0.7 0.7]);
    end
    sgtitle('Fig 1  —  symmetry-adapted FTLM/ED reproduces independent full ED (s=1/2 Heisenberg AFM)', ...
            'FontWeight', 'bold');
    png = fullfile('figures', 'fig1_validation.png');
    exportgraphics(fig, png, 'Resolution', 200);
    close(fig);

    %% ---- save + summary ---------------------------------------------------
    save(fullfile('figures', 'fig1_validation.mat'), 'P', '-v7.3');
    fprintf('\n=== Fig 1 summary ===\n');
    for i = 1:numel(P)
        fprintf('  %-8s symED-vs-ED %.2e | sumw %.2e', P(i).tag, P(i).err_sym, P(i).sumw_err);
        if ~isempty(P(i).C_ftlm), fprintf(' | FTLM-vs-ED %.2e', P(i).err_ftlm); end
        fprintf('\n');
    end
    fprintf('Saved %s (+ .mat)\n', png);
end

function p = pack(tag, ttl, T, C_ED, C_sym, C_ftlm, sum_w, n_full, R, M_lz)
    nz = C_ED > 1e-9 * max(C_ED);
    err_sym = max(abs(C_sym(nz) - C_ED(nz)) ./ C_ED(nz));
    if isempty(C_ftlm)
        err_ftlm = NaN;
    else
        err_ftlm = max(abs(C_ftlm(nz) - C_ED(nz)) ./ C_ED(nz));
    end
    p = struct('tag', tag, 'title', ttl, 'T', T, 'C_ED', C_ED, 'C_sym', C_sym, ...
        'C_ftlm', C_ftlm, 'err_sym', err_sym, 'err_ftlm', err_ftlm, ...
        'sumw_err', abs(sum_w - n_full)/n_full, 'R', R, 'M_lz', M_lz);
end
