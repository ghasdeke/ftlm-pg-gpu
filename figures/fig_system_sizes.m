function fig_system_sizes
%FIG_SYSTEM_SIZES  Reachable system sizes vs. literature (ZUR ANSICHT).
%   Groesste behandelte Magnetisierungssektor-Dimension (ohne Symmetrie-
%   reduktion gezaehlt), eingefaerbt nach Hardware-Klasse; Marker:
%   Kreis = finite Temperatur (FTLM), Raute = nur Grundzustand.
%   Ausgabe: figures/fig_system_sizes.fig/.png

entries = {
 % Label (TeX)                                          dim      hw     finiteT
 '{\itN} = 42 kagome, spinpack (2018) [34]',            5.38e11,  'cpu',  true;
 '{\itN} = 48–50, GS Lanczos [23]',                     6.3e13,   'cpu',  false;
 '{\itN} = 30 icosidodecahedron [19]',            1.55e8,   'wgpu', true;
 '{\itN} = 20 dodecahedron, {\its} = 3/2 (this work)', 8.70e10,  'b200', true;
 '{\itN} = 40 square 5\times8 (this work)',            1.378e11, 'b200', true;
};

col = struct('cpu', [0.55 0.55 0.55], 'wgpu', [0.00 0.45 0.74], 'b200', [0.85 0.10 0.10]);
font_size = 14; label_size = 17;

fig = figure('Visible','off', 'Color','w', 'InvertHardcopy','off', ...
             'Units','pixels', 'Position',[100 100 860 400]);
try, fig.Theme = 'light'; end %#ok<TRYNC>
ax = axes(fig); hold(ax, 'on');
n = size(entries, 1);
for k = 1:n
    y = n - k + 1;
    d = entries{k,2}; hw = entries{k,3}; ft = entries{k,4};
    plot(ax, [1e7 d], [y y], '-', 'Color', [0.85 0.85 0.85], 'LineWidth', 1.0);
    if ft, mk = 'o'; else, mk = 'd'; end
    plot(ax, d, y, mk, 'MarkerSize', 11, 'MarkerFaceColor', col.(hw), ...
         'MarkerEdgeColor', 'k', 'LineWidth', 0.6);
    text(ax, 1.5e7, y + 0.32, entries{k,1}, 'Interpreter','tex', ...
         'FontName','Times New Roman', 'FontSize', font_size, 'VerticalAlignment','bottom');
end
set(ax, 'XScale', 'log');
xlim(ax, [1e7 3e14]); ylim(ax, [0.4 n + 0.9]);
ax.YTick = [];
xlabel(ax, 'dimension of the largest treated magnetization sector', ...
       'Interpreter','tex', 'FontName','Times New Roman', 'FontSize', label_size);
ax.FontSize = font_size; ax.FontName = 'Times New Roman'; ax.Box = 'on';
grid(ax, 'on'); ax.YGrid = 'off'; ax.XMinorGrid = 'off';

% Legende (Hardware-Klassen + Marker-Bedeutung) als Dummy-Handles
hcpu  = plot(ax, nan, nan, 's', 'MarkerFaceColor', col.cpu,  'MarkerEdgeColor','k');
hwgpu = plot(ax, nan, nan, 's', 'MarkerFaceColor', col.wgpu, 'MarkerEdgeColor','k');
hb200 = plot(ax, nan, nan, 's', 'MarkerFaceColor', col.b200, 'MarkerEdgeColor','k');
hft   = plot(ax, nan, nan, 'o', 'MarkerFaceColor','w', 'MarkerEdgeColor','k');
hgs   = plot(ax, nan, nan, 'd', 'MarkerFaceColor','w', 'MarkerEdgeColor','k');
lg = legend(ax, [hcpu hwgpu hb200 hft hgs], ...
    {'CPU cluster', 'workstation GPU', 'single B200', ...
     'FTLM', 'ground state only'}, ...
    'Location','southeast', 'Box','off', 'FontSize', font_size - 1, 'NumColumns', 1);
lg.FontName = 'Times New Roman';

outdir = fileparts(mfilename('fullpath'));
savefig(fig, fullfile(outdir, 'fig_system_sizes.fig'));
exportgraphics(fig, fullfile(outdir, 'fig_system_sizes.png'), ...
    'Resolution', 300, 'BackgroundColor','white');
close(fig);
fprintf('FIG-SYSTEM-SIZES-FERTIG\n');
end
