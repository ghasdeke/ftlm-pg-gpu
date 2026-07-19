function fig_block_reduction
%FIG_BLOCK_REDUCTION  Block-size reduction by full non-Abelian symmetry.
%   For the M = 0 sector of the s = 1/2 icosidodecahedron and the s = 3/2
%   dodecahedron: sector dimension vs. the largest block under the best
%   Abelian subgroup of I_h (order 10) vs. the largest single-partner-row
%   block of the full group (H_g, Eq. (13)). Numbers are the published
%   per-irrep dimensions (paper Tables 2 and 3).
%   Output: figures/fig_block_reduction.fig/.png

sector  = [155117520,  86981744944];    % D_{M=0}
abelian = sector / 10;                  % largest Abelian subgroup: order 10
nonab   = [6465762,    3624380734];     % largest single-row block (H_g)

data = [sector; abelian; nonab].';      % 2 Systeme x 3 Balken

col_sector = [0.55 0.55 0.55];
col_abel   = [0.00 0.45 0.74];
col_nonab  = [0.85 0.10 0.10];
font_size = 16; label_size = 19; legend_size = 14;

fig = figure('Visible','off', 'Color','w', 'InvertHardcopy','off', ...
             'Units','pixels', 'Position',[100 100 720 430]);
try, fig.Theme = 'light'; end %#ok<TRYNC>
ax = axes(fig);
hb = bar(ax, data, 0.82);
hb(1).FaceColor = col_sector; hb(2).FaceColor = col_abel; hb(3).FaceColor = col_nonab;
set(ax, 'YScale', 'log');
ylim(ax, [1e6 1e13]);
ylabel(ax, 'block dimension', 'Interpreter','latex', 'FontSize', label_size);
xticklabels(ax, {'icosidodecahedron, $s=1/2$', 'dodecahedron, $s=3/2$'});
ax.TickLabelInterpreter = 'latex';
ax.FontSize = font_size; ax.Box = 'on';
grid(ax, 'on'); ax.YMinorGrid = 'off';

% Faktor-Annotationen ueber den reduzierten Balken
for i = 1:2
    x2 = hb(2).XEndPoints(i); x3 = hb(3).XEndPoints(i);
    text(ax, x2, abelian(i)*1.45, '$\div 10$', 'Interpreter','latex', ...
        'HorizontalAlignment','center', 'FontSize', font_size);
    text(ax, x3, nonab(i)*1.45, sprintf('$\\div %d$', round(sector(i)/nonab(i))), ...
        'Interpreter','latex', 'HorizontalAlignment','center', 'FontSize', font_size);
end

legend(ax, {'$M=0$ sector', 'largest block, best Abelian subgroup', ...
            'largest block, full group (one partner row)'}, ...
       'Interpreter','latex', 'Location','northwest', 'Box','off', ...
       'FontSize', legend_size);

outdir = fileparts(mfilename('fullpath'));
savefig(fig, fullfile(outdir, 'fig_block_reduction.fig'));
exportgraphics(fig, fullfile(outdir, 'fig_block_reduction.png'), ...
    'Resolution', 300, 'BackgroundColor','white');
close(fig);
fprintf('FIG-BLOCK-REDUCTION-FERTIG\n');
end
