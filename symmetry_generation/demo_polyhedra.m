%% demo_polyhedra.m -- M=0 ground energies per I_h irrep, s=1/2 Heisenberg
%
% Computes, for the spin-1/2 Heisenberg antiferromagnet on the icosahedron
% (12 sites) and the dodecahedron (20 sites), the ground-state energy in the
% total-S_z = 0 sector for each of the 10 irreducible representations of the
% full icosahedral group I_h.  The icosahedron run also cross-checks every
% sector against a dense diagonalisation; both runs check the symmetry-resolved
% ground against a direct (symmetry-free) eigs computation.

clear; clc;

oi = polyhedron_irrep_groundstates('icosahedron');
od = polyhedron_irrep_groundstates('dodecahedron');

% --- comparison bar chart ---
try
    f = figure('Visible', 'off', 'Position', [100 100 1100 420]);
    subplot(1,2,1);
    bar([oi.R.E0]);
    set(gca, 'XTick', 1:numel(oi.R), 'XTickLabel', {oi.R.label});
    ylabel('E_0  (M=0 sector)'); title('Icosahedron  (12 spins, I_h)'); grid on;
    subplot(1,2,2);
    bar([od.R.E0]);
    set(gca, 'XTick', 1:numel(od.R), 'XTickLabel', {od.R.label});
    ylabel('E_0  (M=0 sector)'); title('Dodecahedron  (20 spins, I_h)'); grid on;
    saveas(f, 'polyhedra_groundstates.png');
    fprintf('\nSaved bar chart to polyhedra_groundstates.png\n');
catch ME
    fprintf('\n(Plotting skipped: %s)\n', ME.message);
end
