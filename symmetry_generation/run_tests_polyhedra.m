%% run_tests_polyhedra.m -- verification for the icosahedron / dodecahedron I_h study
%
% Validates the geometric symmetry group, the Dixon irreps, and the M=0
% per-irrep ground-state computation for both solids.  Each check prints
% PASS / FAIL against an explicit tolerance.

clear; clc;
tol = 1e-8;
names = {};  oks = [];

fprintf('Building icosahedron (12 sites) ...\n');
oi = polyhedron_irrep_groundstates('icosahedron', struct('verbose', false));
fprintf('Building dodecahedron (20 sites) ...\n');
od = polyhedron_irrep_groundstates('dodecahedron', struct('verbose', false));

expDims = [1 1 3 3 3 3 4 4 5 5];

% ---- group / irrep structure ----
names{end+1} = 'Icosa: group order |G| = 120';
oks(end+1)   = oi.G.n == 120;
names{end+1} = 'Icosa: 10 inequivalent irreps';
oks(end+1)   = oi.info.nirr == 10;
names{end+1} = 'Icosa: irrep dims = [1 1 3 3 3 3 4 4 5 5]';
oks(end+1)   = isequal(sort(oi.info.dims), expDims);
names{end+1} = 'Icosa: completeness sum d^2 = 120';
oks(end+1)   = sum(oi.info.dims.^2) == 120;
names{end+1} = 'Icosa: 10 distinct irrep labels';
oks(end+1)   = numel(unique({oi.R.label})) == 10;

names{end+1} = 'Dodeca: group order |G| = 120';
oks(end+1)   = od.G.n == 120;
names{end+1} = 'Dodeca: irrep dims = [1 1 3 3 3 3 4 4 5 5]';
oks(end+1)   = isequal(sort(od.info.dims), expDims);
names{end+1} = 'Dodeca: completeness sum d^2 = 120';
oks(end+1)   = sum(od.info.dims.^2) == 120;

% ---- sector dimension counts ----
names{end+1} = 'Icosa: sum sector dims = C(12,6) = 924';
oks(end+1)   = sum([oi.R.secdim]) == nchoosek(12, 6);
names{end+1} = 'Dodeca: sum sector dims = C(20,10) = 184756';
oks(end+1)   = sum([od.R.secdim]) == nchoosek(20, 10);

% ---- eigs vs dense (icosahedron, per sector) ----
names{end+1} = 'Icosa: eigs == dense diagonalisation (every sector)';
oks(end+1)   = max(abs([oi.R.E0] - [oi.R.E0_exact])) < tol;

% ---- symmetry-resolved ground == direct symmetry-free ground ----
names{end+1} = 'Icosa: min-over-irreps == direct global ground';
oks(end+1)   = abs(oi.global_ground - oi.global_direct) < tol;
names{end+1} = 'Dodeca: min-over-irreps == direct global ground';
oks(end+1)   = abs(od.global_ground - od.global_direct) < tol;

% ---- report ----
fprintf('\n==== POLYHEDRA TEST RESULTS (tol = %.1e) ====\n', tol);
for k = 1:numel(names)
    if oks(k), st = 'PASS'; else, st = 'FAIL'; end
    fprintf('  [%s] %s\n', st, names{k});
end
fprintf('-------------------------------------------------\n');
fprintf('%d passed, %d failed\n', sum(oks), sum(~oks));
if all(oks), fprintf('ALL TESTS PASSED\n'); else, fprintf('SOME TESTS FAILED\n'); end

fprintf('\nIcosahedron  global M=0 ground = %.8f  (%s)\n', ...
        oi.global_ground, oi.R(argmin([oi.R.E0])).label);
fprintf('Dodecahedron global M=0 ground = %.8f  (%s)\n', ...
        od.global_ground, od.R(argmin([od.R.E0])).label);

function i = argmin(x)
    [~, i] = min(x);
end
