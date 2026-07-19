%% demo_realify.m -- Frobenius-Schur classification and realification of irreps
%
% Shows, for several groups, which irreps are of real / complex / pseudoreal
% type, and demonstrates transforming a real-type irrep from Dixon's (complex)
% basis into an exactly real (orthogonal) form.

clear; clc;

fprintf('Frobenius-Schur classification (real +1 / complex 0 / pseudoreal -1):\n\n');
report('C6 (cyclic)',      group_closure({[2 3 4 5 6 1]}, 6));
report('D6 (ring)',        group_closure({[2 3 4 5 6 1],[6 5 4 3 2 1]}, 6));
Gih = icosahedral_group();
report('I_h (icosahedron)', Gih);
report('square 4x4',        square_lattice_group(4, 4));

%% before/after: a 3-dimensional I_h irrep
[ir, di] = dixon_irreps(Gih, 1, 1e-10);
[rir, ri] = realify_irreps(ir, Gih, 1e-9, 0);
a = find(di.dims == 3, 1);          % first 3-dim irrep
g = 2;                              % a non-identity element
fprintf('\n--- I_h irrep #%d (dim 3, type ''%s''), element g = %d ---\n', a, ri.types{a}, g);
fprintf('BEFORE (Dixon, complex basis):\n');   disp(round(ir{a}{g}  * 1e4) / 1e4);
fprintf('AFTER realify (exactly real, orthogonal):\n');  disp(round(rir{a}{g} * 1e4) / 1e4);
fprintf('isreal(after) = %d,  ||B B''-I|| = %.2e,  imag residual removed = %.2e\n', ...
        isreal(rir{a}{g}), norm(rir{a}{g}*rir{a}{g}' - eye(3)), ri.maximag(a));

%% C6: a real-type vs a complex-type irrep
Gc6 = group_closure({[2 3 4 5 6 1]}, 6);
[irc, ~] = dixon_irreps(Gc6, 1, 1e-10);
[ric, ci] = realify_irreps(irc, Gc6, 1e-9, 0);
ar = find(ci.fs == 1, 1, 'last');   % a real-type 1-dim irrep
ac = find(ci.fs == 0, 1);           % a complex-type 1-dim irrep
fprintf('\n--- C6 (cyclic) ---\n');
fprintf('real-type irrep #%d at g=2:  %s  (realified, real)\n', ar, num2str(ric{ar}{2}));
fprintf('complex-type irrep #%d at g=2: %s  (cannot be made real -> left complex)\n', ...
        ac, num2str(irc{ac}{2}));

% ---- local functions ----
function report(name, G)
    [ir, ~]  = dixon_irreps(G, 1, 1e-10);
    [~, ri]  = realify_irreps(ir, G, 1e-9, 0);
    fprintf('  %-18s |G|=%3d :  %2d real,  %2d complex,  %2d pseudoreal   (realified %d)\n', ...
            name, G.n, sum(ri.fs==1), sum(ri.fs==0), sum(ri.fs==-1), sum(ri.realified));
end

function G = icosahedral_group()
    m = polyhedron_model('icosahedron');
    [perms, ~] = polyhedron_symmetries(m.V);
    gens = {};
    for k = 1:numel(perms)
        if ~isequal(perms{k}, 1:m.N), gens = {perms{k}}; break; end
    end
    G = group_closure(gens, m.N);
    for k = 1:numel(perms)
        if G.n == numel(perms), break; end
        if ~ismember(perms{k}, G.elements, 'rows')
            gens{end+1} = perms{k}; %#ok<AGROW>
            G = group_closure(gens, m.N);
        end
    end
end
