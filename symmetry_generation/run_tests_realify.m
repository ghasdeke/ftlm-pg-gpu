%% run_tests_realify.m -- Frobenius-Schur classification + realification tests
%
% Applies realify_irreps to irreps from several different groups and checks:
%   - the Frobenius-Schur summary (real / complex / pseudoreal counts),
%   - that every real-type irrep becomes EXACTLY real (isreal == true),
%   - that realified matrices are still a unitary (orthogonal) homomorphism
%     with unchanged characters,
%   - that complex-type irreps are correctly left complex (not realified).
% The cyclic group C6 is included precisely because it HAS complex irreps.

clear; clc;
tol = 1e-9;
names = {};  oks = [];

% ---- build a variety of groups ----
cases = {};

Gc6 = group_closure({[2 3 4 5 6 1]}, 6);
cases{end+1} = struct('name','C6 (cyclic)', 'G', Gc6);

Gd6 = group_closure({[2 3 4 5 6 1],[6 5 4 3 2 1]}, 6);
cases{end+1} = struct('name','D6 (ring)', 'G', Gd6);

m = polyhedron_model('icosahedron');
[perms,~] = polyhedron_symmetries(m.V);
gens = {};
for k=1:numel(perms), if ~isequal(perms{k},1:m.N), gens={perms{k}}; break; end, end
Gih = group_closure(gens, m.N);
for k=1:numel(perms)
    if Gih.n==numel(perms), break; end
    if ~ismember(perms{k},Gih.elements,'rows'), gens{end+1}=perms{k}; Gih=group_closure(gens,m.N); end
end
cases{end+1} = struct('name','I_h (icosahedron)', 'G', Gih);

Gsq = square_lattice_group(4, 4);
cases{end+1} = struct('name','square 4x4', 'G', Gsq);

% ---- run ----
for ci = 1:numel(cases)
    C = cases{ci};  G = C.G;  n = G.n;
    [ir, ~]  = dixon_irreps(G, 1, 1e-10);
    [rir, ri] = realify_irreps(ir, G, tol, 0);
    nirr = numel(ir);

    nreal   = sum(ri.fs == 1);
    ncplx   = sum(ri.fs == 0);
    npseudo = sum(ri.fs == -1);
    fprintf('\n%-18s |G|=%3d, %2d irreps  ->  FS: %d real(+1), %d complex(0), %d pseudoreal(-1)\n', ...
            C.name, n, nirr, nreal, ncplx, npseudo);
    if nreal > 0
        fprintf('    realified %d/%d real-type irreps; max imag residual removed = %.2e\n', ...
                sum(ri.realified), nreal, max(ri.maximag(ri.fs==1)));
    end

    % (1) every real-type irrep is realified and EXACTLY real
    realIdx = find(ri.fs == 1);
    exactlyreal = all(ri.realified(realIdx));
    for a = realIdx
        for g = 1:n
            if ~isreal(rir{a}{g}), exactlyreal = false; end
        end
    end
    names{end+1} = [C.name ': real-type made exactly real']; %#ok<*SAGROW>
    oks(end+1)   = exactlyreal;

    % (2) realified irreps still valid: orthogonal + homomorphism + same character
    uerr = 0; herr = 0; cerr = 0;
    for a = realIdx
        d = size(rir{a}{1}, 1);
        for g = 1:n
            uerr = max(uerr, max(abs(reshape(rir{a}{g}'*rir{a}{g} - eye(d), [], 1))));
            cerr = max(cerr, abs(trace(rir{a}{g}) - trace(ir{a}{g})));
        end
        for g = 1:n
            for h = 1:n
                df = rir{a}{G.multtab(g,h)} - rir{a}{g}*rir{a}{h};
                herr = max(herr, max(abs(df(:))));
            end
        end
    end
    names{end+1} = [C.name ': realified still orthogonal+homomorphism+char'];
    oks(end+1)   = (uerr<tol) && (herr<tol) && (cerr<tol);

    % (3) complex-type irreps correctly left complex
    if ncplx > 0
        cplxIdx = find(ri.fs == 0);
        leftcomplex = ~any(ri.realified(cplxIdx));
        stillcomplex = false;
        for a = cplxIdx
            for g = 1:n
                if ~isreal(ir{a}{g}), stillcomplex = true; break; end
            end
        end
        names{end+1} = [C.name ': complex-type correctly NOT realified'];
        oks(end+1)   = leftcomplex && stillcomplex;
    end
end

% ---- report ----
fprintf('\n==== REALIFICATION TEST RESULTS (tol = %.1e) ====\n', tol);
for k = 1:numel(names)
    if oks(k), st = 'PASS'; else, st = 'FAIL'; end
    fprintf('  [%s] %s\n', st, names{k});
end
fprintf('-------------------------------------------------------\n');
fprintf('%d passed, %d failed\n', sum(oks), sum(~oks));
if all(oks), fprintf('ALL TESTS PASSED\n'); else, fprintf('SOME TESTS FAILED\n'); end
