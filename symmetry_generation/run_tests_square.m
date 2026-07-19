%% run_tests_square.m -- verification for the A x B square-lattice space groups
%
% For several lattices (square and rectangular, periodic) this checks the group
% order, that #conjugacy classes = #irreps, completeness sum d^2 = |G|, the
% homomorphism property (on generators x whole group, which is sufficient),
% unitarity, character orthogonality, and -- for the small groups -- the full
% Schur matrix-element orthogonality.  Each check prints PASS / FAIL.

clear; clc;
tol = 1e-9;
examples = {[3 3], [4 4], [6 6], [3 4], [4 6]};
names = {};  oks = [];

for e = 1:numel(examples)
    A = examples{e}(1);  B = examples{e}(2);  tag = sprintf('%dx%d', A, B);
    [G, ~]   = square_lattice_group(A, B);
    [ir, di] = dixon_irreps(G, 1, 1e-10);
    n = G.n;  nirr = di.nirr;  dims = di.dims;
    expord = A*B * (8*(A==B) + 4*(A~=B));

    % generator indices in the element list
    gi = zeros(1, numel(G.gens));
    for k = 1:numel(G.gens)
        gi(k) = find(ismember(G.elements, G.gens{k}, 'rows'));
    end

    names{end+1} = [tag ': |G| = A*B*|P|'];          oks(end+1) = (n == expord);
    names{end+1} = [tag ': #classes == #irreps'];     oks(end+1) = (numel(G.classes) == nirr);
    names{end+1} = [tag ': completeness sum d^2=|G|']; oks(end+1) = (sum(dims.^2) == n);

    % homomorphism: D(s*h) = D(s)D(h) for generators s and all h  (=> all g)
    herr = 0;
    for a = 1:nirr
        for s = gi
            for h = 1:n
                d = ir{a}{G.multtab(s,h)} - ir{a}{s} * ir{a}{h};
                herr = max(herr, max(abs(d(:))));
            end
        end
    end
    names{end+1} = [tag ': homomorphism (gens x G)'];  oks(end+1) = (herr < tol);

    % unitarity
    uerr = 0;
    for a = 1:nirr
        for g = 1:n
            d = ir{a}{g}' * ir{a}{g} - eye(dims(a));
            uerr = max(uerr, max(abs(d(:))));
        end
    end
    names{end+1} = [tag ': unitarity'];               oks(end+1) = (uerr < tol);

    % character orthogonality
    cerr = 0;
    for a = 1:nirr
        for b = 1:nirr
            sab = sum(di.chars(a,:) .* conj(di.chars(b,:))) / n;
            cerr = max(cerr, abs(sab - (a==b)));
        end
    end
    names{end+1} = [tag ': character orthogonality']; oks(end+1) = (cerr < tol);

    % full Schur matrix-element orthogonality (small groups only)
    if n <= 72
        serr = 0;
        for a = 1:nirr
            for b = 1:nirr
                da = dims(a);  db = dims(b);
                for i = 1:da, for j = 1:da, for k = 1:db, for l = 1:db
                    ss = 0;
                    for g = 1:n, ss = ss + ir{a}{g}(i,j) * conj(ir{b}{g}(k,l)); end
                    serr = max(serr, abs(ss/n - (a==b)*(i==k)*(j==l)/da));
                end, end, end, end
            end
        end
        names{end+1} = [tag ': matrix-element Schur'];  oks(end+1) = (serr < tol);
    end
end

% ---- report ----
fprintf('==== SQUARE-LATTICE TEST RESULTS (tol = %.1e) ====\n', tol);
for k = 1:numel(names)
    if oks(k), st = 'PASS'; else, st = 'FAIL'; end
    fprintf('  [%s] %s\n', st, names{k});
end
fprintf('-------------------------------------------------\n');
fprintf('%d passed, %d failed\n', sum(oks), sum(~oks));
if all(oks), fprintf('ALL TESTS PASSED\n'); else, fprintf('SOME TESTS FAILED\n'); end
