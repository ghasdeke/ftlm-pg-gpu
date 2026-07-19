%% run_tests.m -- verification suite for the symmetry toolbox
%
% Builds the D6 ring example (N = 6) and checks, against an explicit
% tolerance, the defining properties of the irreps, projectors and SALCs:
%   - irreps form a homomorphism and are unitary
%   - completeness  sum d_a^2 = |G|
%   - Schur orthogonality of characters and of matrix elements
%   - spin representation is a homomorphism
%   - isotypic projectors are idempotent and sum to the identity
%   - the SALC basis is unitary
%   - [P_a, H] = 0 and H is block diagonal in the SALC basis
%   - sum of SALC sector dimensions = 2^N
% Every check prints PASS / FAIL; a summary is printed at the end.

clear; clc;
tol = 1e-10;

% ---- build the example ----
N = 6;
T = [2 3 4 5 6 1];
R = [6 5 4 3 2 1];
G  = group_closure({T, R}, N);
[irreps, info] = dixon_irreps(G, 42, tol);
S  = salc_projectors(irreps, G, N, tol);
H  = heisenberg_ring(N);

n    = G.n;
nirr = info.nirr;
dims = info.dims;

names = {};  oks = [];

fprintf('Setup: |G| = %d, irreps = %d, dims = %s, 2^N = %d\n\n', ...
        n, nirr, mat2str(dims), 2^N);

%% 1) homomorphism of the irreps:  D(g*h) = D(g) D(h)
err = 0;
for a = 1:nirr
    for g = 1:n
        for h = 1:n
            gh = G.multtab(g, h);
            d  = irreps{a}{gh} - irreps{a}{g} * irreps{a}{h};
            err = max(err, max(abs(d(:))));
        end
    end
end
names{end+1} = 'Irrep homomorphism  D(g*h)=D(g)D(h)';   oks(end+1) = err < tol;

%% 2) unitarity of every irrep matrix
err = 0;
for a = 1:nirr
    for g = 1:n
        d = irreps{a}{g}' * irreps{a}{g} - eye(dims(a));
        err = max(err, max(abs(d(:))));
    end
end
names{end+1} = 'Irrep unitarity     D(g)''D(g)=I';       oks(end+1) = err < tol;

%% 3) completeness:  sum d_a^2 = |G|
names{end+1} = 'Completeness        sum d_a^2 = |G|';    oks(end+1) = (sum(dims.^2) == n);

%% 4a) character orthogonality:  (1/|G|) sum_g chi_a(g) conj(chi_b(g)) = d_ab
err = 0;
for a = 1:nirr
    for b = 1:nirr
        s   = sum(info.chars(a, :) .* conj(info.chars(b, :))) / n;
        err = max(err, abs(s - (a == b)));
    end
end
names{end+1} = 'Character orthogonality (Schur)';        oks(end+1) = err < tol;

%% 4b) matrix-element orthogonality (grand orthogonality theorem)
%      (1/|G|) sum_g D^a_ij(g) conj(D^b_kl(g)) = d_ab d_ik d_jl / d_a
err = 0;
for a = 1:nirr
    for b = 1:nirr
        da = dims(a);  db = dims(b);
        for i = 1:da
            for j = 1:da
                for k = 1:db
                    for l = 1:db
                        s = 0;
                        for g = 1:n
                            s = s + irreps{a}{g}(i, j) * conj(irreps{b}{g}(k, l));
                        end
                        s        = s / n;
                        expected = (a == b) * (i == k) * (j == l) / da;
                        err      = max(err, abs(s - expected));
                    end
                end
            end
        end
    end
end
names{end+1} = 'Matrix-element orthogonality (Schur)';   oks(end+1) = err < tol;

%% 5) spin representation is a homomorphism
err = 0;
for g = 1:n
    for h = 1:n
        gh = G.multtab(g, h);
        d  = S.Dspin{g} * S.Dspin{h} - S.Dspin{gh};
        err = max(err, max(max(abs(d))));
    end
end
names{end+1} = 'Spin rep homomorphism';                  oks(end+1) = full(err) < tol;

%% 6) isotypic projectors idempotent:  P_a^2 = P_a
err = 0;
for a = 1:nirr
    d   = S.Piso{a} * S.Piso{a} - S.Piso{a};
    err = max(err, max(max(abs(d))));
end
names{end+1} = 'Projector idempotency  P_a^2 = P_a';     oks(end+1) = full(err) < tol;

%% 7) projector completeness:  sum_a P_a = I
Psum = sparse(S.M, S.M);
for a = 1:nirr, Psum = Psum + S.Piso{a}; end
err  = max(max(abs(Psum - speye(S.M))));
names{end+1} = 'Projector completeness sum P_a = I';     oks(end+1) = full(err) < tol;

%% 8) SALC basis is unitary
err = max(max(abs(S.U' * S.U - eye(S.M))));
names{end+1} = 'SALC basis unitarity   U''U = I';         oks(end+1) = err < tol;

%% 9) projectors commute with H:  [P_a, H] = 0
err = 0;
for a = 1:nirr
    d   = S.Piso{a} * H - H * S.Piso{a};
    err = max(err, max(max(abs(d))));
end
names{end+1} = 'Symmetry  [P_a, H] = 0';                 oks(end+1) = full(err) < tol;

%% 10) H block diagonal in the SALC basis
Hs      = S.U' * H * S.U;
[P, Q]  = ndgrid(S.blockId, S.blockId);
offmask = (P ~= Q);
leak    = max([abs(Hs(offmask)); 0]);
names{end+1} = 'H block diagonal in SALC basis';         oks(end+1) = leak < tol;

%% 11) sum of SALC sector dimensions = 2^N
names{end+1} = 'SALC dimension count   sum = 2^N';       oks(end+1) = (sum(S.sectordim) == 2^N);

%% ---- report ----
fprintf('==== TEST RESULTS (tol = %.1e) ====\n', tol);
for k = 1:numel(names)
    if oks(k), st = 'PASS'; else, st = 'FAIL'; end
    fprintf('  [%s] %s\n', st, names{k});
end
fprintf('-----------------------------------------\n');
fprintf('%d passed, %d failed\n', sum(oks), sum(~oks));
if all(oks)
    fprintf('ALL TESTS PASSED\n');
else
    fprintf('SOME TESTS FAILED\n');
end
