function validate_generators_square4x4()
%VALIDATE_GENERATORS_SQUARE4X4  Verify the generic generators-only entry point.
%
%   Proves that the user-facing "supply only the permutation generators" path
%   (GROUP_FROM_GENERATORS + the 'generators' geometry case) reproduces the
%   hard-coded provider and the full ED on the 4x4 s=1/2 Heisenberg AFM under
%   the C_4v space group (order 128). Two layers:
%
%   (1) STRUCTURAL: the group closed from hand-written generators is the SAME
%       permutation group as SQUARE_LATTICE_SPACEGROUP(4,4) -- identical set of
%       site permutations (sortrows-equal), identical order, self-consistent
%       mul/inv/perm axioms, and the bond list invariant under every element.
%       Set-identity of the elements means the entire downstream pipeline is
%       bit-identical to the validated native path; the irrep extractor +
%       realifier are then sanity-checked (sum d^2 = |G|, unitarity,
%       homomorphism, character orthogonality).
%
%   (2) END-TO-END: the CPU driver run through geometry='generators'
%       (input_generators_square4x4_ED.m, exact ED of every block) reproduces
%       the independent full ED: sum-rule sum_w == 2^16 and C(T) to ~1e-8.
%
%   See also GROUP_FROM_GENERATORS, GROUP_CLOSURE, IRREPS_FROM_GROUP,
%            REALIFY_IRREPS, SQUARE_LATTICE_SPACEGROUP, VALIDATE_SQUARE_4X4_C4V.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    rng(11);
    Lx = 4; Ly = 4; N = Lx * Ly; s_val = 0.5; J = 1.0;

    %% (1) STRUCTURAL: generators -> group == native provider -----------------
    fprintf('\n>>> Structural check: group_from_generators vs square_lattice_spacegroup(4,4)\n');
    gens = c4v_generators(Lx, Ly);
    g    = group_from_generators(gens);              % infer N from generators
    gref = square_lattice_spacegroup(Lx, Ly);

    check(g.N == N,             'N = %d != %d', g.N, N);
    check(g.order == gref.order,'order %d != native %d', g.order, gref.order);
    check(g.order == 128,       'order %d != 128 (C_4v space group)', g.order);

    % same permutation group: identical set of elements (order-independent)
    check(isequal(sortrows(g.perms), sortrows(double(gref.perms))), ...
        'generator-built perms are not the same set as the native provider');

    % every row a permutation; mul/perm consistency; Latin square; inverses
    for k = 1 : g.order
        check(isequal(sort(g.perms(k, :)), 1:N), 'row %d not a permutation', k);
    end
    for a = 1 : g.order
        pa = g.perms(a, :);
        for b = 1 : g.order
            check(isequal(g.perms(double(g.mul(a,b)), :), pa(g.perms(b, :))), ...
                'mul/perm mismatch (%d,%d)', a, b);
        end
        check(isequal(sort(double(g.mul(a, :))), 1:g.order),  'mul row %d not Latin', a);
        check(isequal(sort(double(g.mul(:, a)))', 1:g.order), 'mul col %d not Latin', a);
        check(g.mul(a, g.inv(a)) == g.identity, 'g*g^{-1} != e at %d', a);
    end

    % conjugacy bookkeeping self-consistent
    check(numel(g.class_size) == g.n_class, 'n_class mismatch');
    check(sum(double(g.class_size)) == g.order, 'class sizes do not sum to |G|');
    for c = 1 : g.n_class
        check(sum(g.class_idx == c) == double(g.class_size(c)), ...
            'class_idx count != class_size for class %d', c);
    end

    % bonds geometry: invariant under every group element
    bonds = adjacency_square_lattice(Lx, Ly);
    base  = sortrows([min(bonds,[],2), max(bonds,[],2)]);
    for k = 1 : g.order
        pk = g.perms(k, :);
        mp = sortrows([min(pk(bonds(:,1))', pk(bonds(:,2))'), ...
                       max(pk(bonds(:,1))', pk(bonds(:,2))')]);
        check(isequal(mp, base), 'bond set not invariant under element %d', k);
    end
    fprintf('  group identical to native provider; axioms + bond invariance OK.\n');

    %% irrep extraction + realification on the generator-built group ----------
    irr  = irreps_from_group(g);
    irr  = realify_irreps(irr, g);
    dims = arrayfun(@(z) z.d, irr);
    n    = double(g.order);
    check(sum(dims.^2) == n, 'sum d^2 = %d != |G| = %d', sum(dims.^2), n);
    if isfield(g, 'n_class')
        check(numel(irr) == g.n_class, '#irreps %d != #classes %d', numel(irr), g.n_class);
    end
    mul = double(g.mul);
    homerr = 0; unierr = 0;
    for p = 1 : numel(irr)
        M = irr(p).mats; d = irr(p).d;
        for s = 1 : 40
            gx = randi(n);
            unierr = max(unierr, norm(M(:,:,gx)*M(:,:,gx)' - eye(d), 'fro'));
        end
        for s = 1 : 400
            a = randi(n); b = randi(n);
            homerr = max(homerr, norm(M(:,:,a)*M(:,:,b) - M(:,:,mul(a,b)), 'fro'));
        end
    end
    X = zeros(n, numel(irr));
    for p = 1 : numel(irr)
        for gx = 1 : n, X(gx,p) = trace(irr(p).mats(:,:,gx)); end
    end
    ortherr = max(abs(reshape((X'*X)/n - eye(numel(irr)), [], 1)));
    check(unierr  < 1e-9, 'irrep non-unitary (%.2e)', unierr);
    check(homerr  < 1e-9, 'irrep not a homomorphism (%.2e)', homerr);
    check(ortherr < 1e-8, 'character orthogonality (%.2e)', ortherr);
    fprintf('  irreps: dims=[%s], unit %.1e, hom %.1e, char-orth %.1e : OK\n', ...
        num2str(sort(dims)), unierr, homerr, ortherr);

    %% (2) END-TO-END: driver via 'generators' geometry vs full ED -----------
    fprintf('\n>>> End-to-end: ftlm_observables_pg_Ih(generators) vs full ED\n');
    r = ftlm_observables_pg_Ih('input_generators_square4x4_ED.m');
    T = r.T_range(:)';
    E_all = ed_full_heisenberg(bonds, N, s_val, J);
    C_ED  = compute_observables_pg(E_all, ones(numel(E_all),1), zeros(numel(E_all),1), T);

    nz       = C_ED > 1e-9 * max(C_ED);
    err_C    = max(abs(r.C_T(nz) - C_ED(nz)) ./ C_ED(nz));
    sumw_err = abs(r.sum_w - 2^N) / 2^N;
    fprintf('  sum-rule: sum_w = %.6g, expected %d, rel.err = %.2e\n', r.sum_w, 2^N, sumw_err);
    fprintf('  C_gen vs full ED: max rel.err = %.3e\n', err_C);
    check(sumw_err < 1e-10, 'sum-rule violated: %.2e', sumw_err);
    check(err_C    < 1e-8,  'generators spectrum != full ED: %.2e', err_C);

    fprintf('\nPASS: generators-only path reproduces the native provider and full ED.\n');
end

% ----------------------------------------------------------------
function gens = c4v_generators(Lx, Ly)
%C4V_GENERATORS  Hand-written generators of the LxL torus C_4v space group.
%   Site (x,y), index i = 1 + x + Lx*y; P(i) = image of site i (1-based).
    N    = Lx * Ly;
    idx  = 1 : N;
    cx   = mod(idx - 1, Lx);
    cy   = floor((idx - 1) / Lx);
    site = @(xp, yp) 1 + mod(xp, Lx) + Lx * mod(yp, Ly);
    gens = { site(cx + 1, cy), ...      % t_x
             site(cx, cy + 1), ...      % t_y
             site(-cy, cx), ...         % C_4: (x,y) -> (-y, x)
             site(-cx, cy) };           % mirror x -> -x
end

function check(cond, msg, varargin)
    if ~cond, error(['validate_generators_square4x4: ' msg], varargin{:}); end
end
