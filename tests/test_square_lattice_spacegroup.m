function test_square_lattice_spacegroup()
%TEST_SQUARE_LATTICE_SPACEGROUP  Verify the square-lattice space-group provider.
%
%   Checks SQUARE_LATTICE_SPACEGROUP on square (C_4v) and rectangular (C_2v)
%   tori: group order, permutation/closure/inverse axioms (Latin-square mul
%   table), that the translation subgroup C_Lx x C_Ly is contained, that the
%   site stabiliser is the point group, bond invariance under every element,
%   and bit-identical orbit minima vs a brute force over all elements
%   (feeding MIN_IMAGE_IH, which the FTLM pipeline uses).
%
%   Run:  test_square_lattice_spacegroup
%
%   See also SQUARE_LATTICE_SPACEGROUP, SQUARE_LATTICE_TRANSLATION_GROUP,
%            MIN_IMAGE_IH, APPLY_PERM_TO_STATE.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    rng(7);
    cases = [4 4; 6 6; 4 3];           % square (C_4v) + rectangular (C_2v)
    for ci = 1 : size(cases, 1)
        Lx = cases(ci, 1);  Ly = cases(ci, 2);
        g  = square_lattice_spacegroup(Lx, Ly);
        bonds = adjacency_square_lattice(Lx, Ly);
        N = Lx * Ly;
        pg_order = 4 + 4 * (Lx == Ly);
        fprintf('=== %dx%d  (%s, order %d) ===\n', Lx, Ly, g.point_group, g.order);

        check(g.order == N * pg_order, 'order %d != %d', g.order, N*pg_order);

        % every row a permutation
        for k = 1 : g.order
            check(isequal(sort(g.perms(k, :)), 1:N), 'row %d not a permutation', k);
        end

        % mul / perm consistency: perms(mul(a,b),:) == perms(a, perms(b,:))
        for a = 1 : g.order
            pa = g.perms(a, :);
            for b = 1 : g.order
                check(isequal(g.perms(double(g.mul(a,b)), :), pa(g.perms(b, :))), ...
                    'mul/perm mismatch (%d,%d)', a, b);
            end
        end

        % Latin-square: each row & column of mul is a permutation of 1..order
        for a = 1 : g.order
            check(isequal(sort(double(g.mul(a, :))), 1:g.order), 'mul row %d not Latin', a);
            check(isequal(sort(double(g.mul(:, a)))', 1:g.order), 'mul col %d not Latin', a);
        end

        % inverses
        for k = 1 : g.order
            check(g.mul(k, g.inv(k)) == g.identity, 'g*g^{-1}!=e at %d', k);
        end

        % translation subgroup contained
        gt = square_lattice_translation_group(Lx, Ly);
        sgkeys = containers.Map('KeyType','char','ValueType','logical');
        for k = 1 : g.order, sgkeys(sprintf('%d,', g.perms(k,:))) = true; end
        for k = 1 : gt.order
            check(isKey(sgkeys, sprintf('%d,', gt.perms(k,:))), ...
                'translation %d missing from space group', k);
        end

        % site stabiliser of the origin (site 1) has order = point group
        stab1 = sum(g.perms(:, 1) == 1);
        check(stab1 == pg_order, 'stab(site1)=%d != pg_order %d', stab1, pg_order);

        % bond invariance under every element
        base = sortrows(bonds);
        for k = 1 : g.order
            pk = g.perms(k, :);
            mp = [pk(bonds(:,1))', pk(bonds(:,2))'];
            mp = sortrows([min(mp,[],2), max(mp,[],2)]);
            check(isequal(mp, base), 'bond set not invariant under element %d', k);
        end

        % MIN_IMAGE_IH vs brute force (s=1/2)
        check_min_image(g, N);

        fprintf('  all checks passed.\n');
    end
    fprintf('\nALL TESTS PASSED.\n');
end

% ----------------------------------------------------------------
function check(cond, msg, varargin)
    if ~cond, error(['test_square_lattice_spacegroup: ' msg], varargin{:}); end
end

function check_min_image(g, N)
    d_loc = 2;  s_val = 0.5;
    if N <= 12
        states = (0 : 2^N - 1)';
    else
        states = double(randi([0 1], 1500, N)) * (d_loc .^ (0:N-1))';
    end
    states = int64(states);
    rep_brute = states;
    for k = 1 : g.order
        rep_brute = min(rep_brute, apply_perm_to_state(g.perms(k,:), states, d_loc, N));
    end
    [reps, g_min] = min_image_Ih(states, g, s_val);
    check(isequal(reps, rep_brute), 'min_image reps != brute force');
    for k = 1 : g.order
        sel = (g_min == k);
        if ~any(sel), continue; end
        img = apply_perm_to_state(g.perms(k,:), states(sel), d_loc, N);
        check(isequal(int64(img), reps(sel)), 'g_min %d does not map to rep', k);
    end
end
