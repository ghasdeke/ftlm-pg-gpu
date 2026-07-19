function test_square_lattice_group()
%TEST_SQUARE_LATTICE_GROUP  Verify the C_Lx x C_Ly translation-group provider.
%
%   Checks, for several lattices, that SQUARE_LATTICE_TRANSLATION_GROUP and
%   ADJACENCY_SQUARE_LATTICE produce a self-consistent group + geometry that
%   plugs into the group-generic machinery (MIN_IMAGE_IH, APPLY_PERM_TO_STATE):
%
%     (1) structural: sizes, identity, every perms-row a permutation;
%     (2) group axioms: mul(a,b) matches the composed site permutation,
%         inv(k) inverts k;
%     (3) irreps: homomorphism chi_p(a)*chi_p(b) = chi_p(a*b), character
%         orthogonality X'*X = order*I, and sum of d^2 = order;
%     (4) geometry: the bond set is invariant under every group element;
%     (5) action: MIN_IMAGE_IH agrees bit-for-bit with a brute-force orbit
%         minimum over all group elements (APPLY_PERM_TO_STATE), and the
%         returned g_min actually maps each state to its representative;
%     (6) 1D consistency: a 1 x Ly lattice reproduces MIN_IMAGE_RING reps
%         (the 2D code subsumes the original C_N ring path).
%
%   Run:  test_square_lattice_group   (errors out on the first failed check)
%
%   See also SQUARE_LATTICE_TRANSLATION_GROUP, ADJACENCY_SQUARE_LATTICE,
%            MIN_IMAGE_IH, MIN_IMAGE_RING, APPLY_PERM_TO_STATE.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    rng(12345);
    tol = 1e-10;

    lattices = [3 3; 4 3; 6 6; 2 4; 1 6];   % incl. degenerate axes (2, 1)
    for li = 1 : size(lattices, 1)
        Lx = lattices(li, 1);  Ly = lattices(li, 2);
        fprintf('=== Lattice %d x %d (N=%d) ===\n', Lx, Ly, Lx * Ly);
        g     = square_lattice_translation_group(Lx, Ly);
        bonds = adjacency_square_lattice(Lx, Ly);

        check_structure(g, Lx, Ly);
        check_group_axioms(g);
        check_irreps(g, tol);
        check_bond_invariance(g, bonds);
        check_min_image(g, bonds);
        fprintf('  all checks passed.\n');
    end

    %% 1D consistency: 1 x Ly lattice vs the original ring min-image.
    check_ring_consistency(8);
    check_ring_consistency(6);

    fprintf('\nALL TESTS PASSED.\n');
end

% ----------------------------------------------------------------
function check(cond, msg, varargin)
    if ~cond
        error(['test_square_lattice_group: ' msg], varargin{:});
    end
end

% ----------------------------------------------------------------
function check_structure(g, Lx, Ly)
    N = Lx * Ly;
    check(g.N == N && g.order == N, 'N/order mismatch');
    check(isequal(size(g.perms), [N, N]), 'perms size');
    check(g.identity == 1, 'identity index must be 1');
    check(isequal(g.perms(1, :), 1 : N), 'element 1 not the identity perm');
    for k = 1 : g.order
        check(isequal(sort(g.perms(k, :)), 1 : N), 'perms row %d not a permutation', k);
    end
    % shifts <-> index round trip
    for k = 1 : g.order
        gx = g.shifts(k, 1);  gy = g.shifts(k, 2);
        check(k == 1 + gx + Lx * gy, 'shift/index round trip at k=%d', k);
    end
end

% ----------------------------------------------------------------
function check_group_axioms(g)
    n = g.order;
    % mul(a,b): composed permutation perms(a, perms(b,:)) == perms(mul(a,b),:)
    for a = 1 : n
        pa = g.perms(a, :);
        for b = 1 : n
            composed = pa(g.perms(b, :));               % (g_a g_b)(v) = g_a(g_b(v))
            check(isequal(g.perms(double(g.mul(a, b)), :), composed), ...
                'mul/perm inconsistency at (a=%d,b=%d)', a, b);
        end
    end
    % inverses
    for k = 1 : n
        check(g.mul(k, g.inv(k)) == g.identity, 'g*g^{-1} != e at k=%d', k);
    end
end

% ----------------------------------------------------------------
function check_irreps(g, tol)
    n = g.order;
    check(numel(g.irreps) == g.n_irreps && g.n_irreps == n, 'n_irreps');
    % sum of d^2 == order
    d2 = sum(arrayfun(@(s) s.d^2, g.irreps));
    check(d2 == n, 'sum d^2 = %d != order %d', d2, n);

    % Character matrix X(k,p) = chi_p(g_k); homomorphism + orthogonality.
    X = zeros(n, n);
    for p = 1 : n
        check(g.irreps(p).d == 1, 'irrep %d not 1-dimensional', p);
        M = squeeze(g.irreps(p).mats);                  % [order x 1]
        check(numel(M) == n, 'irrep %d mats length', p);
        X(:, p) = M(:);
        % homomorphism: M(a)*M(b) == M(mul(a,b)) for all a,b
        outer = M(:) * M(:).';                          % [n x n]
        Mmul  = M(double(g.mul));                       % [n x n], indexed by mul table
        check(max(abs(outer(:) - Mmul(:))) < tol, ...
            'irrep %d homomorphism violated (max err %.2e)', p, max(abs(outer(:) - Mmul(:))));
    end
    % character orthogonality: X' * X = order * I
    G = X' * X;
    check(max(abs(G(:) - n * reshape(eye(n), [], 1))) < tol * n, ...
        'character orthogonality violated (max err %.2e)', max(abs(G(:) - n * reshape(eye(n), [], 1))));
    % trivial irrep is index 1 (all ones)
    check(max(abs(X(:, 1) - 1)) < tol, 'irrep 1 is not the trivial rep');
end

% ----------------------------------------------------------------
function check_bond_invariance(g, bonds)
    % Applying any group element to the bond set yields the same set.
    base = sortrows(bonds);
    for k = 1 : g.order
        pk = g.perms(k, :);
        mapped = [pk(bonds(:, 1))', pk(bonds(:, 2))'];
        mapped = [min(mapped, [], 2), max(mapped, [], 2)];
        mapped = sortrows(mapped);
        check(isequal(mapped, base), 'bond set not invariant under element %d', k);
    end
end

% ----------------------------------------------------------------
function check_min_image(g, bonds) %#ok<INUSD>
    % Brute-force orbit minimum over all group elements must equal MIN_IMAGE_IH,
    % and g_min must map each state to its representative. s = 1/2 (d_loc = 2).
    s_val = 0.5;  d_loc = 2;  N = g.N;

    if N <= 12
        states = (0 : 2^N - 1)';                 % exhaustive for small N
    else
        n_test = 2000;                           % random sample for large N
        digits = randi([0, d_loc - 1], n_test, N);
        states = double(digits) * (d_loc .^ (0 : N - 1))';
    end
    states = int64(states);

    % brute force: min over all group images
    n  = numel(states);
    rep_brute = states;
    for k = 1 : g.order
        img = apply_perm_to_state(g.perms(k, :), states, d_loc, N);
        rep_brute = min(rep_brute, img);
    end

    [reps, g_min] = min_image_Ih(states, g, s_val);
    check(isequal(reps, rep_brute), 'min_image_Ih reps differ from brute force');

    % g_min maps state -> rep
    for k = 1 : g.order
        sel = (g_min == k);
        if ~any(sel), continue; end
        img = apply_perm_to_state(g.perms(k, :), states(sel), d_loc, N);
        check(isequal(int64(img), reps(sel)), 'g_min element %d does not map to rep', k);
    end
end

% ----------------------------------------------------------------
function check_ring_consistency(Ly)
    % A 1 x Ly lattice is a single Ly-ring; its orbit reps must equal
    % MIN_IMAGE_RING (the original C_N path), confirming the convention.
    d_loc = 2;  N = Ly;
    g = square_lattice_translation_group(1, Ly);

    states = int64((0 : 2^N - 1)');
    reps_2d = min_image_Ih(states, g, 0.5);

    reps_ring = zeros(size(states), 'int64');
    for i = 1 : numel(states)
        reps_ring(i) = min_image_ring(states(i), N, d_loc);
    end
    check(isequal(reps_2d, reps_ring), ...
        '1 x %d lattice reps differ from min_image_ring', Ly);
    fprintf('=== 1 x %d ring consistency: OK ===\n', Ly);
end
