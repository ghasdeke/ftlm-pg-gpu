function group = square_lattice_translation_group(Lx, Ly)
%SQUARE_LATTICE_TRANSLATION_GROUP  Translation group C_Lx x C_Ly of a torus.
%
%   GROUP = SQUARE_LATTICE_TRANSLATION_GROUP(LX, LY) returns a group struct
%   describing the lattice-translation symmetry of an LX x LY square lattice
%   with periodic boundary conditions. The group is the direct product of
%   two cyclic groups, C_LX x C_LY, an ABELIAN group of order LX*LY, whose
%   LX*LY irreducible representations are all one-dimensional and labelled
%   by a lattice momentum (px, py).
%
%   The struct mirrors the field contract of ICOSAHEDRON_IH_FULL so that the
%   same group-generic machinery (MIN_IMAGE_IH, ENUMERATE_M_ORBITS_IH, ...)
%   consumes it unchanged, with two differences: the irreps are delivered as
%   a GENERIC LIST (group.irreps) instead of the I_h-specific named fields
%   (Ag, T1g, ...), and every irrep has dimension 1.
%
%   Site labelling: site (x, y), x = 0..LX-1, y = 0..LY-1, has 1-based index
%       i = 1 + x + LX * y.
%   Group-element labelling: shift (gx, gy), gx = 0..LX-1, gy = 0..LY-1, has
%   1-based index
%       k = 1 + gx + LX * gy,
%   so element k = 1 is the identity (gx = gy = 0). The matching
%   nearest-neighbour bonds are produced by ADJACENCY_SQUARE_LATTICE.
%
%   Output struct fields:
%       N            = LX*LY, number of sites.
%       order        = LX*LY, group order.
%       Lx, Ly       the two lattice extents (convenience).
%       perms        [order x N] double: row k is the 1-indexed site
%                    permutation induced by the shift g_k. perms(k, i) = j
%                    means site i is sent to site j (same convention as
%                    ICOSAHEDRON_IH_FULL / APPLY_PERM_TO_STATE).
%       perm_mats    [N x N x order] double: P(:,:,k), P(a,b)=1 iff b -> a.
%       shifts       [order x 2] double: the (gx, gy) of each element.
%       inv          [order x 1] int32: index of the inverse of element k.
%       mul          [order x order] int32: mul(a,b) = index of g_a * g_b
%                    (composition (gh)(v) = g(h(v)); abelian, so = g_a+g_b).
%       identity     int32 index of the identity element (always 1).
%       class_idx    [order x 1] int32: conjugacy-class index. The group is
%                    abelian, so every element is its own class: class_idx=k.
%       class_size   [order x 1] int32: all ones.
%       det          [order x 1] int8: all +1 (translations are proper).
%       irreps       [1 x order] struct array, the GENERIC irrep interface:
%                       .name   char, e.g. 'k(2,3)'
%                       .d      irrep dimension (always 1 here)
%                       .label  [px, py] lattice momentum (0-based indices)
%                       .mats   [1 x 1 x order] complex: mats(1,1,k) =
%                               chi_p(g_k) = exp(2*pi*i*(px*gx/LX + py*gy/LY))
%                    Irrep p has 1-based index 1 + px + LX*py; irrep 1 is the
%                    trivial (zero-momentum) representation.
%       n_irreps     = order.
%       momenta      [order x 2] double: the (px, py) of each irrep.
%
%   Convention. The character is chi_p(g) = exp(+2*pi*i*(px*gx/LX+py*gy/LY)).
%   Because C_LX x C_LY is abelian this is a genuine homomorphism
%   (chi_p(a)*chi_p(b) = chi_p(a*b)) for either sign; the sign only fixes
%   which physical momentum a label denotes and is verified by the
%   character-orthogonality test in TEST_SQUARE_LATTICE_GROUP. The eventual
%   aggregated-spectrum check against the no-symmetry ED is what pins the
%   physics, exactly as for the I_h path.
%
%   See also ADJACENCY_SQUARE_LATTICE, ICOSAHEDRON_IH_FULL,
%            MIN_IMAGE_IH, APPLY_PERM_TO_STATE, TEST_SQUARE_LATTICE_GROUP.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    assert(isscalar(Lx) && Lx >= 1 && Lx == round(Lx), 'Lx must be a positive integer');
    assert(isscalar(Ly) && Ly >= 1 && Ly == round(Ly), 'Ly must be a positive integer');

    N     = Lx * Ly;
    order = Lx * Ly;

    %% Index helpers (0-based coordinates <-> 1-based linear index).
    site_idx  = @(x, y) 1 + mod(x, Lx) + Lx * mod(y, Ly);

    %% Group elements: shifts (gx, gy), index k = 1 + gx + Lx*gy.
    shifts = zeros(order, 2);
    for gy = 0 : Ly - 1
        for gx = 0 : Lx - 1
            k = 1 + gx + Lx * gy;
            shifts(k, :) = [gx, gy];
        end
    end

    %% Site permutations: shift (gx, gy) sends site (x, y) -> (x+gx, y+gy).
    %  perms(k, i) = j  with i the index of (x,y) and j of the shifted site.
    perms = zeros(order, N);
    for k = 1 : order
        gx = shifts(k, 1);  gy = shifts(k, 2);
        for y = 0 : Ly - 1
            for x = 0 : Lx - 1
                i = site_idx(x, y);
                perms(k, i) = site_idx(x + gx, y + gy);
            end
        end
    end

    %% Sanity: identity is element 1 and acts trivially; every row a perm.
    assert(isequal(perms(1, :), 1 : N), 'element 1 is not the identity');
    for k = 1 : order
        assert(isequal(sort(perms(k, :)), 1 : N), 'perms row %d is not a permutation', k);
    end

    %% Permutation matrices P(:,:,k), P(a,b)=1 iff b -> a (i.e. a = perms(k,b)).
    perm_mats = zeros(N, N, order);
    for k = 1 : order
        for b = 1 : N
            perm_mats(perms(k, b), b, k) = 1;
        end
    end

    %% Multiplication / inverse tables (abelian: add/negate the shifts mod L).
    shift_to_k = @(gx, gy) 1 + mod(gx, Lx) + Lx * mod(gy, Ly);
    mul = zeros(order, order, 'int32');
    for a = 1 : order
        ax = shifts(a, 1);  ay = shifts(a, 2);
        for b = 1 : order
            bx = shifts(b, 1);  by = shifts(b, 2);
            mul(a, b) = int32(shift_to_k(ax + bx, ay + by));
        end
    end
    inv_idx = zeros(order, 1, 'int32');
    for k = 1 : order
        inv_idx(k) = int32(shift_to_k(-shifts(k, 1), -shifts(k, 2)));
    end

    %% Irreps: 36 (= order) one-dimensional momentum representations.
    %  Irrep p = (px, py) has 1-based index 1 + px + Lx*py.
    momenta = zeros(order, 2);
    irreps  = struct('name', {}, 'd', {}, 'label', {}, 'mats', {});
    gx_all  = shifts(:, 1);    % [order x 1]
    gy_all  = shifts(:, 2);
    for py = 0 : Ly - 1
        for px = 0 : Lx - 1
            p = 1 + px + Lx * py;
            momenta(p, :) = [px, py];
            chi = exp(2i * pi * (px * gx_all / Lx + py * gy_all / Ly));  % [order x 1]
            irreps(p).name  = sprintf('k(%d,%d)', px, py);
            irreps(p).d     = 1;
            irreps(p).label = [px, py];
            irreps(p).mats  = reshape(chi, [1, 1, order]);
        end
    end

    %% Pack.
    group.N          = N;
    group.order      = order;
    group.Lx         = Lx;
    group.Ly         = Ly;
    group.perms      = perms;
    group.perm_mats  = perm_mats;
    group.shifts     = shifts;
    group.inv        = inv_idx;
    group.mul        = mul;
    group.identity   = int32(1);
    group.class_idx  = int32((1 : order)');     % abelian: each element its own class
    group.class_size = ones(order, 1, 'int32');
    group.det        = ones(order, 1, 'int8');  % translations are proper
    group.irreps     = irreps;
    group.n_irreps   = order;
    group.momenta    = momenta;
end
