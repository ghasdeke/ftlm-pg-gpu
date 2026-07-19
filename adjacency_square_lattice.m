function bonds = adjacency_square_lattice(Lx, Ly)
%ADJACENCY_SQUARE_LATTICE  Nearest-neighbour bonds of an Lx x Ly torus.
%
%   BONDS = ADJACENCY_SQUARE_LATTICE(LX, LY) returns the nearest-neighbour
%   bonds of a two-dimensional square lattice of LX * LY spin centres with
%   PERIODIC boundary conditions (a torus) in the vertex labelling used by
%   SQUARE_LATTICE_TRANSLATION_GROUP:
%
%       site (x, y), x = 0..LX-1, y = 0..LY-1, has 1-based index
%           i = 1 + x + LX * y.
%
%   Each row [a, b] is a bond with 1 <= a < b <= LX*LY. Every interior
%   site couples to its +x and +y neighbour; the wrap-around makes the
%   lattice a torus, which is exactly the geometry whose symmetry group is
%   the translation group C_LX x C_LY (see SQUARE_LATTICE_TRANSLATION_GROUP).
%
%   For LX, LY >= 3 there are exactly 2 * LX * LY distinct bonds (every
%   site has four neighbours, each bond shared by two sites). For a
%   degenerate axis (LX == 1, LY == 1, or == 2) the +dir and -dir
%   neighbours coincide; self-bonds (a == b, only at L == 1) are dropped
%   and duplicate bonds (at L == 2, where +x and -x hit the same site) are
%   de-duplicated, so a 1 x LY lattice reduces to a single LY-ring and a
%   2 x LY lattice keeps one rung per row.
%
%   The result is sorted ascending so downstream code can use binary
%   lookups if needed.
%
%   See also SQUARE_LATTICE_TRANSLATION_GROUP, ADJACENCY_ICOSAHEDRON_IH,
%            ADJACENCY_ICOSIDODECAHEDRON_IH.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    assert(isscalar(Lx) && Lx >= 1 && Lx == round(Lx), 'Lx must be a positive integer');
    assert(isscalar(Ly) && Ly >= 1 && Ly == round(Ly), 'Ly must be a positive integer');

    idx = @(x, y) 1 + mod(x, Lx) + Lx * mod(y, Ly);   % 0-based (x,y) -> 1-based index

    raw = zeros(2 * Lx * Ly, 2);
    p   = 0;
    for y = 0 : Ly - 1
        for x = 0 : Lx - 1
            i  = idx(x, y);
            ix = idx(x + 1, y);     % +x neighbour (wraps)
            iy = idx(x, y + 1);     % +y neighbour (wraps)
            p = p + 1;  raw(p, :) = [min(i, ix), max(i, ix)];
            p = p + 1;  raw(p, :) = [min(i, iy), max(i, iy)];
        end
    end

    %% Drop self-bonds (only at L == 1 along an axis) and duplicates
    %  (only at L == 2, where the +dir and -dir neighbour coincide).
    raw   = raw(raw(:, 1) ~= raw(:, 2), :);
    bonds = unique(raw, 'rows');     % sorts ascending as a side effect

    if Lx >= 3 && Ly >= 3
        assert(size(bonds, 1) == 2 * Lx * Ly, ...
            'adjacency_square_lattice: expected %d bonds, got %d', ...
            2 * Lx * Ly, size(bonds, 1));
    end
end
