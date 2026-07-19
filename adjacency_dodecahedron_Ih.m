function bonds = adjacency_dodecahedron_Ih()
%ADJACENCY_DODECAHEDRON_IH  30 nearest-neighbour bonds of the dodecahedron.
%
%   BONDS = ADJACENCY_DODECAHEDRON_IH() returns the 30 nearest-neighbour
%   bonds of the dodecahedron in the vertex labelling used by
%   DODECAHEDRON_IH_FULL (vertex f = the f-th icosahedron triangular face in
%   ICOSAHEDRON_TRIANGLES). Each row [a, b] is a bond with 1 <= a < b <= 20.
%
%   Construction. The dodecahedron is the dual of the icosahedron: its 20
%   vertices are the 20 icosahedron faces, and two dodecahedron vertices are
%   joined by an edge iff the two icosahedron faces share an edge (i.e. share
%   exactly two of their three vertices). Each triangular face has 3 edges,
%   each shared with exactly one neighbouring face, so every dodecahedron
%   vertex has degree 3 and there are 20*3/2 = 30 bonds.
%
%   The result is sorted ascending.
%
%   See also ICOSAHEDRON_TRIANGLES, DODECAHEDRON_IH_FULL,
%            ADJACENCY_ICOSAHEDRON_IH.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    tri = icosahedron_triangles();          % [20 x 3] faces = dodec vertices
    n_f = size(tri, 1);

    %% Two faces are NN-adjacent iff they share an icosahedron edge
    %  (exactly two common vertices).
    bonds = zeros(0, 2);
    for a = 1 : n_f - 1
        for b = a + 1 : n_f
            if numel(intersect(tri(a, :), tri(b, :))) == 2
                bonds(end+1, :) = [a, b]; %#ok<AGROW>
            end
        end
    end

    bonds = sortrows(bonds);
    assert(size(bonds, 1) == 30, ...
        'adjacency_dodecahedron_Ih: expected 30 bonds, got %d', size(bonds, 1));
    assert(size(unique(bonds, 'rows'), 1) == 30, ...
        'adjacency_dodecahedron_Ih: duplicate bonds');

    %% The dodecahedron is 3-regular.
    deg = accumarray([bonds(:, 1); bonds(:, 2)], 1, [n_f, 1]);
    assert(all(deg == 3), 'adjacency_dodecahedron_Ih: graph is not 3-regular');
end
