function tri = icosahedron_triangles()
%ICOSAHEDRON_TRIANGLES  The 20 triangular faces of the icosahedron.
%
%   TRI = ICOSAHEDRON_TRIANGLES() returns a [20 x 3] integer array; row f
%   lists the three icosahedron vertices (1..12, ascending) of the f-th
%   triangular face, in the I_h vertex labelling of ICOSAHEDRON_IH_FULL.
%   The faces are enumerated as the 3-cliques of the icosahedron edge graph
%   in lexicographic order, so the ordering is canonical and reproducible.
%
%   These 20 faces are the 20 vertices of the dual DODECAHEDRON: both
%   DODECAHEDRON_IH_FULL (site permutations) and ADJACENCY_DODECAHEDRON_IH
%   (bonds) use this list, so they share a single, consistent vertex
%   labelling.
%
%   See also ADJACENCY_ICOSAHEDRON_IH, DODECAHEDRON_IH_FULL,
%            ADJACENCY_DODECAHEDRON_IH.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    ico_bonds = adjacency_icosahedron_Ih();    % [30 x 2], 1 <= i < j <= 12
    adj = false(12, 12);
    for e = 1 : size(ico_bonds, 1)
        i = ico_bonds(e, 1);  j = ico_bonds(e, 2);
        adj(i, j) = true;  adj(j, i) = true;
    end

    tri = zeros(0, 3);
    for i = 1 : 10
        for j = i + 1 : 11
            if ~adj(i, j), continue; end
            for k = j + 1 : 12
                if adj(i, k) && adj(j, k)
                    tri(end+1, :) = [i, j, k]; %#ok<AGROW>
                end
            end
        end
    end
    assert(size(tri, 1) == 20, ...
        'icosahedron_triangles: expected 20 faces, got %d', size(tri, 1));
end
