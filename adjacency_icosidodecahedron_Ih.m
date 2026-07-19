function bonds = adjacency_icosidodecahedron_Ih()
%ADJACENCY_ICOSIDODECAHEDRON_IH  60 nearest-neighbour bonds of the icosidodecahedron.
%
%   BONDS = ADJACENCY_ICOSIDODECAHEDRON_IH() returns the 60 nearest-
%   neighbour bonds of the icosidodecahedron in the vertex labelling
%   used by ICOSIDODECAHEDRON_IH_FULL (vertex k = midpoint of the k-th
%   icosahedron edge in ADJACENCY_ICOSAHEDRON_IH). Each row [a, b] is a
%   bond with 1 <= a < b <= 30.
%
%   Construction. The icosidodecahedron is the rectified icosahedron;
%   its faces are 20 triangles (one per icosahedron face) and 12
%   pentagons (one per icosahedron vertex). Each icosidodecahedron edge
%   belongs to exactly one triangle and one pentagon, so enumerating
%   the 20 triangles and taking their 3 edges each yields all 60 edges
%   without duplication.
%
%   Algorithm:
%     1. Build the icosahedron's vertex-vertex adjacency from
%        ADJACENCY_ICOSAHEDRON_IH.
%     2. Enumerate all 3-cliques (i, j, k) with i < j < k. Each is a
%        triangular face of the icosahedron (there are exactly 20).
%     3. For each triangle {i, j, k}, look up the icosidodecahedron-
%        vertex indices of the three icosahedron edges {i,j}, {i,k},
%        {j,k} and emit the three pairwise bonds.
%
%   The result is sorted ascending so that downstream code can use
%   binary lookups if needed.
%
%   See also ADJACENCY_ICOSAHEDRON_IH, ICOSIDODECAHEDRON_IH_FULL.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    ico_bonds = adjacency_icosahedron_Ih();
    n_e       = size(ico_bonds, 1);
    assert(n_e == 30, 'adjacency_icosidodecahedron_Ih: expected 30 icosahedron edges');

    %% 12 x 12 logical adjacency matrix of the icosahedron.
    adj = false(12, 12);
    for e = 1 : n_e
        i = ico_bonds(e, 1);  j = ico_bonds(e, 2);
        adj(i, j) = true;
        adj(j, i) = true;
    end

    %% Edge -> icosidodecahedron-vertex index lookup (same key convention
    %  as icosidodecahedron_Ih_full).
    edge_key = @(a, b) sprintf('%d-%d', min(a, b), max(a, b));
    key2edge = containers.Map();
    for e = 1 : n_e
        key2edge(edge_key(ico_bonds(e, 1), ico_bonds(e, 2))) = e;
    end

    %% Enumerate 3-cliques of the icosahedron graph. Each is a triangular
    %  face. There must be exactly 20.
    triangles = zeros(0, 3);
    for i = 1 : 10
        for j = i + 1 : 11
            if ~adj(i, j), continue; end
            for k = j + 1 : 12
                if adj(i, k) && adj(j, k)
                    triangles(end+1, :) = [i, j, k]; %#ok<AGROW>
                end
            end
        end
    end
    assert(size(triangles, 1) == 20, ...
        'adjacency_icosidodecahedron_Ih: expected 20 triangles, got %d', size(triangles, 1));

    %% Per triangle: emit the 3 bonds between its edge midpoints.
    %  Each icosidodecahedron edge belongs to exactly one icosahedron
    %  face (triangle), so no de-duplication is needed.
    bonds = zeros(60, 2);
    pos = 0;
    for t = 1 : size(triangles, 1)
        tri = triangles(t, :);
        v_ij = key2edge(edge_key(tri(1), tri(2)));
        v_ik = key2edge(edge_key(tri(1), tri(3)));
        v_jk = key2edge(edge_key(tri(2), tri(3)));
        pairs = [v_ij, v_ik; v_ij, v_jk; v_ik, v_jk];
        for p = 1 : 3
            a = min(pairs(p, :));
            b = max(pairs(p, :));
            pos = pos + 1;
            bonds(pos, :) = [a, b];
        end
    end

    bonds = sortrows(bonds);
    assert(size(bonds, 1) == 60, ...
        'adjacency_icosidodecahedron_Ih: expected 60 bonds, got %d', size(bonds, 1));
    assert(size(unique(bonds, 'rows'), 1) == 60, ...
        'adjacency_icosidodecahedron_Ih: duplicates in the 60-bond list');
end
