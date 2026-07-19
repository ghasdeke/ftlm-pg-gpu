function bonds = adjacency_icosahedron_Ih()
%ADJACENCY_ICOSAHEDRON_IH  30 nearest-neighbour bonds, I_h vertex labelling.
%
%   BONDS = ADJACENCY_ICOSAHEDRON_IH() returns the 30 nearest-neighbour
%   bonds of the icosahedron in the vertex labelling used by
%   ICOSAHEDRON_IH_FULL (compatible with the user's irrep generator
%   convention). Each row [i, j] is a bond with 1 <= i < j <= 12.
%
%   The bond set is obtained by taking the 5 known neighbours of vertex
%   1 (the C_5-orbit (2, 10, 9, 5, 12) around the C_5 axis through
%   vertex 1) and propagating with the 120 group elements. This avoids
%   any explicit coordinate-based geometry computation and stays inside
%   the chosen vertex labelling.
%
%   Note: this labelling differs from release/ftlm_observables.m
%   adjacency_icosahedron(). When the I_h pipeline is used, use this
%   function; when only S^z and matrix-free SpMV on the icosahedron
%   without I_h are used, the release labelling is fine.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    group = icosahedron_Ih_full();

    % Vertex 1's neighbours come straight from the C_5 cycle around it.
    v1_nbrs = [2, 5, 9, 10, 12];

    seen = containers.Map();
    for k = 1 : group.order
        p = double(group.perms(k, :));
        v1_image = p(1);
        for q = 1 : 5
            v2_image = p(v1_nbrs(q));
            a = min(v1_image, v2_image);
            b = max(v1_image, v2_image);
            seen(sprintf('%d-%d', a, b)) = [a, b];
        end
    end

    bonds = cell2mat(values(seen)');
    bonds = sortrows(bonds);
    assert(size(bonds, 1) == 30, ...
        'adjacency_icosahedron_Ih: expected 30 bonds, got %d', size(bonds, 1));
end
