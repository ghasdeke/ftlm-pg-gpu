function m = polyhedron_model(name)
%POLYHEDRON_MODEL  Vertices and nearest-neighbour bonds of a Platonic solid.
%
%   M = POLYHEDRON_MODEL(NAME) with NAME = 'icosahedron' (12 vertices,
%   5-regular) or 'dodecahedron' (20 vertices, 3-regular) returns a struct:
%     M.name     lower-case name
%     M.N        number of vertices
%     M.V        N x 3 vertex coordinates
%     M.bonds    P x 2 nearest-neighbour edge list (i < j)
%     M.edgelen  nearest-neighbour distance
%     M.D        N x N matrix of pairwise distances
%
%   Bonds are detected geometrically as the pairs at minimum distance, so the
%   same code serves both solids without hard-coded adjacency.

    phi = (1 + sqrt(5)) / 2;
    switch lower(name)
        case 'icosahedron'
            % cyclic permutations of (0, +-1, +-phi)
            V = zeros(0, 3);
            for r = 0:2
                p  = circshift([0 1 phi], r);
                nz = find(p ~= 0);
                for s1 = [-1 1]
                    for s2 = [-1 1]
                        v = p;  v(nz(1)) = v(nz(1)) * s1;  v(nz(2)) = v(nz(2)) * s2;
                        V(end+1, :) = v; %#ok<AGROW>
                    end
                end
            end
        case 'dodecahedron'
            iphi = 1 / phi;
            V = zeros(0, 3);
            for s1 = [-1 1]                       % cube (+-1,+-1,+-1)
                for s2 = [-1 1]
                    for s3 = [-1 1]
                        V(end+1, :) = [s1 s2 s3]; %#ok<AGROW>
                    end
                end
            end
            for r = 0:2                            % cyclic perms of (0,+-1/phi,+-phi)
                p  = circshift([0 iphi phi], r);
                nz = find(p ~= 0);
                for s1 = [-1 1]
                    for s2 = [-1 1]
                        v = p;  v(nz(1)) = v(nz(1)) * s1;  v(nz(2)) = v(nz(2)) * s2;
                        V(end+1, :) = v; %#ok<AGROW>
                    end
                end
            end
        otherwise
            error('polyhedron_model:name', 'Unknown solid "%s".', name);
    end

    n   = size(V, 1);
    D   = sqrt(sum((reshape(V, n, 1, 3) - reshape(V, 1, n, 3)).^2, 3));
    dd  = D + diag(inf(n, 1));
    el  = min(dd(:));
    tol = 1e-6 * max(abs(V(:)));
    [I, J] = find(triu(abs(D - el) < tol, 1));

    m = struct('name', lower(name), 'N', n, 'V', V, 'bonds', [I J], ...
               'edgelen', el, 'D', D);
end
