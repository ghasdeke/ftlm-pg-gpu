function [group, bonds] = cuboctahedron_Oh()
%CUBOCTAHEDRON_OH  12-site cuboctahedron + full octahedral group O_h (|G|=48).
%
%   [GROUP, BONDS] = CUBOCTAHEDRON_OH() returns the FTLM provider struct for
%   the cuboctahedron (the quasiregular Archimedean solid: 12 vertices, 24
%   edges, 8 triangles + 6 squares, every vertex 4-coordinated) and its edge
%   list. The 8 corner-sharing triangles make the antiferromagnetic
%   cuboctahedron a classic frustrated magnet; N = 12 is fully ED-checkable
%   (VALIDATE_CUBOCTAHEDRON12).
%
%   Vertices are the cube edge midpoints, coordinates = all permutations of
%   (+-1, +-1, 0). The full symmetry group O_h (order 48, the signed
%   coordinate permutations) is built from THREE generators via
%   GROUP_FROM_GENERATORS (Dimino closure):
%       C4z : (x,y,z) -> (-y,  x,  z)
%       C3  : (x,y,z) -> ( z,  x,  y)   (about the (1,1,1) cube diagonal)
%       i   : (x,y,z) -> (-x, -y, -z)   (inversion)
%
%   Irreps are NOT attached here (provider convention -- same as the
%   space-group providers): the driver attaches them generically via
%       group.irreps = irreps_from_group(group);
%       group.irreps = realify_irreps(group.irreps, group);
%   O_h is ambivalent (all Frobenius-Schur +1), so all 10 irreps
%   ({A1,A2,E,T1,T2} x {g,u}, dims {1,1,2,3,3}, sum d^2 = 48, max d = 3)
%   become real orthogonal -> the REAL FP32 GPU kernel path applies.
%
%   See also GROUP_FROM_GENERATORS, IRREPS_FROM_GROUP, REALIFY_IRREPS,
%            ASSERT_BONDS_GROUP_INVARIANT, VALIDATE_CUBOCTAHEDRON12,
%            DODECAHEDRON_IH_FULL.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    %% Vertices: all permutations of (+-1, +-1, 0), deterministic order.
    V = zeros(12, 3);  k = 0;
    for zpos = 1 : 3                        % which coordinate is zero
        rest = setdiff(1 : 3, zpos);
        for s1 = [1, -1]
            for s2 = [1, -1]
                v = zeros(1, 3);  v(rest(1)) = s1;  v(rest(2)) = s2;
                k = k + 1;  V(k, :) = v;
            end
        end
    end

    %% O_h from three generators (coordinate maps -> site permutations).
    R4 = [0 -1 0; 1 0 0; 0 0 1];            % C4 about z
    R3 = [0 0 1; 1 0 0; 0 1 0];             % C3 about (1,1,1)
    Ri = -eye(3);                            % inversion
    gens  = {perm_of(V, R4), perm_of(V, R3), perm_of(V, Ri)};
    group = group_from_generators(gens, 12, 'O_h');
    assert(group.order == 48, ...
        'cuboctahedron_Oh: closure gave |G| = %d, expected 48.', group.order);

    %% Edges: vertex pairs at squared distance 2 (edge length sqrt(2)).
    bonds = zeros(0, 2);
    for i = 1 : 12
        for j = i + 1 : 12
            if sum((V(i, :) - V(j, :)).^2) == 2
                bonds(end + 1, :) = [i, j];   %#ok<AGROW>
            end
        end
    end
    assert(size(bonds, 1) == 24, ...
        'cuboctahedron_Oh: %d edges, expected 24.', size(bonds, 1));
    deg = accumarray(bonds(:), 1, [12, 1]);
    assert(all(deg == 4), 'cuboctahedron_Oh: vertex degrees differ from 4.');

    % Invariance holds by construction (integer coordinates, exact matching);
    % assert it anyway -- the check is the same guard the user-facing
    % 'generators' path runs, and it is milliseconds.
    assert_bonds_group_invariant(bonds, group, 'cuboctahedron O_h');
end


% ----------------------------------------------------------------
function p = perm_of(V, R)
%PERM_OF  Site permutation induced by the orthogonal map R:
%   p(i) = j with v_j = R * v_i (exact integer coordinates), in the
%   project-wide convention perms(k,i) = j : site i -> j.
    W = V * R.';                             % row i of W = (R * v_i)'
    [tf, p] = ismember(W, V, 'rows');
    assert(all(tf), 'cuboctahedron_Oh: map does not preserve the vertex set.');
    p = p(:).';
end
