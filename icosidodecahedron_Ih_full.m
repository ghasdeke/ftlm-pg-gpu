function group = icosidodecahedron_Ih_full()
%ICOSIDODECAHEDRON_IH_FULL  Full I_h group on the icosidodecahedron (N=30).
%
%   GROUP = ICOSIDODECAHEDRON_IH_FULL() returns the same I_h group struct
%   as ICOSAHEDRON_IH_FULL but with the permutation representation acting
%   on the 30 vertices of the icosidodecahedron (instead of the 12
%   vertices of the icosahedron).
%
%   The icosidodecahedron is the rectified icosahedron: its 30 vertices
%   are the midpoints of the 30 icosahedron edges. We exploit this to
%   derive the 30-vertex permutation representation directly from the
%   12-vertex one, without any 3D coordinate computation:
%
%       For group element g with icosahedron-vertex permutation p_g:
%       icosahedron edge e = {i, j}  is sent to the edge {p_g(i), p_g(j)},
%       which we look up in the canonical 30-edge list ADJACENCY_ICOSAHEDRON_IH
%       to obtain the new vertex index in 1..30.
%
%   All other group data (mul, inv, class indices, det, conjugacy class
%   labels) and the ten irreducible representations (Ag, Au, T1g, T1u,
%   T2g, T2u, Fg, Fu, Hg, Hu) are intrinsic to the abstract group I_h
%   and are reused VERBATIM from ICOSAHEDRON_IH_FULL.
%
%   Output struct fields (identical layout to ICOSAHEDRON_IH_FULL):
%       N            = 30
%       order        = 120
%       perms        [120 x 30] int8: row k is the 1-indexed permutation
%                    of the 30 icosidodecahedron vertices induced by
%                    group element k.
%       perm_mats    [30 x 30 x 120] double: corresponding permutation
%                    matrices (P(a, b) = 1 iff b -> a).
%       inv, mul, class_idx, class_size, class_label, det
%                    copied unchanged from ICOSAHEDRON_IH_FULL.
%       identity     int32 index of the identity element.
%       Ag, Au, T1g, T1u, T2g, T2u, Fg, Fu, Hg, Hu
%                    copied unchanged from ICOSAHEDRON_IH_FULL.
%
%   Vertex labelling: 1..30 in the order produced by
%   ADJACENCY_ICOSAHEDRON_IH (lexicographically sorted edges of the
%   icosahedron in the user's I_h vertex labelling). The matching bond
%   set on the icosidodecahedron is produced by
%   ADJACENCY_ICOSIDODECAHEDRON_IH and contains 60 bonds.
%
%   See also ICOSAHEDRON_IH_FULL, ADJACENCY_ICOSAHEDRON_IH,
%            ADJACENCY_ICOSIDODECAHEDRON_IH.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    persistent cached
    if ~isempty(cached)
        group = cached;
        return;
    end

    base   = icosahedron_Ih_full();
    bonds  = adjacency_icosahedron_Ih();    % [30 x 2], sorted, 1 <= i < j <= 12
    n_e    = size(bonds, 1);
    assert(n_e == 30, 'icosidodecahedron_Ih_full: expected 30 icosahedron edges, got %d', n_e);

    %% Hash icosahedron edges -> 1..30 by their sorted endpoints.
    edge_key = @(a, b) sprintf('%d-%d', min(a, b), max(a, b));
    key2edge = containers.Map();
    for e = 1 : n_e
        key2edge(edge_key(bonds(e, 1), bonds(e, 2))) = e;
    end

    %% Derive the 30-vertex permutation rep from the 12-vertex one.
    %  perms_new(k, e) = index in 1..30 of the edge {p_k(i_e), p_k(j_e)}
    %  where (i_e, j_e) = bonds(e, :) and p_k = base.perms(k, :).
    perms_new = zeros(120, 30, 'int8');
    for k = 1 : 120
        pk = double(base.perms(k, :));
        for e = 1 : n_e
            i = bonds(e, 1);  j = bonds(e, 2);
            a = pk(i);        b = pk(j);
            perms_new(k, e) = int8(key2edge(edge_key(a, b)));
        end
    end

    %% Sanity: the identity permutation maps every edge to itself.
    id_row = perms_new(base.identity, :);
    assert(isequal(double(id_row), 1 : 30), ...
        'icosidodecahedron_Ih_full: identity element does not act trivially on edges');

    %% Sanity: closure under the multiplication table inherited from base.
    %  For a random pair (a, b), perms_new(mul(a,b), :) must equal the
    %  composition (perms_new(a, :)) o (perms_new(b, :)) under the same
    %  convention used everywhere (g h)(v) = g(h(v)).
    rng(42);
    for trial = 1 : 5
        a = randi(120); b = randi(120);
        pa = double(perms_new(a, :));
        pb = double(perms_new(b, :));
        composed = pa(pb);                              % (g h)(v) = g(h(v))
        c = base.mul(a, b);
        assert(isequal(double(perms_new(c, :)), composed), ...
            'icosidodecahedron_Ih_full: closure failed for (a=%d, b=%d)', a, b);
    end

    %% Build the [30 x 30 x 120] permutation matrices (P(row=a, col=b)=1 iff b -> a).
    perm_mats_new = zeros(30, 30, 120);
    for k = 1 : 120
        for col = 1 : 30
            row = double(perms_new(k, col));
            perm_mats_new(row, col, k) = 1;
        end
    end

    %% Pack: everything irrep-related and the multiplication / class data
    %  comes verbatim from the icosahedron group; only the site action
    %  and N are replaced.
    group           = base;
    group.N         = 30;
    group.perms     = perms_new;
    group.perm_mats = perm_mats_new;

    cached = group;
end
