function group = dodecahedron_Ih_full()
%DODECAHEDRON_IH_FULL  Full I_h group on the dodecahedron (N=20).
%
%   GROUP = DODECAHEDRON_IH_FULL() returns the same I_h group struct as
%   ICOSAHEDRON_IH_FULL but with the permutation representation acting on
%   the 20 vertices of the dodecahedron (instead of the 12 vertices of the
%   icosahedron).
%
%   The dodecahedron is the DUAL of the icosahedron: its 20 vertices are in
%   one-to-one correspondence with the 20 triangular faces of the
%   icosahedron, and the I_h action on the dodecahedron vertices is exactly
%   the I_h action on those faces. We use this to derive the 20-vertex
%   permutation representation directly from the 12-vertex one, without any
%   3D coordinate computation (mirroring ICOSIDODECAHEDRON_IH_FULL, which
%   derives its 30-vertex action from the 30 icosahedron edges):
%
%       For group element g with icosahedron-vertex permutation p_g, the
%       icosahedron face f = {i, j, k} is sent to {p_g(i), p_g(j), p_g(k)},
%       which we look up in the canonical 20-face list ICOSAHEDRON_TRIANGLES
%       to obtain the new dodecahedron-vertex index in 1..20.
%
%   All other group data (mul, inv, class indices, det, conjugacy class
%   labels) and the ten irreducible representations (Ag, Au, T1g, T1u, T2g,
%   T2u, Fg, Fu, Hg, Hu) are intrinsic to the abstract group I_h and are
%   reused VERBATIM from ICOSAHEDRON_IH_FULL.
%
%   Output struct fields (identical layout to ICOSAHEDRON_IH_FULL):
%       N            = 20
%       order        = 120
%       perms        [120 x 20] int8: row k is the 1-indexed permutation of
%                    the 20 dodecahedron vertices induced by group element k.
%       perm_mats    [20 x 20 x 120] double: permutation matrices
%                    (P(a, b) = 1 iff b -> a).
%       inv, mul, class_idx, class_size, class_label, det, identity
%                    copied unchanged from ICOSAHEDRON_IH_FULL.
%       Ag, Au, T1g, T1u, T2g, T2u, Fg, Fu, Hg, Hu
%                    copied unchanged from ICOSAHEDRON_IH_FULL.
%
%   Vertex labelling: 1..20 in the order produced by ICOSAHEDRON_TRIANGLES
%   (lexicographically sorted 3-cliques of the icosahedron). The matching
%   30 nearest-neighbour bonds are produced by ADJACENCY_DODECAHEDRON_IH.
%
%   See also ICOSAHEDRON_IH_FULL, ICOSIDODECAHEDRON_IH_FULL,
%            ICOSAHEDRON_TRIANGLES, ADJACENCY_DODECAHEDRON_IH.

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

    base = icosahedron_Ih_full();
    tri  = icosahedron_triangles();         % [20 x 3], sorted, canonical
    n_f  = size(tri, 1);
    assert(n_f == 20, 'dodecahedron_Ih_full: expected 20 icosahedron faces, got %d', n_f);

    %% Hash a (sorted) icosahedron face -> its dodecahedron-vertex index 1..20.
    face_key = @(t) sprintf('%d-%d-%d', t(1), t(2), t(3));   % t ascending
    key2face = containers.Map();
    for f = 1 : n_f
        key2face(face_key(tri(f, :))) = f;
    end

    %% Derive the 20-vertex permutation rep from the 12-vertex one.
    %  perms_new(k, f) = dodecahedron-vertex index of the face
    %  {p_k(i), p_k(j), p_k(k)} where {i,j,k} = tri(f, :), p_k = base.perms(k, :).
    perms_new = zeros(120, 20, 'int8');
    for k = 1 : 120
        pk = double(base.perms(k, :));
        for f = 1 : n_f
            img = sort(pk(tri(f, :)));           % image face, sorted to key form
            perms_new(k, f) = int8(key2face(face_key(img)));
        end
    end

    %% Sanity: the identity permutation maps every face to itself.
    assert(isequal(double(perms_new(base.identity, :)), 1 : 20), ...
        'dodecahedron_Ih_full: identity element does not act trivially on faces');

    %% Sanity: closure consistent with the inherited multiplication table,
    %  same convention everywhere (g h)(v) = g(h(v)).
    rng(42);
    for trial = 1 : 5
        a = randi(120); b = randi(120);
        pa = double(perms_new(a, :));
        pb = double(perms_new(b, :));
        composed = pa(pb);                       % (g h)(v) = g(h(v))
        c = base.mul(a, b);
        assert(isequal(double(perms_new(c, :)), composed), ...
            'dodecahedron_Ih_full: closure failed for (a=%d, b=%d)', a, b);
    end

    %% Build the [20 x 20 x 120] permutation matrices (P(row=a, col=b)=1 iff b -> a).
    perm_mats_new = zeros(20, 20, 120);
    for k = 1 : 120
        for col = 1 : 20
            row = double(perms_new(k, col));
            perm_mats_new(row, col, k) = 1;
        end
    end

    %% Pack: everything irrep / multiplication / class related comes verbatim
    %  from the icosahedron group; only the site action and N are replaced.
    group           = base;
    group.N         = 20;
    group.perms     = perms_new;
    group.perm_mats = perm_mats_new;

    cached = group;
end
