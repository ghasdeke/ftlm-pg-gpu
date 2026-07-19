function group = icosahedron_Ih_group()
%ICOSAHEDRON_IH_GROUP  Construct the icosahedral group I_h on 12 vertices.
%
%   GROUP = ICOSAHEDRON_IH_GROUP() returns a struct describing the full
%   icosahedral symmetry group I_h (|G| = 120) acting as permutations
%   on the 12 vertices of the icosahedron, using the same vertex
%   labelling as release/ftlm_observables.m adjacency_icosahedron.
%
%   The struct contains:
%       group.N            = 12, number of vertices
%       group.order        = 120, group order
%       group.perms        = [120 x 12] int8, row k is the 1-indexed
%                            permutation of vertices induced by group
%                            element k. perms(k, i) = j means vertex i
%                            is sent to vertex j by element k.
%       group.inv          = [120 x 1] int32, inv(k) = index of the
%                            inverse of element k.
%       group.mul          = [120 x 120] int32, mul(a, b) = index of
%                            G[a] * G[b], using composition convention
%                            (gh)(v) = g(h(v)).
%       group.class_idx    = [120 x 1] int32, conjugacy class index per
%                            element (1..10).
%       group.class_size   = [10 x 1] int32, size of each conjugacy
%                            class.
%       group.class_label  = {10 x 1} cell of strings: 'E', '12C5',
%                            '20C3', 'i', '12C5^2', '15C2', '12S10',
%                            '20S6', '12S10^3', '15sigma'.
%       group.det          = [120 x 1] int8, +1 for proper rotations and
%                            -1 for improper rotations. Equivalently the
%                            character of the A_u irrep.
%
%   This is the Phase-alpha deliverable of Milestone G (full I_h
%   exploitation on the icosahedron). All higher-dimensional irreps are
%   built on top of this structure in later phases.
%
%   The group is constructed once on first call and cached via
%   persistent storage; subsequent calls return the same struct without
%   recomputation.

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

    %% Vertex coordinates (same labelling as the rest of the codebase).
    phi = (1 + sqrt(5)) / 2;
    V = [ 0,    1,    phi
          0,    1,   -phi
          0,   -1,    phi
          0,   -1,   -phi
          1,    phi,  0
          1,   -phi,  0
         -1,    phi,  0
         -1,   -phi,  0
          phi,  0,    1
          phi,  0,   -1
         -phi,  0,    1
         -phi,  0,   -1 ];

    %% Generators: C_5 through V(1), C_3 through face-center, inversion.
    ax5 = V(1, :) / norm(V(1, :));
    R_C5 = rodrigues(ax5, 2*pi/5);
    P_C5 = induced_perm(R_C5, V);

    face_center = (V(1, :) + V(3, :) + V(9, :)) / 3.0;
    ax3 = face_center / norm(face_center);
    R_C3 = rodrigues(ax3, 2*pi/3);
    P_C3 = induced_perm(R_C3, V);

    P_inv = induced_perm(-eye(3), V);

    %% Close the group by BFS over products with the generators.
    gens = [P_C5; P_C3; P_inv];
    perms_list = close_group(gens, 200);
    N_g = size(perms_list, 1);
    assert(N_g == 120, 'icosahedron_Ih_group: expected 120 elements, got %d.', N_g);

    %% Build a hashable index for permutations.
    key = @(p) sprintf('%d_', int32(p));
    key2idx = containers.Map();
    for k = 1 : N_g
        key2idx(key(perms_list(k, :))) = k;
    end

    %% Multiplication table.
    mul = zeros(N_g, N_g, 'int32');
    for a = 1 : N_g
        pa = perms_list(a, :);
        for b = 1 : N_g
            pb = perms_list(b, :);
            % Composition: (g*h)(v) = g(h(v)); for 1-indexed: c(v) = g(h(v))
            c = pa(pb);
            mul(a, b) = key2idx(key(c));
        end
    end

    %% Inverses.
    inv_idx = zeros(N_g, 1, 'int32');
    for k = 1 : N_g
        pk = perms_list(k, :);
        pk_inv = zeros(1, 12);
        pk_inv(pk) = 1 : 12;
        inv_idx(k) = key2idx(key(pk_inv));
    end

    e_idx = key2idx(key(1 : 12));
    assert(all(mul(sub2ind([N_g, N_g], (1:N_g)', inv_idx)) == e_idx), ...
        'icosahedron_Ih_group: g * g^{-1} != e somewhere.');

    %% Conjugacy classes.
    assigned = zeros(N_g, 1, 'int32');
    classes  = {};
    for i = 1 : N_g
        if assigned(i) > 0, continue; end
        % Conjugacy class of i
        C = false(N_g, 1);
        for h = 1 : N_g
            x = mul(h, mul(i, inv_idx(h)));
            C(x) = true;
        end
        idx = find(C);
        classes{end+1, 1} = idx; %#ok<AGROW>
        assigned(idx) = numel(classes);
    end
    assert(numel(classes) == 10, 'icosahedron_Ih_group: expected 10 classes, got %d.', numel(classes));

    %% Determinant per element (== A_u character == sign of permutation
    %% in terms of orientation). We compute by reconstructing the 3D
    %% rotation matrix from where three independent vertices go.
    det_vec = zeros(N_g, 1, 'int8');
    src = V([1, 3, 5], :);
    for k = 1 : N_g
        dst = V(perms_list(k, [1, 3, 5]), :);
        Rk = (dst.' / src.');
        d = det(Rk);
        if d > 0.5, det_vec(k) = 1;
        else,        det_vec(k) = -1; end
    end

    %% Class sizes and signature-based labelling.
    class_size = cellfun(@numel, classes);
    [class_size, perm_sort] = sort(class_size);
    classes = classes(perm_sort);

    % Recompute assigned with reordered classes.
    assigned = zeros(N_g, 1, 'int32');
    for c = 1 : numel(classes)
        assigned(classes{c}) = c;
    end

    % Identify class type by determinant + rotation angle of any rep.
    class_label = cell(numel(classes), 1);
    for c = 1 : numel(classes)
        rep = classes{c}(1);
        dst = V(perms_list(rep, [1, 3, 5]), :);
        Rk = (dst.' / src.');
        d = det(Rk);
        tr = trace(Rk);
        if d > 0.5
            cos_th = (tr - 1) / 2;
        else
            cos_th = (tr + 1) / 2;
        end
        cos_th = max(min(cos_th, 1), -1);
        theta_deg = round(acos(cos_th) * 180 / pi);
        if d > 0.5
            switch theta_deg
                case 0,   class_label{c} = 'E';
                case 60,  class_label{c} = '20S6';   % shouldn't happen for proper
                case 72,  class_label{c} = '12C5';
                case 120, class_label{c} = '20C3';
                case 144, class_label{c} = '12C5^2';
                case 180, class_label{c} = '15C2';
                otherwise, class_label{c} = sprintf('proper-%d', theta_deg);
            end
        else
            switch theta_deg
                case 0,   class_label{c} = '15sigma';
                case 36,  class_label{c} = '12S10^3';
                case 60,  class_label{c} = '20S6';
                case 108, class_label{c} = '12S10';
                case 180, class_label{c} = 'i';
                otherwise, class_label{c} = sprintf('improper-%d', theta_deg);
            end
        end
    end

    %% Pack and cache.
    group.N           = 12;
    group.order       = N_g;
    group.perms       = int8(perms_list);
    group.inv         = inv_idx;
    group.mul         = mul;
    group.class_idx   = assigned;
    group.class_size  = int32(class_size);
    group.class_label = class_label;
    group.det         = det_vec;
    group.vertices    = V;

    cached = group;
end

% ----------------------------------------------------------------
function R = rodrigues(axis, angle)
    axis = axis(:) / norm(axis);
    K = [ 0,        -axis(3),  axis(2)
          axis(3),   0,       -axis(1)
         -axis(2),   axis(1),  0      ];
    R = eye(3) + sin(angle) * K + (1 - cos(angle)) * (K * K);
end

function p = induced_perm(R, V)
    n = size(V, 1);
    p = zeros(1, n);
    for i = 1 : n
        Rv = (R * V(i, :)')';
        found = false;
        for j = 1 : n
            if norm(Rv - V(j, :)) < 1e-6
                p(i) = j;
                found = true;
                break;
            end
        end
        if ~found
            error('induced_perm: R * V_%d did not land on any vertex.', i);
        end
    end
end

function perms_list = close_group(gens, max_order)
    seen = containers.Map();
    key = @(p) sprintf('%d_', int32(p));
    id_perm = 1 : 12;
    seen(key(id_perm)) = id_perm;
    for r = 1 : size(gens, 1)
        seen(key(gens(r, :))) = gens(r, :);
    end
    changed = true;
    while changed
        changed = false;
        items = values(seen);
        for ii = 1 : length(items)
            a = items{ii};
            for r = 1 : size(gens, 1)
                b = gens(r, :);
                c = a(b);   % (a * b)(v) = a(b(v)) in 1-indexed perm composition
                k = key(c);
                if ~isKey(seen, k)
                    seen(k) = c;
                    changed = true;
                    if seen.Count > max_order
                        error('Group order exceeded %d.', max_order);
                    end
                end
            end
        end
    end
    items = values(seen);
    perms_list = zeros(length(items), 12);
    for ii = 1 : length(items)
        perms_list(ii, :) = items{ii};
    end
    perms_list = sortrows(perms_list);
end
