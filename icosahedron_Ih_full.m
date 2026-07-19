function group = icosahedron_Ih_full()
%ICOSAHEDRON_IH_FULL  Full I_h group on the icosahedron with all irreps.
%
%   GROUP = ICOSAHEDRON_IH_FULL() returns a struct describing the full
%   icosahedral symmetry group I_h (order 120) together with all ten of
%   its irreducible representations. The construction follows Altmann &
%   Herzig, "Point-Group Theory Tables", p. 644 ff and p. 655 for the
%   explicit irrep generator matrices.
%
%   The struct contains:
%       N            = 12, number of vertices
%       order        = 120, group order
%       perms        [120 x 12] int8: row k is the 1-indexed permutation
%                    induced on the 12 vertices by group element k.
%                    perms(k, i) = j means vertex i is sent to vertex j.
%       perm_mats    [12 x 12 x 120] double: corresponding permutation
%                    matrices (P(a, b) = 1 iff b -> a). Kept for
%                    code paths that prefer the matrix-product view.
%       inv          [120 x 1] int32: index of the inverse of element k.
%       mul          [120 x 120] int32: mul(a, b) = index of G(a) * G(b),
%                    composition convention (gh)(v) = g(h(v)).
%       class_idx    [120 x 1] int32: conjugacy class index per element (1..10).
%       class_size   [10 x 1] int32: size of each conjugacy class.
%       class_label  {10 x 1} cell: 'E', '12C5', '12C5^2', '20C3',
%                    '15C2', 'i', '12S10', '12S10^3', '20S6', '15sigma'.
%       det          [120 x 1] int8: +1 for proper rotations, -1 for
%                    improper. Equivalent to the A_u character.
%       Ag, Au       [120 x 1] complex: 1D irrep characters.
%       T1g, T1u     [3 x 3 x 120] complex: T1g, T1u irrep matrices.
%       T2g, T2u     [3 x 3 x 120] complex: T2g, T2u irrep matrices.
%       Fg, Fu       [4 x 4 x 120] complex: F_g, F_u (4D, Altmann-Herzig
%                    notation; this is the irrep also called G_g, G_u in
%                    the Mulliken convention used by most international
%                    chemistry tables).
%       Hg, Hu       [5 x 5 x 120] complex: H_g, H_u irrep matrices.
%
%   Vertex labelling: convention from icosahedron_ih_representation_-
%   matrices_and_irreps.m. C_5 fixes vertices 1 and 7 (antipodal pair);
%   the inversion pairs are
%       (1, 7), (2, 8), (3, 9), (4, 10), (5, 6), (11, 12).
%   This differs from the labelling used by release/ftlm_observables.m
%   adjacency_icosahedron; for the I_h-aware pipeline in mit_pg we use
%   the labelling here. The corresponding 30 nearest-neighbour bonds are
%   produced by adjacency_icosahedron_Ih.
%
%   The group + irreps are built by closure over two rotation generators
%   (C_5 and C_3), then I_h is formed as I x {e, i} with the inversion
%   element. The structure is cached internally so subsequent calls
%   return the same struct without recomputation.
%
%   This function supersedes icosahedron_Ih_group (Phase alpha "1D-only"
%   variant). For the full PG-FTLM pipeline on the icosahedron the
%   higher-dimensional irreps from this function are required.
%
%   Source of the generator irrep matrices:
%       S. L. Altmann, P. Herzig, "Point-Group Theory Tables",
%       Clarendon Press, Oxford (1994), p. 655.

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

    %% Generator permutations (12 x 12 matrices, P(a, b) = 1 iff b -> a)
    C5 = cycles_to_perm({[2, 10, 9, 5, 12], [3, 6, 11, 8, 4]});
    C3 = cycles_to_perm({[1, 12, 2], [7, 11, 8], [4, 6, 9], [3, 10, 5]});
    Ci = cycles_to_perm({[1, 7], [2, 8], [3, 9], [4, 10], [5, 6], [11, 12]});

    %% Irrep generator matrices (Altmann & Herzig p. 655)
    phi  = (sqrt(5) + 1) / 2;     % g_p
    phi_inv = (sqrt(5) - 1) / 2;  % g_m = 1/phi
    t   = sqrt(5);
    lam = exp(1i * atan(sqrt(5/3)));
    om  = exp(2i * pi / 3);

    % C5 irrep generator matrices
    A_C5_gen  = 1;
    T1_C5_gen = 0.5 * [ phi_inv,    -1i,         -1i * phi
                       -1i,         phi,         -phi_inv
                       -1i * phi,   -phi_inv,     1       ];
    T2_C5_gen = 0.5 * [-phi,        -1i,          1i * phi_inv
                       -1i,         -phi_inv,     phi
                        1i * phi_inv, phi,        1          ];
    F_C5_gen = 0.25 * [-1,          -t,          -1i * t,    -1i * t
                       -t,          -1,           3i,        -1i
                       -1i * t,      3i,          1,          1
                       -1i * t,     -1i,          1,         -3   ];
    H_C5_gen = 0.5 * [ 0,                       lam^2 * conj(om),    -lam,                    -1i * lam * conj(om),    -1i * lam * om
                       conj(lam)^2 * om,        0,                   -conj(lam),              -1i * conj(lam) * om,    -1i * conj(lam) * conj(om)
                      -conj(lam),               -lam,                 1,                       0,                       1i
                      -1i * conj(lam) * om,    -1i * lam * conj(om), 0,                       -1,                      -1
                      -1i * conj(lam) * conj(om), -1i * lam * om,    1i,                      -1,                       0 ];

    % C3 irrep generator matrices
    A_C3_gen  = 1;
    T1_C3_gen = [ 0,   0,   -1i
                 -1i,  0,    0
                  0,  -1,    0 ];
    T2_C3_gen = T1_C3_gen;       % same matrix as T1, but T2 is a different irrep
    F_C3_gen  = [ 1,  0,    0,   0
                  0,  0,    0,  -1i
                  0, -1i,   0,   0
                  0,  0,   -1,   0 ];
    H_C3_gen  = [ om,         0,           0,    0,    0
                  0,          conj(om),    0,    0,    0
                  0,          0,           0,    0,   -1i
                  0,          0,          -1i,   0,    0
                  0,          0,           0,   -1,    0 ];

    %% Close under {C_5, C_3} -> rotation subgroup I (60 elements)
    [M_list, A_list, T1_list, T2_list, F_list, H_list] = close_I( ...
        C5, C3, A_C5_gen, A_C3_gen, T1_C5_gen, T1_C3_gen, ...
        T2_C5_gen, T2_C3_gen, F_C5_gen, F_C3_gen, H_C5_gen, H_C3_gen);
    assert(size(M_list, 3) == 60, 'icosahedron_Ih_full: |I| != 60');

    %% Form I_h = I x {e, i} (120 elements)
    perm_mats = zeros(12, 12, 120);
    Ag        = ones(120, 1);
    Au        = zeros(120, 1);
    T1g       = zeros(3, 3, 120);
    T1u       = zeros(3, 3, 120);
    T2g       = zeros(3, 3, 120);
    T2u       = zeros(3, 3, 120);
    Fg        = zeros(4, 4, 120);
    Fu        = zeros(4, 4, 120);
    Hg        = zeros(5, 5, 120);
    Hu        = zeros(5, 5, 120);
    for m = 1 : 60
        perm_mats(:, :, 2*m - 1) = M_list(:, :, m);
        perm_mats(:, :, 2*m)     = M_list(:, :, m) * Ci;

        Au(2*m - 1) =  1;
        Au(2*m)     = -1;

        T1g(:, :, 2*m - 1) =  T1_list(:, :, m);
        T1g(:, :, 2*m)     =  T1_list(:, :, m);
        T1u(:, :, 2*m - 1) =  T1_list(:, :, m);
        T1u(:, :, 2*m)     = -T1_list(:, :, m);

        T2g(:, :, 2*m - 1) =  T2_list(:, :, m);
        T2g(:, :, 2*m)     =  T2_list(:, :, m);
        T2u(:, :, 2*m - 1) =  T2_list(:, :, m);
        T2u(:, :, 2*m)     = -T2_list(:, :, m);

        Fg(:, :, 2*m - 1)  =  F_list(:, :, m);
        Fg(:, :, 2*m)      =  F_list(:, :, m);
        Fu(:, :, 2*m - 1)  =  F_list(:, :, m);
        Fu(:, :, 2*m)      = -F_list(:, :, m);

        Hg(:, :, 2*m - 1)  =  H_list(:, :, m);
        Hg(:, :, 2*m)      =  H_list(:, :, m);
        Hu(:, :, 2*m - 1)  =  H_list(:, :, m);
        Hu(:, :, 2*m)      = -H_list(:, :, m);
    end

    %% Convert permutation matrices to permutation arrays
    %  perms(k, i) = j means vertex i is sent to vertex j by element k.
    perms = zeros(120, 12, 'int8');
    for k = 1 : 120
        P = perm_mats(:, :, k);
        for col = 1 : 12
            row = find(P(:, col), 1);
            perms(k, col) = int8(row);
        end
    end

    %% Multiplication table
    %  Build a hash for each permutation matrix and find the index of
    %  every product. We use a perm-array key because perm_mats are
    %  large; the array form is integer and trivially hashable.
    key_of = @(p) sprintf('%d_', int32(p));
    key2idx = containers.Map();
    for k = 1 : 120
        key2idx(key_of(perms(k, :))) = k;
    end

    mul = zeros(120, 120, 'int32');
    for a = 1 : 120
        pa = double(perms(a, :));
        for b = 1 : 120
            pb = double(perms(b, :));
            c  = pa(pb);   % composition (gh)(v) = g(h(v)) in 1-indexed form
            mul(a, b) = key2idx(key_of(c));
        end
    end

    %% Inverses
    inv_idx = zeros(120, 1, 'int32');
    for k = 1 : 120
        pk = double(perms(k, :));
        pinv = zeros(1, 12);
        pinv(pk) = 1 : 12;
        inv_idx(k) = key2idx(key_of(pinv));
    end

    e_idx = key2idx(key_of(1 : 12));
    assert(mul(1, inv_idx(1)) >= 1, 'inversion table sanity');
    assert(all(arrayfun(@(k) mul(k, inv_idx(k)) == e_idx, 1 : 120)), ...
        'g * g^{-1} != e somewhere');

    %% Conjugacy classes
    assigned = zeros(120, 1, 'int32');
    classes  = {};
    for i = 1 : 120
        if assigned(i) > 0, continue; end
        cls = false(120, 1);
        for h = 1 : 120
            x = mul(h, mul(i, inv_idx(h)));
            cls(x) = true;
        end
        classes{end+1, 1} = find(cls); %#ok<AGROW>
        assigned(classes{end}) = numel(classes);
    end
    assert(numel(classes) == 10, 'expected 10 conjugacy classes');

    %% Determinant per element (+1 = rotation, -1 = improper)
    %  Equivalent to A_u character.
    det_vec = int8(Au);

    %% Class signatures from one representative
    class_size  = cellfun(@numel, classes);
    class_label = cell(numel(classes), 1);
    for c = 1 : numel(classes)
        rep_idx = classes{c}(1);
        d = double(det_vec(rep_idx));
        % rotation angle from T1g matrix trace
        tr = real(trace(T1g(:, :, rep_idx)));
        if d > 0
            cos_th = (tr - 1) / 2;
        else
            cos_th = (tr + 1) / 2;
        end
        cos_th = max(min(cos_th, 1), -1);
        theta_deg = round(acos(cos_th) * 180 / pi);
        class_label{c} = classify_signature(d, theta_deg);
    end

    %% Pack result
    group.N           = 12;
    group.order       = 120;
    group.perms       = perms;
    group.perm_mats   = perm_mats;
    group.inv         = inv_idx;
    group.mul         = mul;

    % Identity index (1..120). The closure starts from {C_5, C_3}, so
    % perms(1, :) is C_5, NOT the identity. We must search.
    group.identity    = int32(0);
    id_row            = int8(1 : 12);
    for k = 1 : 120
        if isequal(perms(k, :), id_row)
            group.identity = int32(k);
            break;
        end
    end
    assert(group.identity > 0, 'icosahedron_Ih_full: identity element not found');
    group.class_idx   = assigned;
    group.class_size  = int32(class_size);
    group.class_label = class_label;
    group.det         = det_vec;
    group.Ag          = Ag;
    group.Au          = Au;
    group.T1g = T1g; group.T1u = T1u;
    group.T2g = T2g; group.T2u = T2u;
    group.Fg  = Fg;  group.Fu  = Fu;
    group.Hg  = Hg;  group.Hu  = Hu;

    cached = group;
end

% ----------------------------------------------------------------
function P = cycles_to_perm(cyc_list)
    n = 12;
    P = zeros(n, n);
    fixed = true(1, n);
    for c = 1 : numel(cyc_list)
        cy = cyc_list{c};
        for k = 1 : numel(cy)
            if k < numel(cy), nxt = k + 1; else, nxt = 1; end
            a = cy(nxt);
            b = cy(k);
            P(a, b) = 1;
            fixed(b) = false;
        end
    end
    for f = find(fixed), P(f, f) = 1; end
end

function [M, A_, T1, T2, F, H] = close_I( ...
    C5, C3, A_C5_g, A_C3_g, T1_C5_g, T1_C3_g, T2_C5_g, T2_C3_g, ...
    F_C5_g, F_C3_g, H_C5_g, H_C3_g)
% BFS closure under {C5, C3}. Stores permutation matrices in M (12x12xN)
% and the corresponding irrep matrices in parallel arrays.
    M  = cat(3, C5, C3);
    A_ = [A_C5_g; A_C3_g];
    T1 = cat(3, T1_C5_g, T1_C3_g);
    T2 = cat(3, T2_C5_g, T2_C3_g);
    F  = cat(3, F_C5_g,  F_C3_g);
    H  = cat(3, H_C5_g,  H_C3_g);

    gens_M  = {C5, C3};
    gens_A  = [A_C5_g; A_C3_g];
    gens_T1 = {T1_C5_g, T1_C3_g};
    gens_T2 = {T2_C5_g, T2_C3_g};
    gens_F  = {F_C5_g,  F_C3_g};
    gens_H  = {H_C5_g,  H_C3_g};

    key = @(P) sprintf('%d_', int32(P(:)));
    seen = containers.Map();
    for k = 1 : size(M, 3)
        seen(key(M(:, :, k))) = k;
    end

    changed = true;
    while changed
        changed = false;
        for i = 1 : size(M, 3)
            for gi = 1 : 2
                N = M(:, :, i) * gens_M{gi};
                k = key(N);
                if ~isKey(seen, k)
                    seen(k) = size(M, 3) + 1;
                    M  = cat(3, M, N);
                    A_ = [A_; A_(i) * gens_A(gi)];                     %#ok<AGROW>
                    T1 = cat(3, T1, T1(:, :, i) * gens_T1{gi});
                    T2 = cat(3, T2, T2(:, :, i) * gens_T2{gi});
                    F  = cat(3, F,  F(:, :, i)  * gens_F{gi});
                    H  = cat(3, H,  H(:, :, i)  * gens_H{gi});
                    changed = true;
                end
            end
        end
    end
end

function lab = classify_signature(d, theta_deg)
    if d > 0
        switch theta_deg
            case 0,   lab = 'E';
            case 72,  lab = '12C5';
            case 120, lab = '20C3';
            case 144, lab = '12C5^2';
            case 180, lab = '15C2';
            otherwise, lab = sprintf('proper-%d', theta_deg);
        end
    else
        switch theta_deg
            case 0,   lab = '15sigma';
            case 36,  lab = '12S10^3';
            case 60,  lab = '20S6';
            case 108, lab = '12S10';
            case 180, lab = 'i';
            otherwise, lab = sprintf('improper-%d', theta_deg);
        end
    end
end
