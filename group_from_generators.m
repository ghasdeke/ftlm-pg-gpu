function group = group_from_generators(gens, N, point_group_label)
%GROUP_FROM_GENERATORS  FTLM provider struct from user-supplied permutation
%                       generators (generic, any system not hard-coded).
%
%   GROUP = GROUP_FROM_GENERATORS(GENS) closes the permutation generators GENS
%   into the full symmetry group (Dimino, via GROUP_CLOSURE) and returns a
%   struct in EXACTLY the field layout the FTLM pipeline consumes -- the same
%   contract produced by SQUARE_LATTICE_SPACEGROUP / ICOSAHEDRON_IH_FULL, so the
%   driver's geometry switch, MIN_IMAGE_IH, ENUMERATE_M_ORBITS_IH and
%   COLLECT_CLT_ENTRIES_IH all consume it unchanged. This is the user-facing
%   entry point for treating a symmetry group / system that is not one of the
%   hard-coded geometries: supply only the generators, get a usable group.
%
%   GENS : the generators, as one of
%            - a cell array of 1xN permutation vectors,
%            - an MxN matrix whose rows are permutation vectors, or
%            - a single 1xN permutation vector.
%          A permutation P is a vector with P(i) = image of site i (1-based);
%          group multiplication is composition (A*B)(i) = A(B(i)). This is the
%          SAME convention as every FTLM provider (perms(k,i)=j: site i -> j).
%   N    : (optional) number of sites. Defaults to the length of the first
%          generator; if given, must match.
%   POINT_GROUP_LABEL : (optional) a label string stored in group.point_group
%          for printouts (default 'custom'). Purely cosmetic.
%
%   The irreps are NOT attached here (kept parallel to the other providers).
%   The driver attaches them generically right after selecting the group:
%       group.irreps = irreps_from_group(group);
%       group.irreps = realify_irreps(group.irreps, group);   % FS=+1 -> real
%   (IRREPS_FROM_GROUP needs only order/mul/inv/identity; REALIFY_IRREPS needs
%   only order/mul. Frobenius-Schur +1 irreps become real orthogonal; genuinely
%   complex irreps -- e.g. a pure cyclic C_n, n>2 -- are correctly left complex.)
%
%   The nearest-neighbour BOND list is geometry, NOT symmetry, and is a SEPARATE
%   input (see the 'generators' geometry case in the drivers): it must use the
%   same 1-based site labelling as the generators and be invariant under every
%   group element (the central correctness condition).
%
%   Output struct fields (identical layout to SQUARE_LATTICE_SPACEGROUP):
%       N, order        number of sites / group order |G|
%       point_group     the label string (cosmetic)
%       perms           [order x N] double site permutations, perms(k,i)=j
%       perm_mats       [N x N x order] permutation matrices (P(a,b)=1 iff b->a)
%       inv             [order x 1] int32 inverse index
%       mul             [order x order] int32 multiplication table (g(a)*g(b))
%       identity        int32 identity index
%       class_idx       [order x 1] int32 conjugacy-class index per element
%       class_size      [n_class x 1] int32 class sizes
%       n_class         number of conjugacy classes
%       gens            cell array of the (normalised) generators
%
%   See also GROUP_CLOSURE, IRREPS_FROM_GROUP, REALIFY_IRREPS,
%            SQUARE_LATTICE_SPACEGROUP, MIN_IMAGE_IH, ADDING_A_GEOMETRY (docs).

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    % --- infer / validate N -------------------------------------------------
    if iscell(gens)
        assert(~isempty(gens), 'group_from_generators: GENS cell is empty.');
        Ninf = numel(gens{1});
    elseif isvector(gens)
        Ninf = numel(gens);
    else
        Ninf = size(gens, 2);
    end
    if nargin < 2 || isempty(N), N = Ninf; end
    assert(N == Ninf, ...
        'group_from_generators: N=%d does not match generator length %d.', N, Ninf);
    if nargin < 3 || isempty(point_group_label), point_group_label = 'custom'; end

    % --- close the generators into the full permutation group ---------------
    G = group_closure(gens, N);              % vendored Dimino (root copy)
    order = G.n;
    assert(order <= 65535, ...
        ['group_from_generators: |G| = %d exceeds the 65535 cap (uint16 g-index ', ...
         'in the kernel/collect path).'], order);

    % --- per-element conjugacy-class index from the class cell --------------
    class_idx = zeros(order, 1, 'int32');
    for c = 1 : numel(G.classes)
        class_idx(G.classes{c}) = int32(c);
    end

    % --- permutation matrices P(a,b)=1 iff b -> a (== perms(k,b)) -----------
    perm_mats = zeros(N, N, order);
    for k = 1 : order
        for b = 1 : N
            perm_mats(G.elements(k, b), b, k) = 1;
        end
    end

    % --- pack into the FTLM provider contract (rename + cast) ---------------
    group.N           = N;
    group.order       = order;                       % G.n      -> order
    group.point_group = point_group_label;
    group.perms       = G.elements;                  % G.elements -> perms (same convention)
    group.perm_mats   = perm_mats;
    group.inv         = int32(G.inv(:));             % 1xn row  -> [order x 1] int32
    group.mul         = int32(G.multtab);            % G.multtab -> mul (int32)
    group.identity    = int32(G.id);                 % G.id     -> identity (=1)
    group.class_idx   = class_idx;
    group.class_size  = int32(G.classSizes(:));
    group.n_class     = numel(G.classes);
    group.gens        = G.gens;
end
