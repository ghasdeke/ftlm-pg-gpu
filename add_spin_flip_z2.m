function group2 = add_spin_flip_z2(group)
%ADD_SPIN_FLIP_Z2  Extend a symmetry group by the spin-inversion Z2 (M=0 only).
%
%   GROUP2 = ADD_SPIN_FLIP_Z2(GROUP)
%
%   returns the direct-product group G' = G x {1, P}, where P is the global
%   spin-inversion (Sz -> -Sz). In the Sz product basis P is the PHASE-FREE
%   digit complement P|m> = |-m> (NOT exp(i*pi*Sy), which carries
%   (-1)^(s-m) phases): with S+-|m> = c+-(m)|m+-1>, c+-(m) =
%   sqrt(s(s+1) - m(m+-1)), one has c+(-m) = c-(m) exactly, hence
%   P S+- P = S-+ and [H_Heisenberg, P] = 0 for EVERY spin s, with all
%   matrix elements +1. P commutes with every site permutation, so G' is a
%   direct product and acts by "permute, then optionally complement" --
%   the whole pipeline stays a pure state-relabeling machinery and the
%   collect coefficients c_a are unchanged.
%
%   P preserves an M sector only for M = 0: use GROUP2 exclusively for the
%   M = 0 enumerate/collect/irrep chain. Payoff: n_reps, n_entries and the
%   per-irrep n_basis all (approximately) halve.
%
%   Output GROUP2 fields (all other GROUP fields are copied through):
%       perms     [2|G| x N]  the permutation PART of each element; the
%                 flip half duplicates the first half (the complement is
%                 NOT a site permutation -- it is carried by .flip)
%       flip      [2|G| x 1] logical; true for the elements g' = (g, P)
%       n_g0      |G| (number of pure-permutation elements, = first half)
%       order     2|G|
%       has_flip  true
%       identity  unchanged (lies in the first half)
%       irreps    doubled: each Gamma of G yields Gamma+ / Gamma- of G',
%                 rho'((g,f)) = (+-1)^f * rho(g); mats get 2|G| slices,
%                 names are suffixed '+' / '-'. NB: only .mats/.name/.d
%                 are rewritten; any extra per-irrep metadata (e.g.
%                 characters over the original |G| elements) is copied
%                 verbatim and must not be consumed for GROUP2.
%
%   Integer identity used downstream (min_image and friends): a site
%   permutation only reorders digit positions, so for every g
%       flip(g*state) = (d_loc^N - 1) - (g*state),
%   i.e. the flip branch of a min-image search needs NO extra matmul --
%   only a max-reduction over the plain images.
%
%   See also MIN_IMAGE_IH, ENUMERATE_M_ORBITS_IH_GPU, APPLY_IRREP_TO_ORBITS.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    assert(~(isfield(group, 'has_flip') && group.has_flip), ...
        'add_spin_flip_z2: group is already flip-extended.');

    n_g0 = double(group.order);

    group2          = group;
    group2.perms    = [group.perms; group.perms];
    group2.flip     = [false(n_g0, 1); true(n_g0, 1)];
    group2.n_g0     = n_g0;
    group2.order    = 2 * n_g0;
    group2.has_flip = true;

    % Double group.irreps if present (the generic space-group convention).
    % I_h polyhedron groups carry NAMED irrep fields (g.Ag, g.T1g, ...)
    % instead -- those are NOT rewritten here (they stay |G|-sliced and must
    % not be consumed for GROUP2); callers there double their own irrep
    % list, e.g. via the driver's use_spin_flip option.
    if isfield(group, 'irreps') && ~isempty(group.irreps)
        irr  = group.irreps;
        irr2 = irr([]);                      % empty struct array, same fields
        for k = 1 : numel(irr)
            for sgn = [1, -1]
                e = irr(k);
                if e.d == 1
                    m      = e.mats(:).';    % [1 x |G|] row, indexable mats(g)
                    e.mats = [m, sgn * m];   % [1 x 2|G|]
                else
                    e.mats = cat(3, e.mats, sgn * e.mats);   % [d x d x 2|G|]
                end
                if isfield(e, 'name')
                    if sgn > 0, e.name = [e.name '+']; else, e.name = [e.name '-']; end
                end
                irr2(end + 1) = e; %#ok<AGROW>
            end
        end
        group2.irreps = irr2;
    end
end
