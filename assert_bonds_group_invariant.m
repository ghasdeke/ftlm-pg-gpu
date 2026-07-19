function assert_bonds_group_invariant(bonds, group, label)
%ASSERT_BONDS_GROUP_INVARIANT  Error if the bond set is not group-invariant.
%
%   ASSERT_BONDS_GROUP_INVARIANT(BONDS, GROUP, LABEL) checks that the
%   nearest-neighbour bond list is closed under EVERY permutation of the
%   symmetry group and errors with an actionable message if not.
%
%   WHY THIS MUST BE A RUNTIME GUARD: the symmetry-adapted decomposition is
%   only correct when the coupling set is invariant under the group -- and a
%   violation is NOT caught by the FTLM sum rule (sum_i w_i = dim is a trace
%   identity that holds for ANY Hamiltonian, including a wrong one), so a
%   user-supplied bond list with a typo would produce silently wrong physics.
%   This guard makes it fail loudly at startup instead. Cost: one sortrows
%   per group element, O(|G| * E log E) -- milliseconds even at |G| = 65535.
%
%   BONDS : [E x 2] 1-based site pairs (orientation irrelevant).
%   GROUP : provider struct with .perms ([|G| x N], perms(k,i)=j: site i->j)
%           and .order. Any provider / GROUP_FROM_GENERATORS output works.
%   LABEL : optional context string for the error message.
%
%   See also GROUP_FROM_GENERATORS, FTLM_OBSERVABLES_PG_GPU_IH,
%            FTLM_OBSERVABLES_PG_IH.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    if nargin < 3, label = 'generators'; end
    assert(size(bonds, 2) == 2 && ~isempty(bonds), ...
        'assert_bonds_group_invariant: BONDS must be a non-empty [E x 2] list.');
    N = double(group.N);
    assert(all(bonds(:) >= 1 & bonds(:) <= N), ...
        ['%s: bond list contains site indices outside 1..%d -- bonds must use ', ...
         'the same 1-based site labelling as the permutations.'], label, N);

    B  = sort(double(bonds), 2);            % canonical orientation
    Bs = sortrows(B);
    E  = size(B, 1);
    for g = 1 : double(group.order)
        p  = double(group.perms(g, :));
        Bg = sortrows(sort(reshape(p(B), E, 2), 2));    % image of the bond set
        if ~isequal(Bg, Bs)
            miss = setdiff(Bg, Bs, 'rows');
            if isempty(miss), miss = [0, 0]; end   % multiplicity-only mismatch
            error('assert_bonds_group_invariant:notInvariant', ...
                ['%s: the bond list is NOT invariant under group element %d ', ...
                 '(e.g. image bond (%d,%d) is not in the list). The symmetry-', ...
                 'adapted decomposition would be silently WRONG (the sum rule ', ...
                 'does not catch this). Fix the bond list or the generators; ', ...
                 'both must use the same 1-based site labelling.'], ...
                label, g, miss(1, 1), miss(1, 2));
        end
    end
end
