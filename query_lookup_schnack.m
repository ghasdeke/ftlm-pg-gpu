function idx = query_lookup_schnack(lookup, states)
%QUERY_LOOKUP_SCHNACK  Batched Schnack-CR state -> super_rep_idx query.
%
%   IDX = QUERY_LOOKUP_SCHNACK(LOOKUP, STATES) returns the 1-based
%   position of each state in LOOKUP.SUPER_REPS, or 0 if the state is
%   not a super-rep. Signature compatible with QUERY_CLT_LOOKUP so the
%   two lookup backends are interchangeable in COLLECT_CLT_ENTRIES_IH.
%
%   Two-step query:
%       1. SCHNACK_RANK(state) -> rank in M-sector (O(N) ops, no table
%          lookup besides D_cum which stays in L1).
%       2. ISMEMBER(rank, LOOKUP.SUPER_REPS_RANK) -> position
%          (binary-search internally; SUPER_REPS_RANK is sorted).
%
%   The result is exact: a state s is a super-rep iff its M-sector
%   rank equals some LOOKUP.SUPER_REPS_RANK(i), in which case the
%   returned index is i.
%
%   Assumption: STATES belong to the same M-sector as the super-reps
%   (i.e., sum of digits = LOOKUP.A_TOTAL). For Heisenberg spin flips
%   this holds by construction (M is conserved). If a state with
%   wrong digit sum is queried, SCHNACK_RANK returns garbage; the
%   subsequent ISMEMBER will then most likely return 0 (no match),
%   but the behaviour is not guaranteed. We do NOT add an explicit
%   M-check here for performance reasons; callers that may pass
%   foreign sectors should pre-filter.
%
%   See also BUILD_LOOKUP_SCHNACK, SCHNACK_RANK, QUERY_CLT_LOOKUP.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    if isempty(states)
        idx = zeros(0, 1, 'int32');
        return;
    end

    %% Rank each state.
    ranks = schnack_rank(states, lookup.D_cum, lookup.N_sites, ...
                          lookup.two_s, lookup.d_loc, lookup.A_total);

    %% Binary search in the sorted super-rep ranks. ismember on int64
    %  with a sorted second argument uses sort-and-binary-search
    %  internally; it returns 0 for non-matches.
    [~, pos] = ismember(ranks, lookup.super_reps_rank);

    idx = int32(pos);                              % 0 = not a super-rep
end
