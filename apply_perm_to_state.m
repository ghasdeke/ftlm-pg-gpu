function out_states = apply_perm_to_state(perm, in_states, d_loc, N_sites)
%APPLY_PERM_TO_STATE  Apply a vertex permutation to integer state labels.
%
%   OUT_STATES = APPLY_PERM_TO_STATE(PERM, IN_STATES, D_LOC, N_SITES)
%
%   Given a permutation PERM of N_SITES vertices (PERM(i) = j means
%   site i is mapped to site j) and an array IN_STATES of integer
%   state labels (each in [0, D_LOC^N_SITES)), returns the array of
%   permuted state labels.
%
%   Encoding convention (matches the rest of the codebase):
%       state n = sum_k a_k * D_LOC^k
%   where a_k is the shifted local quantum number at site k. The new
%   state's digit at site PERM(k) is a_k:
%       n' = sum_k a_k * D_LOC^{PERM(k)-1}    (1-indexed sites)
%
%   Vectorized over IN_STATES.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    in_states  = int64(in_states(:));
    d_loc_i64  = int64(d_loc);
    out_states = zeros(size(in_states), 'int64');

    powers_dst = int64(d_loc) .^ int64(perm(:) - 1);   % d_loc^(perm(k)-1)
    tmp = in_states;
    for k = 1 : N_sites
        dg  = mod(tmp, d_loc_i64);
        out_states = out_states + dg * powers_dst(k);
        tmp = idivide(tmp, d_loc_i64);
    end
end
