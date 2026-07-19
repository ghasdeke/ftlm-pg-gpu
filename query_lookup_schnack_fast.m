function idx = query_lookup_schnack_fast(lookup, states)
%QUERY_LOOKUP_SCHNACK_FAST  MEX-accelerated drop-in for QUERY_LOOKUP_SCHNACK.
%
%   IDX = QUERY_LOOKUP_SCHNACK_FAST(LOOKUP, STATES)
%
%   Identical contract and identical results to QUERY_LOOKUP_SCHNACK
%   (1-based super-rep index, 0 if STATES is not a super-rep), but routes
%   the work through the fused single-pass kernel SCHNACK_QUERY_MEX
%   (rank + binary search in one multithreaded loop) when it is built and
%   LOOKUP carries the host-side D_cum / super_reps_rank tables. Otherwise
%   it falls back to the pure-MATLAB QUERY_LOOKUP_SCHNACK, so the path is
%   always correct whether or not the MEX has been compiled.
%
%   The MEX is bit-for-bit equivalent to QUERY_LOOKUP_SCHNACK for states
%   in the same M-sector as the super-reps (guaranteed for M-conserving
%   Heisenberg flips), verified by TEST_SCHNACK_QUERY_MEX and, end-to-end,
%   by TEST_LOOKUP_SCHNACK_VS_BITMAP.
%
%   Note: the "is the MEX available" check is cached in a persistent flag
%   to keep the hot collect loop cheap. If you build the MEX mid-session,
%   run `clear query_lookup_schnack_fast` to re-detect it.
%
%   See also SCHNACK_QUERY_MEX, QUERY_LOOKUP_SCHNACK, BUILD_LOOKUP_SCHNACK,
%            COLLECT_CLT_ENTRIES_IH, BUILD_SCHNACK_QUERY_MEX.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    persistent have_mex
    if isempty(have_mex)
        have_mex = (exist('schnack_query_mex', 'file') == 3);
    end

    if isempty(states)
        idx = zeros(0, 1, 'int32');
        return;
    end

    if have_mex && isfield(lookup, 'super_reps_rank') && isfield(lookup, 'D_cum')
        idx = schnack_query_mex(int64(states(:)), lookup.D_cum, lookup.N_sites, ...
                                lookup.two_s, lookup.A_total, lookup.super_reps_rank);
    else
        idx = query_lookup_schnack(lookup, states);
    end
end
