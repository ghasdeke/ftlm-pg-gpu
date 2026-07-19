function idx = query_lookup_schnack_gpu(lookup, states)
%QUERY_LOOKUP_SCHNACK_GPU  Batched GPU-native Schnack lookup query.
%
%   IDX = QUERY_LOOKUP_SCHNACK_GPU(LOOKUP, STATES)
%
%   STATES is a gpuArray int64 column (or convertible). LOOKUP comes
%   from BUILD_LOOKUP_SCHNACK_GPU and carries D_cum + sorted
%   super_reps_rank on the GPU. The two-step query is:
%
%     1. SCHNACK_RANK_GPU(state)   -> rank in M-sector
%     2. ISMEMBER(rank, super_reps_rank_gpu)   -> position (0 if miss)
%
%   Returns IDX as gpuArray int32; 0 marks states that are not
%   super-reps of the current M sector.
%
%   See also BUILD_LOOKUP_SCHNACK_GPU, SCHNACK_RANK_GPU,
%            QUERY_LOOKUP_SCHNACK.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    if ~isa(states, 'gpuArray')
        states = gpuArray(int64(states(:)));
    elseif ~strcmp(classUnderlying(states), 'int64')
        states = int64(states(:));
    else
        states = states(:);
    end

    if isempty(states)
        idx = gpuArray(zeros(0, 1, 'int32'));
        return;
    end

    ranks = schnack_rank_gpu(states, lookup.D_cum_gpu, lookup.N_sites, ...
                              lookup.two_s, lookup.d_loc, lookup.A_total);

    %% GPU ismember. MATLAB's gpuArray/ismember REJECTS int64/uint64
    %  inputs (as of R2024+), so we cast to double on both sides.
    %  ranks are integer-valued and bounded by D(N_sites, A_total);
    %  for any system size we will ever consider this stays well below
    %  2^53 (the exact-integer limit of double), so the cast is exact
    %  and the equality test remains bit-correct.
    [~, pos] = ismember(double(ranks), double(lookup.super_reps_rank_gpu));
    idx = int32(pos);
end
