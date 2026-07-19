function ranks = schnack_rank_gpu(states, D_cum_gpu, N_sites, two_s, d_loc, A_total)
%SCHNACK_RANK_GPU  GPU-resident Schnack combinatorial ranking.
%
%   RANKS = SCHNACK_RANK_GPU(STATES, D_CUM_GPU, N_SITES, TWO_S, D_LOC, A_TOTAL)
%
%   Same algorithm as SCHNACK_RANK but every step operates on
%   gpuArrays. STATES is gpuArray int64; D_CUM_GPU is the Schnack
%   cumulative dimension table uploaded once and cached by the caller.
%   The N_sites-step loop runs uniformly on every state, so the
%   throughput is bounded by the lookup-table read pattern (already
%   tiny enough to stay in L1) and the gpuArray indexing cost.
%
%   See also SCHNACK_RANK, BUILD_LOOKUP_SCHNACK_GPU.

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
    n = numel(states);
    if n == 0
        ranks = gpuArray(zeros(0, 1, 'int64'));
        return;
    end

    N_sites = double(N_sites);
    A_total = double(A_total);
    d_loc_i64 = int64(d_loc);

    %% Pre-extract digits on GPU. Three exact paths (GPU_INT64_ARITH_OK):
    %  float per-position extraction for n_total <= 2^52 (any d_loc; rem/
    %  division of exact double integers -- see the MIN_IMAGE_IH_GPU note),
    %  the int64 loop where the device supports gpuArray int64 arithmetic,
    %  else host decomposition + upload (CUDA forward compatibility, e.g.
    %  B200 on R2024b). Digits are identical integers on every path.
    n_total = double(d_loc) ^ double(N_sites);
    digits  = gpuArray.zeros(N_sites, n, 'int8');
    if n_total <= 2^52
        sd = double(states);
        dl = double(d_loc);
        for p = 1 : N_sites
            dgp = rem(sd, dl);              % exact: sd is an integer < 2^52
            digits(p, :) = int8(dgp).';
            sd = (sd - dgp) / dl;           % exact (an integer multiple of dl)
        end
    elseif gpu_int64_arith_ok()
        tmp = states;
        for p = 1 : N_sites
            dg = mod(tmp, d_loc_i64);
            digits(p, :) = int8(dg).';
            tmp = (tmp - dg) / d_loc_i64;
        end
    else
        digits_h = zeros(N_sites, n, 'int8');
        tmp = gather(states);
        for p = 1 : N_sites
            dg = mod(tmp, d_loc_i64);
            digits_h(p, :) = int8(dg).';
            tmp = (tmp - dg) / d_loc_i64;
        end
        digits = gpuArray(digits_h);
    end

    A_stride = int64(N_sites + 1);
    a_stride = int64((N_sites + 1) * (A_total + 1));

    ranks = gpuArray.zeros(n, 1, 'int64');
    A_vec = int64(A_total) * gpuArray.ones(n, 1, 'int64');

    for p = N_sites - 1 : -1 : 0
        a_p = int64(digits(p + 1, :)).';
        lin_idx = int64(p + 1) + A_vec * A_stride + a_p * a_stride;
        ranks   = ranks + D_cum_gpu(lin_idx);
        A_vec   = A_vec - a_p;
    end
end
