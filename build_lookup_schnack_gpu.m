function lookup = build_lookup_schnack_gpu(super_reps, s_val, N_sites)
%BUILD_LOOKUP_SCHNACK_GPU  Schnack lookup with GPU-resident tables.
%
%   LOOKUP = BUILD_LOOKUP_SCHNACK_GPU(SUPER_REPS, S_VAL, N_SITES)
%
%   Same role as BUILD_LOOKUP_SCHNACK but the large rank array
%   (super_reps_rank) is computed and stored on the GPU. SUPER_REPS
%   may be passed as host int64 or gpuArray int64.
%
%   The D_cum table is tiny (~ KB) and uploaded to GPU once. The
%   super_reps_rank array is n_super_reps x int64 (~ 5 MB at N=30,
%   ~ 300 MB at N=36), stays on GPU throughout the M sector.
%
%   See also BUILD_LOOKUP_SCHNACK, SCHNACK_RANK_GPU,
%            QUERY_LOOKUP_SCHNACK_GPU.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    %% Normalise input to gpuArray int64.
    if ~isa(super_reps, 'gpuArray')
        super_reps_gpu = gpuArray(int64(super_reps(:)));
    elseif ~strcmp(classUnderlying(super_reps), 'int64')
        super_reps_gpu = int64(super_reps(:));
    else
        super_reps_gpu = super_reps(:);
    end
    n_reps = numel(super_reps_gpu);

    two_s = round(2 * s_val);
    d_loc = two_s + 1;

    %% A_total from a single super-rep (or 0 for empty basis).
    if n_reps == 0
        A_total = round(N_sites * s_val);
    else
        % cheap: gather one element to host, decompose digits
        first_rep = gather(super_reps_gpu(1));
        A_total = sum_digits(first_rep, d_loc, N_sites);
    end

    %% D and D_cum on host (tiny), then upload D_cum to GPU.
    [D, D_cum_h] = build_D_table(N_sites, two_s, A_total);
    D_cum_gpu    = gpuArray(D_cum_h);

    %% Rank every super-rep on the GPU.
    if n_reps == 0
        super_reps_rank_gpu = gpuArray(zeros(0, 1, 'int64'));
    else
        super_reps_rank_gpu = schnack_rank_gpu(super_reps_gpu, D_cum_gpu, ...
                                                N_sites, two_s, d_loc, A_total);
    end

    %% Pack
    lookup.type                = 'schnack_gpu';
    lookup.N_sites             = double(N_sites);
    lookup.s_val               = s_val;
    lookup.two_s               = two_s;
    lookup.d_loc               = d_loc;
    lookup.A_total             = A_total;
    lookup.D                   = D;                 % host, tiny
    lookup.D_cum_gpu           = D_cum_gpu;
    lookup.super_reps_gpu      = super_reps_gpu;    % may be useful for sanity
    lookup.super_reps_rank_gpu = super_reps_rank_gpu;
    lookup.n_reps              = n_reps;
end


% ----------------------------------------------------------------
function A = sum_digits(state, d_loc, N_sites)
    A = 0;
    s = int64(state);
    di64 = int64(d_loc);
    for k = 1 : N_sites
        d = mod(s, di64);
        A = A + double(d);
        s = (s - d) / di64;
    end
end
