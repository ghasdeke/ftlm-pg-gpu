function [reps, g_min] = min_image_Ih_gpu(states, group, s_val, perm_powers_gpu)
%MIN_IMAGE_IH_GPU  GPU-native version of MIN_IMAGE_IH.
%
%   [REPS, G_MIN] = MIN_IMAGE_IH_GPU(STATES, GROUP, S_VAL)
%   [REPS, G_MIN] = MIN_IMAGE_IH_GPU(STATES, GROUP, S_VAL, PERM_POWERS_GPU)
%
%   Same contract as MIN_IMAGE_IH but everything stays in VRAM:
%   STATES is a gpuArray int64 column; REPS comes back as gpuArray
%   int64 and G_MIN as gpuArray int32. No host transient memory beyond
%   the small group tables. Used by the GPU-native COLLECT_CLT_ENTRIES
%   path (Phase B.2.1) to avoid the host -> gpuArray uploads per chunk
%   that defeated Stufe 6b.
%
%   PERM_POWERS_GPU (optional, [order x N_sites] double gpuArray) can
%   be cached by the caller and reused across many min_image calls in
%   the same M sector. If omitted it is built and uploaded internally.
%
%   The per-axis matmul perm_powers * digits is chunked over STATES so
%   the [group.order x chunk] intermediate stays bounded (target ~ 2 GB).
%
%   See also MIN_IMAGE_IH, COLLECT_CLT_ENTRIES_IH_GPU.

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

    d_loc   = round(2*s_val + 1);
    N_sites = double(group.N);

    %% Spin-flip Z2 (ADD_SPIN_FLIP_Z2): only the PERMUTATION half enters the
    %  matmul; the flip half follows from flip(g*state) = C - (g*state) with
    %  C = d_loc^N - 1 (same matmul, one extra max-reduction; g_min of a flip
    %  winner is n_g0 + argmax). Callers may pass a cached perm_powers table
    %  built from the FULL doubled group.perms -- slice its first half here.
    has_flip = isfield(group, 'flip') && any(group.flip);
    if has_flip
        n_g0   = double(group.n_g0);
        C_flip = double(d_loc)^N_sites - 1;
    end
    if nargin < 4 || isempty(perm_powers_gpu)
        perms_d = double(group.perms);
        if has_flip, perms_d = perms_d(1:n_g0, :); end
        perm_powers_gpu = gpuArray(d_loc .^ (perms_d - 1));
    elseif has_flip && size(perm_powers_gpu, 1) > n_g0
        perm_powers_gpu = perm_powers_gpu(1:n_g0, :);
    end
    identity_idx = double(group.identity);
    d_loc_i64    = int64(d_loc);

    if n == 0
        reps  = gpuArray(zeros(0, 1, 'int64'));
        g_min = gpuArray(zeros(0, 1, 'int32'));
        return;
    end

    %% Fast digit decomposition selector (mirrors MIN_IMAGE_IH). The int64
    %  mod/divide loop is emulated on the GPU and dominates min_image's
    %  cost; the broadcast extraction replaces it for ANY d_loc as long as
    %  n_total <= 2^52: with x < 2^52 an integer and p = d_loc^k an exact
    %  double, the correctly-rounded x./p errs by < (x/p)*2^-53 < 1/p while
    %  a non-integer x/p is at least 1/p away from the nearest integer --
    %  floor(x./p) is therefore EXACT (an integer quotient is returned
    %  exactly), and so is rem(.,d_loc) on the resulting < 2^52 integers.
    %  Above 2^52 the exact int64 loop is kept where the device supports
    %  gpuArray int64 arithmetic; under CUDA forward compatibility (e.g.
    %  B200 on R2024b, where int64 MOD is unavailable -- GPU_INT64_ARITH_OK)
    %  the digits are decomposed on the host and uploaded.
    n_total_mi  = double(d_loc) ^ N_sites;
    use_vec_dig = (n_total_mi <= 2^52);
    use_gpu_i64 = ~use_vec_dig && gpu_int64_arith_ok();
    % cumprod, not .^: every intermediate is an exact integer < 2^52, so the
    % powers are exact by induction on ANY platform (host pow happens to be
    % exact too, but that is a libm property, not an IEEE guarantee).
    pw_row      = cumprod([1, repmat(d_loc, 1, N_sites - 1)]);   % [1 x N_sites]

    %% Pure-streaming over states. Per chunk: digits_chunk + matmul + min.
    bytes_per_double = 8;
    bytes_per_state  = (N_sites + double(group.order)) * bytes_per_double;
    target_bytes     = 2e9;
    chunk_size       = max(1, floor(target_bytes / bytes_per_state));
    chunk_size       = min(chunk_size, n);

    reps_d_full  = gpuArray.zeros(1, n);
    g_min_d_full = gpuArray.zeros(1, n);

    for cs = 1 : chunk_size : n
        ce  = min(cs + chunk_size - 1, n);
        idx = cs : ce;
        states_chunk = states(idx);

        % Vectorised exact extraction (any d_loc, n_total <= 2^52), else the
        % exact int64 loop, else host decomposition + upload (forward-compat
        % GPUs without int64 mod). All yield the same [N_sites x len] double
        % gpuArray -- the digits are exact integers on every path.
        if use_vec_dig
            digits_chunk = rem(floor(double(states_chunk) ./ pw_row), d_loc).';
        elseif use_gpu_i64
            digits_chunk = gpuArray.zeros(N_sites, numel(idx));
            tmp = states_chunk;
            for k = 1 : N_sites
                dg = mod(tmp, d_loc_i64);
                digits_chunk(k, :) = double(dg).';
                tmp = (tmp - dg) / d_loc_i64;
            end
        else
            digits_chunk = gpuArray(host_digits_i64(gather(states_chunk), ...
                                                    d_loc_i64, N_sites));
        end

        states_g_chunk = perm_powers_gpu * digits_chunk;     % [order(/2) x len]
        if has_flip
            % Flip branch BEFORE the identity row is poked to inf (the pure
            % flip (e, P) is a valid non-identity element of G x Z2).
            [mx_chunk, g_max_chunk] = max(states_g_chunk, [], 1);
            flip_min_chunk = C_flip - mx_chunk;
        end
        states_g_chunk(identity_idx, :) = inf;
        [reps_non_id, g_min_non_id] = min(states_g_chunk, [], 1);
        if has_flip
            take_flip = flip_min_chunk < reps_non_id;        % perm branch wins ties
            g_min_non_id(take_flip) = n_g0 + g_max_chunk(take_flip);
            reps_non_id = min(reps_non_id, flip_min_chunk);
        end

        input_d_chunk   = double(states_chunk).';
        take_identity_c = input_d_chunk <= reps_non_id;
        reps_chunk      = min(input_d_chunk, reps_non_id);
        g_min_chunk     = g_min_non_id;
        g_min_chunk(take_identity_c) = identity_idx;

        reps_d_full(idx)  = reps_chunk;
        g_min_d_full(idx) = g_min_chunk;

        clear digits_chunk states_g_chunk;
    end

    reps  = int64(reps_d_full).';
    g_min = int32(g_min_d_full).';
end

% ----------------------------------------------------------------
function digits = host_digits_i64(states_h, d_loc_i64, N_sites)
%   Exact base-d_loc digits on the HOST (int64 loop), [N_sites x n] double.
%   Fallback for forward-compatibility GPUs (no gpuArray int64 mod) when
%   n_total > 2^52 rules out the float-exact broadcast path.
    digits = zeros(N_sites, numel(states_h));
    tmp    = states_h(:);
    for k = 1 : N_sites
        dg = mod(tmp, d_loc_i64);
        digits(k, :) = double(dg).';
        tmp = (tmp - dg) / d_loc_i64;
    end
end
