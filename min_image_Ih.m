function [reps, g_min] = min_image_Ih(states, group, s_val)
%MIN_IMAGE_IH  Minimum-image search over the I_h group (BLAS-batched).
%
%   [REPS, G_MIN] = MIN_IMAGE_IH(STATES, GROUP, S_VAL)
%
%   For each integer-encoded state in STATES, computes:
%       REPS(i)  : smallest integer in the I_h orbit of STATES(i)
%                  (the super-rep)
%       G_MIN(i) : index (1..120) of a group element g such that
%                  g(STATES(i)) = REPS(i). If STATES(i) is already
%                  its own orbit minimum, G_MIN(i) = GROUP.IDENTITY.
%
%   Vectorised implementation: rather than calling APPLY_PERM_TO_STATE
%   120 times inside a for-loop, this implementation expresses the entire
%   action of all 120 group elements on all input states as a SINGLE
%   matrix multiplication
%
%       states_g_all[g, i] = sum_k perm_powers[g, k] * digits[k, i]
%
%   where digits[k, i] is the k-th base-d_loc digit of state i and
%   perm_powers[g, k] = d_loc^(perms(g, k) - 1) encodes the contribution
%   of the k-th input digit to the state index after applying group
%   element g. The matmul is delegated to MATLAB's BLAS backend and is
%   typically 30-50x faster than the previous loop on a many-thousand
%   state vector.
%
%   Convention: matches the previous per-g-element loop exactly. When
%   the input state is already the orbit minimum, G_MIN returns
%   GROUP.IDENTITY (i.e., never any other group element with non-trivial
%   stabiliser on the rep).
%
%   See also APPLY_PERM_TO_STATE, ICOSAHEDRON_IH_FULL.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    d_loc   = round(2*s_val + 1);
    N_sites = double(group.N);     % was hardcoded 12; now driven by group
    states = int64(states(:));
    n = numel(states);

    if n == 0
        reps  = int64([]);
        g_min = int32([]);
        return;
    end

    %% Per-group-element power-of-d_loc table (small, group_order x N_sites).
    %  Spin-flip Z2 (ADD_SPIN_FLIP_Z2): only the PERMUTATION half of the
    %  doubled group enters the matmul; the flip half follows from the
    %  integer identity flip(g*state) = C - (g*state) with C = d_loc^N - 1
    %  (a permutation only reorders digit positions), i.e. min over the
    %  flipped images = C - max over the plain images -- same matmul, one
    %  extra max-reduction. g_min for a flip winner is n_g0 + argmax.
    has_flip = isfield(group, 'flip') && any(group.flip);
    if has_flip
        n_g0    = double(group.n_g0);
        perms_d = double(group.perms(1:n_g0, :));
        C_flip  = double(d_loc)^N_sites - 1;
    else
        perms_d = double(group.perms);
    end
    perm_powers  = d_loc .^ (perms_d - 1);          % [order(/2) x N_sites]
    identity_idx = double(group.identity);
    d_loc_i64    = int64(d_loc);

    %% Fast digit decomposition selector. The per-position int64 idivide
    %  loop below is ~60% of min_image's cost at N=36 (BENCH_MIN_IMAGE_
    %  BREAKDOWN). For power-of-two d_loc (s=1/2 -> 2, s=3/2 -> 4) division
    %  by d_loc^k is EXACT in floating point (a pure exponent shift), so a
    %  single broadcast extraction replaces the loop (~2.6x faster on the
    %  digit phase, identical output, same memory). Non-power-of-two spins
    %  (s=1 -> 3, s=2 -> 5) and anything near the 2^52 mantissa limit keep
    %  the exact idivide loop.
    n_total_mi   = double(d_loc) ^ N_sites;
    is_pow2_dloc = (bitand(uint64(d_loc), uint64(d_loc) - 1) == 0);
    use_vec_dig  = is_pow2_dloc && (n_total_mi <= 2^52);
    pw_row       = d_loc .^ (0 : N_sites - 1);      % [1 x N_sites], exact doubles

    %% Pure-streaming over the state axis. No N_sites x n digits array
    %  is EVER allocated; per chunk we build digits_chunk, do the BLAS
    %  matmul, take the min, write outputs, and discard. At
    %  icosidodecahedron / s=1/2 / M=0 scale (n ~ 1.55e8) this keeps
    %  peak transient host memory at the chunk level (~ 1-2 GB) instead
    %  of the previous 37 GB N_sites x n digits + 138 GB matmul.
    bytes_per_double = 8;
    bytes_per_state  = (N_sites + double(group.order)) * bytes_per_double;
    target_bytes     = 2e9;                          % ~ 2 GB peak per chunk
    chunk_size       = max(1, floor(target_bytes / bytes_per_state));
    chunk_size       = min(chunk_size, n);

    % g_min is materialised ONLY when the caller actually requests it.
    % The super-rep test in the M-sector enumeration calls this with a
    % single output and discards g_min; allocating the full-length
    % g_min_d_full (a double array the size of STATES) and running the
    % take-identity bookkeeping there is pure waste. Gate both on
    % nargout. The reps value is computed identically in both branches.
    want_gmin = (nargout >= 2);

    reps_d_full = zeros(1, n);
    if want_gmin
        g_min_d_full = zeros(1, n);
    end

    for cs = 1 : chunk_size : n
        ce  = min(cs + chunk_size - 1, n);
        idx = cs : ce;
        states_chunk = states(idx);

        % digits_chunk built per-chunk only. Vectorised exact extraction
        % for power-of-two d_loc, else the exact int64 idivide loop.
        if use_vec_dig
            digits_chunk = double(rem(floor(double(states_chunk) ./ pw_row), d_loc)).';
        else
            digits_chunk = zeros(N_sites, numel(idx));
            tmp = states_chunk;
            for k = 1 : N_sites
                digits_chunk(k, :) = double(mod(tmp, d_loc_i64)).';
                tmp = idivide(tmp, d_loc_i64);
            end
        end

        states_g_chunk = perm_powers * digits_chunk;     % [order(/2) x len]
        if has_flip
            % Min over the FLIPPED images = C - max over the plain images,
            % INCLUDING the identity row (the pure flip (e, P) is a valid
            % non-identity element of G x Z2). Taken BEFORE the identity row
            % is poked to inf below.
            if want_gmin
                [mx_chunk, g_max_chunk] = max(states_g_chunk, [], 1);
            else
                mx_chunk = max(states_g_chunk, [], 1);
            end
            flip_min_chunk = C_flip - mx_chunk;
        end
        states_g_chunk(identity_idx, :) = inf;
        input_d_chunk = double(states_chunk).';          % [1 x len]

        if want_gmin
            [reps_non_id, g_min_non_id] = min(states_g_chunk, [], 1);
            if has_flip
                take_flip = flip_min_chunk < reps_non_id;     % perm branch wins ties
                g_min_non_id(take_flip) = n_g0 + g_max_chunk(take_flip);
                reps_non_id = min(reps_non_id, flip_min_chunk);
            end
            take_identity_c = input_d_chunk <= reps_non_id;
            reps_chunk      = min(input_d_chunk, reps_non_id);
            g_min_chunk     = g_min_non_id;
            g_min_chunk(take_identity_c) = identity_idx;
            reps_d_full(idx)  = reps_chunk;
            g_min_d_full(idx) = g_min_chunk;
        else
            reps_non_id      = min(states_g_chunk, [], 1);
            if has_flip
                reps_non_id  = min(reps_non_id, flip_min_chunk);
            end
            reps_d_full(idx) = min(input_d_chunk, reps_non_id);
        end

        % Help GC across iterations on tight-RAM runs.
        clear digits_chunk states_g_chunk;
    end

    reps = int64(reps_d_full).';
    if want_gmin
        g_min = int32(g_min_d_full).';
    end
end
