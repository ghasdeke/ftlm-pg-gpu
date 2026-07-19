function is_min = is_super_rep_Ih_gpu(states, group, s_val, perm_powers_gpu, B1)
%IS_SUPER_REP_IH_GPU  Staged early-reject orbit-minimum test (R2, GPU).
%
%   IS_MIN = IS_SUPER_REP_IH_GPU(STATES, GROUP, S_VAL, PERM_POWERS_GPU, B1)
%
%   Returns a logical gpuArray column: IS_MIN(i) is true iff STATES(i) is the
%   minimum of its I_h orbit (i.e. a super-rep). This is exactly equivalent to
%
%       MIN_IMAGE_IH_GPU(STATES, GROUP, S_VAL) == STATES
%
%   but it avoids materialising the full [order x n] image array for every
%   state. In an M sector only ~0.84% of states are super-reps, so most are
%   non-minimal -- and a non-minimal state has, on average, ~half the group
%   elements producing a strictly smaller image. We therefore test in TWO
%   stages:
%
%     Stage 1 (cheap, all n states): apply only a small block of B1
%       non-identity group elements. If their minimum image is < the state,
%       the state is DEFINITELY not a super-rep -> reject. The vast majority
%       die here, after a [B1 x n] matmul instead of [order x n].
%     Stage 2 (full, survivors only): apply the full non-identity group to the
%       (few) survivors and confirm min(images) >= state.
%
%   Correctness: a true super-rep has EVERY non-identity image >= state, so it
%   always survives Stage 1 and is confirmed in Stage 2; a rejected state has a
%   genuine smaller image. The kept set is therefore byte-identical to the
%   min_image-based test. (The identity maps a state to itself, so "state is
%   the orbit minimum" <=> "min over the NON-identity images >= state".)
%
%   PERM_POWERS_GPU (optional, [order x N] double gpuArray) is cached by the
%   caller and reused. B1 (optional, default 16) is the Stage-1 block size.
%
%   Used by ENUMERATE_M_ORBITS_IH_GPU's super-rep collection. COLLECT does NOT
%   use this -- it needs the full rep + g_min for every flip target (no early
%   out), so it keeps MIN_IMAGE_IH_GPU.
%
%   See also MIN_IMAGE_IH_GPU, ENUMERATE_M_ORBITS_IH_GPU.

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
        is_min = (states > 0);          % empty logical gpuArray of matching type
        return;
    end

    d_loc   = round(2*s_val + 1);
    N_sites = double(group.N);
    order   = double(group.order);
    %% Spin-flip Z2 (ADD_SPIN_FLIP_Z2): a state is a G x Z2 orbit minimum iff
    %  (a) no plain non-identity image is smaller AND (b) no FLIPPED image is
    %  smaller. By flip(g*state) = C - (g*state) with C = d_loc^N - 1, (b) is
    %  "C - max(plain images incl. identity) >= state" -- the identity counts
    %  here because the pure flip (e, P) is a valid non-identity element.
    %  Both stages below carry the flip check alongside the plain one; a
    %  partial-block max gives a valid early REJECT (any image > C - state).
    has_flip = isfield(group, 'flip') && any(group.flip);
    if has_flip
        n_g0   = double(group.n_g0);
        order  = n_g0;                  % matmul rows: the permutation half only
        C_flip = double(d_loc)^N_sites - 1;
    end
    if nargin < 4 || isempty(perm_powers_gpu)
        perms_d = double(group.perms);
        if has_flip, perms_d = perms_d(1:n_g0, :); end
        perm_powers_gpu = gpuArray(double(d_loc) .^ (perms_d - 1));
    elseif has_flip && size(perm_powers_gpu, 1) > n_g0
        perm_powers_gpu = perm_powers_gpu(1:n_g0, :);
    end
    if nargin < 5 || isempty(B1)
        B1 = 8;     % tuned: ~half the group reduces a median non-min state, so
                    % an 8-element block rejects ~all of them; survivors (~6-7%
                    % of an M sector: low-rank non-minima + the 0.84% true minima)
                    % get the full Stage-2 test. ~4.6x vs full min_image at s=7/2.
    end

    %% Digits [N x n] -- same decomposition as MIN_IMAGE_IH_GPU, so the
    %  perm_powers*digits images are bit-identical. The float broadcast path
    %  is exact for ANY d_loc when n_total <= 2^52 (floor of a correctly-
    %  rounded quotient of integers < 2^52 cannot cross an integer boundary
    %  -- see the MIN_IMAGE_IH_GPU selector note). Above 2^52: the exact
    %  int64 loop where the device supports it, else host decomposition +
    %  upload (CUDA forward compatibility, e.g. B200 -- GPU_INT64_ARITH_OK).
    n_total_mi  = double(d_loc) ^ N_sites;
    use_vec_dig = (n_total_mi <= 2^52);
    if use_vec_dig
        % cumprod, not .^: exact by induction below 2^52 on any platform.
        pw_row = cumprod([1, repmat(d_loc, 1, N_sites - 1)]);
        digits = rem(floor(double(states) ./ pw_row), d_loc).';     % [N x n]
    elseif gpu_int64_arith_ok()
        digits  = gpuArray.zeros(N_sites, n);
        tmp     = states;
        dl      = int64(d_loc);
        for k = 1 : N_sites
            dg = mod(tmp, dl);
            digits(k, :) = double(dg).';
            tmp = (tmp - dg) / dl;
        end
    else
        digits = gpuArray(host_digits_i64(gather(states), int64(d_loc), N_sites));
    end
    states_row = double(states).';      % [1 x n]

    % Non-identity group elements (the identity maps state->state, so it is
    % irrelevant to "is there a smaller image").
    idg   = double(group.identity);
    nonid = [1 : idg - 1, idg + 1 : order];
    B1    = min(B1, numel(nonid));

    %% Stage 1: cheap block. Survivors are states with no smaller image yet.
    if has_flip
        % Same B1-block matmul; its max additionally early-rejects via the
        % flip branch (any block image > C - state means a smaller flipped
        % image exists). max(.., states_row) folds in the identity image
        % (pure flip): states above C/2 die here for free.
        blk   = perm_powers_gpu(nonid(1:B1), :) * digits;            % [B1 x n]
        min1  = min(blk, [], 1);
        mx1   = max(blk, [], 1);
        clear blk;
        alive = (min1 >= states_row) & (C_flip - max(mx1, states_row) >= states_row);
        clear min1 mx1;
    else
        min1  = min(perm_powers_gpu(nonid(1:B1), :) * digits, [], 1);    % [1 x n]
        alive = (min1 >= states_row);
        clear min1;
    end

    %% Stage 2: confirm survivors against the full non-identity group.
    %  Sub-chunk the survivors so the [order x n_surv] image array never
    %  exceeds the GPU's max-variable-size / VRAM. At large N with many
    %  low-range survivors n_surv can reach ~1.6e7, and [order x n_surv] then
    %  exceeds 2^31 elements (hit at N=36 kagome, |G|=144). Bit-identical to
    %  the single-shot version (per-survivor min over the same nonid images).
    is_min = alive;                     % rejected states stay false
    surv   = find(alive);
    if ~isempty(surv)
        ns  = numel(surv);
        sub = max(1, floor(2e8 / order));        % bound [order x sub] ~ 2e8 elems
        for c0 = 1 : sub : ns
            c1 = min(c0 + sub - 1, ns);
            sc = surv(c0:c1);
            if has_flip
                % One matmul over ALL permutation rows: its max feeds the
                % flip check (identity row included -- pure flip is valid),
                % then the identity row is poked to inf for the plain min.
                imgs = perm_powers_gpu * digits(:, sc);          % [n_g0 x len]
                mxf  = max(imgs, [], 1);
                imgs(idg, :) = inf;
                mf   = min(imgs, [], 1);
                is_min(sc) = (mf >= states_row(sc)) & ...
                             (C_flip - mxf >= states_row(sc));
                clear imgs;
            else
                mf = min(perm_powers_gpu(nonid, :) * digits(:, sc), [], 1);
                is_min(sc) = (mf >= states_row(sc));
            end
        end
    end
    is_min = is_min(:);
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
