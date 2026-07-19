function cache = enumerate_M_orbits_Ih(s_val, M_target, group)
%ENUMERATE_M_ORBITS_IH  Per-M precompute for the gamma.2 / FTLM pipeline.
%
%   CACHE = ENUMERATE_M_ORBITS_IH(S_VAL, M_TARGET, GROUP) does all the
%   irrep-INDEPENDENT work for the (M, Gamma) basis construction on the
%   icosahedron:
%
%     1. Enumerates the integer-encoded states with total S^z = M_target.
%     2. Calls MIN_IMAGE_IH (vectorised over all M-sector states) to find
%        the I_h-orbit minimum of each state, in one pass.
%     3. Identifies the super-reps (orbit minima sitting in this sector).
%     4. For each super-rep, builds a sorted list of group-element indices
%        that fix it (its stabiliser). The 120 stabiliser tests are done
%        as 120 *vectorised* APPLY_PERM_TO_STATE calls over the entire
%        super-rep array, not as N_reps x 120 scalar calls. This is the
%        single biggest speedup vs the previous all-in-one routine.
%     5. Stores everything in a struct that APPLY_IRREP_TO_ORBITS can
%        consume cheaply for every irrep Gamma.
%
%   The result is independent of the irrep, so this function is called
%   ONCE per M_target and reused across all 10 I_h irreps in the FTLM
%   drivers. On the s = 3/2 icosahedron (n_total ~ 1.7e7) this cuts the
%   prep phase from O(10-20 min) to O(2-3 min).
%
%   Inputs:
%       s_val       local spin
%       M_target    total S^z (integer or half-integer)
%       group       struct from ICOSAHEDRON_IH_FULL
%
%   Output struct CACHE contains:
%       M_target       echoed
%       s_val          echoed
%       super_reps     [n_reps x 1] int64, sorted I_h orbit minima in
%                      the M sector
%       orbit_lens     [n_reps x 1] int32, |G| / |Stab(r)| per rep
%       stab_flat      [sum|Stab| x 1] uint8, all stabiliser group-element
%                      indices (1-based) concatenated in rep order (CSR
%                      values). Replaces the old {n_reps x 1} cell stab_lists.
%       stab_ptr       [n_reps+1 x 1] int64 CSR row pointers: the indices
%                      that fix super_reps(i) are
%                      stab_flat(stab_ptr(i):stab_ptr(i+1)-1).
%
%   See also APPLY_IRREP_TO_ORBITS, ENUMERATE_SECTOR_WITH_IH_GAMMA2,
%            MIN_IMAGE_IH, APPLY_PERM_TO_STATE.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    d_loc   = round(2*s_val + 1);
    N_sites = double(group.N);     % was hardcoded 12; now driven by group
    n_total = double(d_loc)^N_sites;

    %% Step 1: enumerate M-sector basis (irrep-independent).
    if s_val == 0.5
        % Fast popcount path
        k_up = N_sites/2 + M_target;
        if k_up < 0 || k_up > N_sites || abs(k_up - round(k_up)) > 0.01
            cache = empty_cache(s_val, M_target); return;
        end
        k_up = round(k_up);
        n_states_total = int64(2)^int64(N_sites);
        all_states = int64(0 : double(n_states_total) - 1)';
        pop = zeros(numel(all_states), 1);
        for b = 0 : N_sites - 1
            pop = pop + double(bitand(bitshift(all_states, -b), int64(1)));
        end
        basis_M = all_states(pop == k_up);
    else
        % General digit-decomposition path, vectorised over all states
        all_states = int64(0 : double(n_total) - 1)';
        Mv  = zeros(numel(all_states), 1);
        tmp = all_states;
        for kk = 1 : N_sites
            dg = double(mod(tmp, int64(d_loc)));
            Mv = Mv + dg - s_val;
            tmp = (tmp - int64(dg)) / int64(d_loc);
        end
        Mv = round(Mv * 2) / 2;
        basis_M = all_states(Mv == M_target);
    end

    if isempty(basis_M)
        cache = empty_cache(s_val, M_target); return;
    end

    %% Step 2: orbit minima via vectorised MIN_IMAGE_IH.
    [reps_all, ~] = min_image_Ih(basis_M, group, s_val);
    is_rep_mask = (reps_all == basis_M);
    super_reps  = basis_M(is_rep_mask);
    n_reps      = numel(super_reps);

    if n_reps == 0
        cache = empty_cache(s_val, M_target); return;
    end

    %% Step 3+4: stabiliser flag matrix [n_reps x order], vectorised.
    %  For each group element g we apply g to ALL super-reps in one call
    %  and check which ones are fixed. This replaces ~ n_reps * 120 scalar
    %  apply_perm_to_state calls with 120 vectorised ones.
    %  Spin-flip Z2 (ADD_SPIN_FLIP_Z2): only the PERMUTATION half is applied;
    %  a flip element (g, P) stabilises r iff g*r == C - r with
    %  C = d_loc^N - 1 (flip(g*r) = C - g*r), filling column n_g0 + g.
    has_flip = isfield(group, 'flip') && any(group.flip);
    n_act    = double(group.order);
    if has_flip
        n_act  = double(group.n_g0);
        C_flip = int64(n_total - 1);
    end
    stab_flag = false(n_reps, group.order);
    for g = 1 : n_act
        perm_g  = double(group.perms(g, :));
        reps_g  = apply_perm_to_state(perm_g, super_reps, d_loc, N_sites);
        stab_flag(:, g) = (reps_g == super_reps);
        if has_flip
            stab_flag(:, n_act + g) = (reps_g == C_flip - super_reps);
        end
    end

    %% Step 5: stabilisers in CSR form (rep i -> stab_flat(stab_ptr(i):
    %  stab_ptr(i+1)-1)), replacing the n_reps-cell stab_lists. Column-major
    %  find over [order x n_reps] groups g-indices by rep, ascending in g.
    counts      = sum(stab_flag, 2);                 % [n_reps x 1] |Stab_i|
    [g_rows, ~] = find(stab_flag.');                 % flat, rep order
    stab_flat   = uint16(g_rows);                    % g in 1..order (uint16: |G| up to 65535)
    stab_ptr    = [int64(1); int64(1) + cumsum(int64(counts))];
    orbit_lens  = int32(group.order ./ counts);

    %% Pack
    cache.M_target   = M_target;
    cache.s_val      = s_val;
    cache.super_reps = super_reps;
    cache.orbit_lens = orbit_lens;
    cache.stab_flat  = stab_flat;     % uint8  [sum|Stab| x 1]
    cache.stab_ptr   = stab_ptr;      % int64  [n_reps+1 x 1], CSR row pointers
end


% ----------------------------------------------------------------
function cache = empty_cache(s_val, M_target)
    cache.M_target   = M_target;
    cache.s_val      = s_val;
    cache.super_reps = int64([]);
    cache.orbit_lens = int32([]);
    cache.stab_flat  = zeros(0, 1, 'uint16');
    cache.stab_ptr   = int64(1);
end
