function [super_reps, orbit_lens, g_min_arr, stab_chr_sum] = ...
            enumerate_sector_with_Ih(s_val, M_target, irrep_chars, group)
%ENUMERATE_SECTOR_WITH_IH  I_h super-rep enumeration for a 1D irrep.
%
%   [SUPER_REPS, ORBIT_LENS, G_MIN_ARR, STAB_CHR_SUM] = ...
%       ENUMERATE_SECTOR_WITH_IH(S_VAL, M_TARGET, IRREP_CHARS, GROUP)
%
%   Enumerates the (M, Gamma) symmetry-adapted basis for the
%   icosahedron under the full I_h group (|G| = 120), for a 1D
%   irreducible representation Gamma specified by its 120 characters.
%
%   Inputs:
%       s_val         local spin (0.5, 1, 1.5, ...)
%       M_target      total S^z (integer or half-integer)
%       irrep_chars   [120 x 1] real vector of Gamma's character per
%                     group element. For A_g: ones(120, 1). For A_u:
%                     group.det.
%       group         I_h struct returned by icosahedron_Ih_full.
%
%   Outputs:
%       super_reps    sorted column of orbit minima (int64).
%       orbit_lens    I_h orbit length per super-rep (int32). Always
%                     a divisor of 120.
%       g_min_arr     same length as SUPER_REPS, but trivially equal
%                     to the identity index because reps satisfy
%                     g(r) = r at g = identity. Returned for API
%                     consistency with the off-diagonal SpMV path.
%       stab_chr_sum  sum_{h in Stab(r)} chi_Gamma(h) per super-rep.
%                     Equals |Stab(r)| for A_g; can be 0 or |Stab(r)|
%                     in absolute value for A_u depending on stabilizer
%                     parity content.
%
%   Compatibility filter:
%       A super-rep is kept iff STAB_CHR_SUM is non-zero. For A_g this
%       is always the case. For A_u this fails for reps whose
%       stabilizer contains improper rotations whose contributions
%       cancel; those super-reps live exclusively in A_g.
%
%   See also MIN_IMAGE_IH, BUILD_HEISENBERG_SPARSE_IH,
%            ICOSAHEDRON_IH_FULL.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    d_loc = round(2*s_val + 1);
    N_sites = 12;
    n_total = int64(d_loc)^int64(N_sites);

    %% Step 1: enumerate M-sector basis (brute-force digit enumeration).
    if s_val == 0.5
        % Vectorized popcount path
        k_up = N_sites/2 + M_target;
        if k_up < 0 || k_up > N_sites || abs(k_up - round(k_up)) > 0.01
            super_reps = int64([]); orbit_lens = int32([]);
            g_min_arr = int32([]); stab_chr_sum = [];
            return;
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
        super_reps = int64([]); orbit_lens = int32([]);
        g_min_arr = int32([]); stab_chr_sum = []; return;
    end

    %% Step 2: orbit minima via min_image_Ih
    [reps_all, ~] = min_image_Ih(basis_M, group, s_val);
    is_rep_mask = (reps_all == basis_M);
    candidate_reps = basis_M(is_rep_mask);
    dim_cand = numel(candidate_reps);

    %% Step 3: stabilizer + 1D irrep compatibility filter
    orbit_lens_arr = zeros(dim_cand, 1, 'int32');
    stab_sum_arr   = zeros(dim_cand, 1);
    keep_mask      = false(dim_cand, 1);

    for i = 1 : dim_cand
        r = candidate_reps(i);
        stab_sum = 0.0;
        n_stab = 0;
        for g = 1 : group.order
            perm_g = double(group.perms(g, :));
            n_g = apply_perm_to_state(perm_g, r, d_loc, N_sites);
            if n_g == r
                stab_sum = stab_sum + irrep_chars(g);
                n_stab = n_stab + 1;
            end
        end
        orbit_lens_arr(i) = int32(group.order / n_stab);
        stab_sum_arr(i)   = stab_sum;
        if abs(stab_sum) > 1e-8
            keep_mask(i) = true;
        end
    end

    super_reps   = candidate_reps(keep_mask);
    orbit_lens   = orbit_lens_arr(keep_mask);
    stab_chr_sum = stab_sum_arr(keep_mask);
    g_min_arr    = group.identity * ones(numel(super_reps), 1, 'int32');
end
