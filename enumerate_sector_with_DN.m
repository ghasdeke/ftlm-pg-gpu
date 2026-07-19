function [super_reps, orbit_lens, type_arr, sigma_R_idx_in_C, ...
          lam_real, reps_C, L_C] = enumerate_sector_with_DN( ...
                                       N, s_val, M_target, p_irrep, sigma_par)
%ENUMERATE_SECTOR_WITH_DN  Sigma-super-rep basis for the D_N sector at p in {0, N/2}.
%
%   [SUPER_REPS, ORBIT_LENS, TYPE_ARR, SIGMA_R_IDX_IN_C, LAM_REAL,
%    REPS_C, L_C] = ENUMERATE_SECTOR_WITH_DN(N, S_VAL, M_TARGET,
%                                            P_IRREP, SIGMA_PAR)
%
%   enumerates the (M, p, sigma_par) basis under the dihedral group
%   D_N = C_N x Z_2_sigma for an N-site ring, at a real momentum
%   p_irrep in {0, N/2}. For other p_irrep this function still runs but
%   the returned basis is not a true symmetry-adapted block (sigma maps
%   such sectors to the (-k)-partner, not to itself).
%
%   Outputs:
%       super_reps       sorted column of integer labels of the
%                         sigma-super-reps in this (M, p, sigma_par)
%                         block (int64).
%       orbit_lens       C_N orbit length per super-rep (int32).
%       type_arr         1 for self-stable super-reps (sigma_R = r),
%                         0 for cross-paired (sigma_R != r). int32.
%       sigma_R_idx_in_C index in REPS_C of the sigma-image rep per
%                         super-rep; for self-stable equals the super-rep's
%                         own C_N index. int32.
%       lam_real         real part of lambda_sigma per super-rep:
%                         +/-1 (for p in {0, N/2}). Used by the SpMV and
%                         by the sparse H builder.
%       reps_C           the underlying C_N rep basis at (M, p_irrep).
%       L_C              C_N orbit lengths of REPS_C.
%
%   Compatibility filter:
%       Cross-paired r is included iff r <= sigma_partner(r) (canonical
%       choice of the smaller of the pair); self-stable r is included
%       iff its sigma-eigenvalue (real part of lambda_sigma) equals
%       SIGMA_PAR.
%
%   See also APPLY_SIGMA_RING, SIGMA_ACTION_PG,
%            ENUMERATE_SECTOR_WITH_TRANSLATION.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    assert(sigma_par == +1 || sigma_par == -1, ...
        'enumerate_sector_with_DN: sigma_par must be +1 or -1.');

    [reps_C, L_C, dim_C] = enumerate_sector_with_translation( ...
                                N, s_val, M_target, p_irrep);
    if dim_C == 0
        super_reps = int64([]); orbit_lens = int32([]);
        type_arr = int32([]); sigma_R_idx_in_C = int32([]);
        lam_real = []; return;
    end

    [sR_idx, ~, lam, is_self] = sigma_action_pg(reps_C, N, s_val, p_irrep);
    lam_real_all = real(lam);

    % Cross-paired sigma partner state (in reps_C order)
    sigma_partner_state = reps_C(sR_idx);

    keep_mask = false(dim_C, 1);
    type_all  = zeros(dim_C, 1, 'int32');

    for t = 1 : dim_C
        if is_self(t)
            % self-stable: keep if lambda = sigma_par
            if abs(lam_real_all(t) - sigma_par) < 1e-8
                keep_mask(t) = true;
                type_all(t)  = int32(1);
            end
        else
            % cross-paired: keep the smaller of {r, sigma_partner(r)}
            if reps_C(t) <= sigma_partner_state(t)
                keep_mask(t) = true;
                type_all(t)  = int32(0);
            end
        end
    end

    super_reps       = reps_C(keep_mask);
    orbit_lens       = L_C(keep_mask);
    type_arr         = type_all(keep_mask);
    sigma_R_idx_in_C = sR_idx(keep_mask);
    lam_real         = lam_real_all(keep_mask);
end
