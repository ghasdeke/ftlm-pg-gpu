function [super_reps, V_per_rep, eig_per_rep, n_per_rep, L_per_rep] = ...
            enumerate_sector_with_Ih_gamma2(s_val, M_target, irrep_data, ...
                                             d_irrep, group)
%ENUMERATE_SECTOR_WITH_IH_GAMMA2  Super-rep enumeration + column basis for
% an arbitrary I_h irrep (Phase gamma.2 scalable construction).
%
%   Convenience wrapper that combines ENUMERATE_M_ORBITS_IH (irrep-
%   independent, runs once per M_target) and APPLY_IRREP_TO_ORBITS
%   (irrep-dependent, runs once per Gamma). Useful when only a single
%   (M, Gamma) block is needed.
%
%   For FTLM driver runs that loop over all 10 I_h irreps within each M
%   sector, the drivers call ENUMERATE_M_ORBITS_IH directly and then
%   reuse the cache via APPLY_IRREP_TO_ORBITS for each irrep. That
%   avoids redoing the O(n_total) enumeration and the O(120 * n_reps)
%   stabiliser search 10 times per M sector.
%
%   Inputs / outputs are identical to the previous monolithic
%   implementation.
%
%   See also ENUMERATE_M_ORBITS_IH, APPLY_IRREP_TO_ORBITS.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    cache = enumerate_M_orbits_Ih(s_val, M_target, group);
    [super_reps, V_per_rep, eig_per_rep, n_per_rep, L_per_rep] = ...
        apply_irrep_to_orbits(cache, irrep_data, d_irrep, group);
end
