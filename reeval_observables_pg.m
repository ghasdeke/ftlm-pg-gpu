function [C_T, chi_T, Z_eff, S] = reeval_observables_pg(mat_file, T_range)
%REEVAL_OBSERVABLES_PG  Recompute observables from a results file on ANY T grid.
%
%   [C_T, CHI_T, Z_EFF] = REEVAL_OBSERVABLES_PG(MAT_FILE, T_RANGE)
%   [C_T, CHI_T, Z_EFF, S] = ... additionally returns the loaded struct.
%
%   The driver results .mat stores the FTLM RAW DATA -- every Ritz energy
%   (all_E), its weight (all_w) and its M quantum number (all_M) -- not just
%   the observables on the run's T grid. C(T), chi(T) and Z_eff(T) are mere
%   post-processing sums over that data, so they can be re-evaluated on an
%   arbitrary temperature grid WITHOUT rerunning anything:
%
%       [C, chi, Z] = reeval_observables_pg( ...
%           'runs_dodec_s32_M0_R8/ftlm_pg_gpu_Ih_dodec_s3o2.mat', ...
%           logspace(-2, 2, 400));
%
%   On the original grid this reproduces the stored C_T/chi_T/Z_eff exactly
%   (bit-identical; verified on the dodecahedron s=3/2 production data).
%
%   VALIDITY NOTES (physics, not file format):
%   - An only_M0 run stores only the M=0-sector data: C(T) is that sector's
%     contribution; the full physical chi(T) needs a full-M-sweep run.
%   - Extending BELOW the original T range is technically free but only as
%     trustworthy as the low-energy convergence of the underlying Lanczos
%     data (M_lz, R) -- judge with the lz_diag report (E0 residual) of the
%     run. Extending to HIGHER T is unproblematic (converges trivially).
%   - all_M also enables field observables (M(B,T), chi(B,T)) as pure
%     post-processing on full-M-sweep data.
%
%   See also COMPUTE_OBSERVABLES_PG, FTLM_OBSERVABLES_PG_GPU_IH.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    S = load(mat_file);
    for f = {'all_E', 'all_w', 'all_M'}
        assert(isfield(S, f{1}), ...
            'reeval_observables_pg: %s has no raw field ''%s'' -- not a driver results file.', ...
            mat_file, f{1});
    end
    [C_T, chi_T, Z_eff] = compute_observables_pg(S.all_E, S.all_w, S.all_M, T_range);
end
