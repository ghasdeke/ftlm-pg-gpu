function [M_list, C_M, chi_M, Z_M, S] = observables_per_M(mat_file, T_range)
%OBSERVABLES_PER_M  Per-|M|-sector observables from ONE results file.
%
%   [M_LIST, C_M, CHI_M, Z_M] = OBSERVABLES_PER_M(MAT_FILE, T_RANGE)
%
%   Splits the raw FTLM data (all_E / all_w / all_M) of a driver results
%   file by |M| sector and evaluates each sector's observables on T_RANGE:
%       M_LIST [n_M x 1]   the |M| values present (ascending)
%       C_M    [n_M x n_T] sector-internal specific heat
%       CHI_M  [n_M x n_T] sector susceptibility contribution
%       Z_M    [n_M x n_T] sector Z_eff (each shifted by ITS OWN sector E0)
%
%   The stored weights already carry the M <-> -M multiplicity (mult_M),
%   so row M is the FULL |M| contribution including the mirror sector.
%
%   ADDITIVITY WARNING: Z and the weighted moments are additive over
%   sectors, but C is NOT (the variance is nonlinear in the mixture) and
%   the per-sector Z_M use per-sector energy shifts. For TOTAL observables
%   evaluate the concatenated raw data -- REEVAL_OBSERVABLES_PG for one
%   file, MERGE_OBSERVABLES_PG for one-file-per-M runs. This function is
%   for analysing sectors INDIVIDUALLY.
%
%   See also MERGE_OBSERVABLES_PG, REEVAL_OBSERVABLES_PG,
%            COMPUTE_OBSERVABLES_PG.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    S = load(mat_file);
    for f = {'all_E', 'all_w', 'all_M'}
        assert(isfield(S, f{1}), ...
            'observables_per_M: %s has no raw field ''%s'' -- not a driver results file.', ...
            mat_file, f{1});
    end
    T_range = double(T_range(:)');
    M_list  = unique(double(S.all_M(:)));
    n_M = numel(M_list);  n_T = numel(T_range);
    C_M = zeros(n_M, n_T);  chi_M = zeros(n_M, n_T);  Z_M = zeros(n_M, n_T);
    for k = 1 : n_M
        sel = (double(S.all_M(:)) == M_list(k));
        [C_M(k, :), chi_M(k, :), Z_M(k, :)] = compute_observables_pg( ...
            S.all_E(sel), S.all_w(sel), S.all_M(sel), T_range);
    end
end
