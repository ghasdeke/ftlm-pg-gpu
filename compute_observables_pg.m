function [C_T, chi_T, Z_eff] = compute_observables_pg(all_E, all_w, all_M, T_range)
%COMPUTE_OBSERVABLES_PG  C(T), chi(T), Z_eff(T) from aggregated FTLM data.
%
%   [C_T, CHI_T, Z_EFF] = COMPUTE_OBSERVABLES_PG(ALL_E, ALL_W, ALL_M, T_RANGE)
%
%   computes the thermodynamic observables (specific heat, magnetic
%   susceptibility, and effective partition function) from a list of
%   (energy, weight, magnetization) triples collected across all
%   (M, p) sectors.
%
%   Inputs:
%       all_E    column of all Ritz values
%       all_w    column of FTLM weights, already multiplied by the
%                M <-> -M multiplicity factor (mult_M = 1 + (M > 0))
%       all_M    column of magnetic quantum numbers (M >= 0 only)
%       T_range  temperature grid (row or column vector, units of J/k_B)
%
%   Outputs:
%       C_T    specific heat C(T)            [1 x n_T]
%       chi_T  magnetic susceptibility chi(T) [1 x n_T]
%       Z_eff  Z_eff(T) = Z(T) * exp(beta*E0) [1 x n_T]
%
%   Formulas:
%       Z(beta)    = sum_i  w_i * exp(-beta * E_i)
%       <E>(beta)  = sum_i  w_i * E_i * exp(-beta * E_i) / Z
%       C(beta)    = beta^2 * Var(E)
%       <M^2>(beta)= sum_i  w_i * M_i^2 * exp(-beta * E_i) / Z
%       chi(beta)  = beta * <M^2>
%
%   The centred-variance form Var(E) = <(E - <E>)^2> is used, mathematically
%   identical to <E^2> - <E>^2 but free from catastrophic cancellation at
%   low T.
%
%   This is the symmetry-adapted analogue of compute_observables in
%   release/ftlm_observables.m. The only structural difference is that
%   ALL_M is now collected from (M, p)-sector blocks, but each block
%   has a single M value per row so the formulas are identical.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    all_E    = double(all_E(:));
    all_w    = double(all_w(:));
    all_M    = double(all_M(:));
    T_range  = double(T_range(:)');
    n_T      = length(T_range);
    beta_arr = 1.0 ./ T_range;

    if isempty(all_E)
        C_T = zeros(1, n_T); chi_T = zeros(1, n_T); Z_eff = zeros(1, n_T);
        return;
    end

    E_min = min(all_E);
    dE    = all_E - E_min;
    M2    = all_M .^ 2;

    C_T   = zeros(1, n_T);
    chi_T = zeros(1, n_T);
    Z_eff = zeros(1, n_T);

    for iT = 1 : n_T
        bet   = beta_arr(iT);
        boltz = all_w .* exp(-bet * dE);

        Z = sum(boltz);
        if Z < 1e-250
            continue;       % protect against underflow at very large beta
        end

        dE_avg = sum(dE .* boltz) / Z;
        E_var  = sum((dE - dE_avg).^2 .* boltz) / Z;
        M2_avg = sum(M2 .* boltz) / Z;

        C_T(iT)   = bet^2 * E_var;
        chi_T(iT) = bet * M2_avg;
        Z_eff(iT) = Z;
    end
end
