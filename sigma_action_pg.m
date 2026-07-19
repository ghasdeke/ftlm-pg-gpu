function [sigma_R_idx, h_sigma, lambda_sigma, is_self_stable] = ...
            sigma_action_pg(reps, N, s_val, p_irrep)
%SIGMA_ACTION_PG  Action of the ring reflection sigma on a C_N rep basis.
%
%   [SIGMA_R_IDX, H_SIGMA, LAMBDA_SIGMA, IS_SELF_STABLE] =
%       SIGMA_ACTION_PG(REPS, N, S_VAL, P_IRREP)
%
%   For each C_N representative r = reps(t), computes:
%
%   - sigma_R_idx(t) : index in REPS of the orbit minimum of sigma(r).
%   - h_sigma(t)     : translation with T^{h_sigma}(sigma(r)) = reps(sigma_R_idx(t))
%   - lambda_sigma(t): phase factor exp(-1i * k * h_sigma(t)) where k =
%                      2*pi*P_IRREP/N. Real (+/-1) for P_IRREP in {0, N/2}.
%   - is_self_stable : true if sigma_R_idx(t) == t (rep is its own sigma image).
%
%   This is the analogue of PARITY_ACTION_PG for the spatial reflection
%   sigma. It is the building block of the D_N (= C_N x Z_2) basis
%   restriction at real momenta p = 0 and p = N/2.
%
%   The min-image search is vectorized over the full input array: the
%   reflected states are computed once by APPLY_SIGMA_RING and then a
%   single N-1 cyclic-shift pass identifies orbit minima and the
%   translation h_sigma in parallel.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    d_loc   = round(2*s_val + 1);
    dim     = length(reps);
    n_total = int64(d_loc)^int64(N);
    k_phase = 2*pi*double(p_irrep)/double(N);

    % Build a state -> rep-index lookup (rep states only).
    assert(n_total <= 2^32, ...
        'sigma_action_pg: n_total too large for dense lookup.');
    lookup = zeros(double(n_total), 1, 'int32');
    lookup(double(reps) + 1) = int32(1:dim);

    % Vectorized: apply sigma to all reps, then find orbit minima.
    sigma_states = apply_sigma_ring(reps, N, d_loc);

    d_loc_i64 = int64(d_loc);
    d_top     = int64(d_loc)^int64(N-1);
    rep_arr   = sigma_states;
    h_arr     = zeros(dim, 1, 'int32');
    n_cur     = sigma_states;

    for g = 1 : N - 1
        n_cur = idivide(n_cur, d_loc_i64) + mod(n_cur, d_loc_i64) * d_top;
        new_min = n_cur < rep_arr;
        if any(new_min)
            rep_arr(new_min) = n_cur(new_min);
            h_arr(new_min)   = int32(g);
        end
    end

    sigma_R_idx = lookup(double(rep_arr) + 1);
    assert(all(sigma_R_idx > 0), ...
        'sigma_action_pg: reflected rep not found in rep basis (sector inconsistency?).');

    h_sigma        = h_arr;
    lambda_sigma   = exp(-1i * k_phase * double(h_arr));
    is_self_stable = (sigma_R_idx == int32((1:dim)'));
end
