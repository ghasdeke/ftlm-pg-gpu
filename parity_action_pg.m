function [P_idx, P_phase, D_plus, D_minus] = parity_action_pg( ...
                                                 reps, orbit_lens, N, s_val, p_irrep)
%PARITY_ACTION_PG  Action of spin-inversion P on a (M=0, p) representative basis.
%
%   [P_IDX, P_PHASE, D_PLUS, D_MINUS] = PARITY_ACTION_PG(REPS, ORBIT_LENS, ...
%                                            N, S_VAL, P_IRREP)
%
%   computes how the spin-inversion operator P (which sends m_i to -m_i,
%   equivalently maps the integer state n to (d_loc^N - 1) - n) acts on
%   the symmetry-adapted basis {|reps(t), p>} of an (M = 0, p)-sector.
%
%   For each rep at index t, P sends the symmetry-adapted state to a
%   complex multiple of another rep:
%       P |reps(t), p> = P_phase(t) * |reps(P_idx(t)), p>
%   with
%       P_phase(t) = exp(-1i * k * h_min(P reps(t)))
%   where k = 2*pi*p/N and h_min is the translation that brings P reps(t)
%   to its orbit minimum.
%
%   Because [P, T] = 0 and [P, H] = 0 for an isotropic Heisenberg
%   Hamiltonian, P is Hermitian and squares to identity on each (M=0, p)
%   block. The parity dimensions
%       D_plus  = (dim + Tr(P)) / 2
%       D_minus = (dim - Tr(P)) / 2
%   give the +1 and -1 parity subspaces. Tr(P) is recovered as a sum of
%   per-rep contributions: only reps whose C_N-orbit is parity-stable
%   (P_idx(t) == t) contribute, with phase factor +1 (for the rare case
%   P(r) = r exactly) or (-1)^p (for the case P(r) = T^{N/2}(r) on even
%   rings).
%
%   Inputs:
%       reps         sorted column of orbit representatives (int64)
%       orbit_lens   orbit length per rep (int32)
%       N            number of sites
%       s_val        local spin
%       p_irrep      irrep index in [0, N-1]
%
%   Outputs:
%       P_idx        target rep index per input rep (int32)
%       P_phase      complex phase per rep (complex double, size dim x 1)
%       D_plus       dim of parity-even subspace
%       D_minus      dim of parity-odd subspace
%
%   Restriction: meaningful only for the M = 0 sector. P maps M != 0 to
%   -M, which is a different sector.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    d_loc   = round(2*s_val + 1);
    dim     = length(reps);
    n_total = int64(d_loc)^int64(N);
    k_phase = 2*pi*double(p_irrep)/double(N);

    % State -> rep-index lookup over the rep basis only.
    % Validity of every parity image is guaranteed because P preserves
    % the (M = 0) sector and commutes with translation.
    assert(n_total <= 2^32, ...
        'parity_action_pg: n_total too large for dense lookup.');
    lookup = zeros(double(n_total), 1, 'int32');
    lookup(double(reps) + 1) = int32(1:dim);

    % Vectorized parity-image search: same structure as the orbit min in
    % enumerate_sector_with_translation, applied to the array Pr_arr =
    % n_total - 1 - reps. Brings the per-sector cost from O(dim * N)
    % scalar MATLAB calls down to N - 1 vectorized int64 passes.
    d_loc_i64 = int64(d_loc);
    d_top     = int64(d_loc)^int64(N-1);
    Pr_arr    = n_total - int64(1) - reps;

    rep_arr = Pr_arr;
    h_arr   = zeros(dim, 1, 'int32');
    n_cur   = Pr_arr;
    for g = 1 : N - 1
        n_cur = idivide(n_cur, d_loc_i64) + mod(n_cur, d_loc_i64) * d_top;
        new_min = n_cur < rep_arr;
        if any(new_min)
            rep_arr(new_min) = n_cur(new_min);
            h_arr(new_min)   = int32(g);
        end
    end

    P_idx   = lookup(double(rep_arr) + 1);
    assert(all(P_idx > 0), ...
        'parity_action_pg: parity image not found in rep basis (sector inconsistency?).');

    h_d     = double(h_arr);
    P_phase = exp(-1i * k_phase * h_d);

    % Tr(P) is the sum of phases on self-stable reps (P_idx == t). Phases
    % on self-stable reps are ±1 in exact arithmetic.
    self_stable  = (P_idx == int32((1:dim)'));
    trace_P_real = sum(real(P_phase(self_stable)));
    trace_P_int  = round(trace_P_real);

    D_plus  = (dim + trace_P_int) / 2;
    D_minus = (dim - trace_P_int) / 2;
    assert(D_plus + D_minus == dim && D_plus >= 0 && D_minus >= 0, ...
        'parity_action_pg: invalid D_plus/D_minus = %d/%d (dim = %d).', ...
        D_plus, D_minus, dim);
end
