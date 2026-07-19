function [reps, orbit_lens, dim] = enumerate_sector_with_translation( ...
                                       N, s_val, M_target, p_irrep)
%ENUMERATE_SECTOR_WITH_TRANSLATION  Basis of the (M, k) sector on a ring.
%
%   [REPS, ORBIT_LENS, DIM] = ENUMERATE_SECTOR_WITH_TRANSLATION( ...
%       N, S_VAL, M_TARGET, P_IRREP)
%
%   enumerates orbit representatives in the simultaneous (total-S^z,
%   translation-irrep) sector of an N-site ring of local spin S_VAL with
%   cyclic translation symmetry C_N. The momentum quantum number is
%   k = 2*pi*P_IRREP/N with P_IRREP in {0, 1, ..., N-1}.
%
%   Inputs:
%       N         number of sites (>= 2)
%       s_val     local spin (0.5, 1, 1.5, ...)
%       M_target  total S^z (integer or half-integer)
%       p_irrep   irrep index, integer in [0, N-1]
%
%   Outputs:
%       reps        sorted column vector of representative state integers
%                   (int64; smallest integer in each orbit)
%       orbit_lens  column vector (int32) of orbit lengths L_r per rep
%       dim         dim(REPS) (i.e., dimension of the (M, k) sector)
%
%   Algorithm:
%       1. Enumerate all states with total S^z = M_target (chunked digit
%          decomposition; identical convention to release/ftlm_observables.m).
%       2. For each such state, find its orbit minimum and orbit length
%          via cyclic-shift iteration (min_image_ring). Keep only states
%          that are their own minimum.
%       3. Apply the C_N compatibility filter: a rep with orbit length
%          L_r contributes to irrep p iff p*L_r mod N == 0.
%
%   The returned REPS array is sorted in ascending integer order, which
%   is required by the compressed lookup table (CLT) construction.
%
%   See also MIN_IMAGE_RING, BUILD_HEISENBERG_SPARSE_PG.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    assert(N >= 2, 'N must be at least 2.');
    assert(s_val > 0 && abs(2*s_val - round(2*s_val)) < 1e-12, ...
        's_val must be a positive half-integer.');
    assert(p_irrep >= 0 && p_irrep < N && p_irrep == round(p_irrep), ...
        'p_irrep must be an integer in [0, N-1].');

    d_loc = round(2*s_val + 1);
    n_total = int64(d_loc)^int64(N);

    %% Step 1: enumerate the M-sector basis (digit decomposition)
    basis_M = enumerate_M_basis(N, s_val, d_loc, n_total, M_target);
    if isempty(basis_M)
        reps = int64([]); orbit_lens = int32([]); dim = 0;
        return;
    end

    %% Step 2: keep only orbit minima, record orbit lengths (vectorized).
    %  Same logic as a scalar loop calling min_image_ring(basis_M(i), N, d_loc),
    %  but done as N-1 vectorized cyclic shifts over the full basis_M array
    %  with int64 arithmetic. Brings the enumeration from minutes (scalar)
    %  down to under a second for N = 24, s = 1/2 (basis_M ~ 2.7M states).
    %
    %  Algorithm:
    %     rep_arr     : running min of orbit; starts at basis_M
    %     returned_at : first g >= 1 with T^g(state) == state; 0 if not yet
    %     after the loop, L = returned_at (clamped to N if never returned).
    n_states  = length(basis_M);
    d_loc_i64 = int64(d_loc);
    d_top     = int64(d_loc)^int64(N-1);

    rep_arr     = basis_M;
    returned_at = zeros(n_states, 1, 'int32');
    n_cur       = basis_M;

    for g = 1 : N - 1
        n_cur = idivide(n_cur, d_loc_i64) + mod(n_cur, d_loc_i64) * d_top;
        just_returned = (returned_at == 0) & (n_cur == basis_M);
        if any(just_returned)
            returned_at(just_returned) = int32(g);
        end
        new_min = n_cur < rep_arr;
        if any(new_min)
            rep_arr(new_min) = n_cur(new_min);
        end
    end

    L_arr = returned_at;
    L_arr(L_arr == 0) = int32(N);

    is_rep   = rep_arr == basis_M;
    reps_all = basis_M(is_rep);
    L_all    = L_arr(is_rep);

    %% Step 3: C_N compatibility filter
    %  A rep with orbit length L_r contributes to irrep p iff p*L_r mod N == 0.
    %  (Otherwise the symmetry-adapted vector vanishes identically.)
    p64    = int64(p_irrep);
    N64    = int64(N);
    compat = mod(p64 .* int64(L_all), N64) == 0;

    reps       = reps_all(compat);
    orbit_lens = L_all(compat);
    dim        = length(reps);
end

% ----------------------------------------------------------------
% Local helper: M-sector basis enumeration (mirrors release/
% ftlm_observables.m, kept self-contained for the mit_pg branch).
% ----------------------------------------------------------------
function basis = enumerate_M_basis(N, s_val, d_loc, n_total, M_target)
    if s_val == 0.5
        % Vectorized popcount path for spin-1/2.
        k = N/2 + M_target;
        if k < 0 || k > N || abs(k - round(k)) > 0.01
            basis = int64([]); return;
        end
        k = round(k);
        basis = enumerate_half_spin_basis(N, k);
    else
        % Generic radix-d digit enumeration in chunks.
        chunk = 1e7;
        parts = {};
        for start = int64(0) : int64(chunk) : n_total - 1
            stop   = min(start + int64(chunk) - 1, n_total - 1);
            states = (start:stop)';
            Mv  = zeros(length(states), 1);
            tmp = states;
            for kk = 1 : N
                dg  = double(mod(tmp, int64(d_loc)));
                Mv  = Mv + dg - s_val;
                tmp = (tmp - int64(dg)) / int64(d_loc);
            end
            Mv  = round(Mv * 2) / 2;
            sel = (Mv == M_target);
            if any(sel)
                parts{end+1} = states(sel); %#ok<AGROW>
            end
        end
        if isempty(parts)
            basis = int64([]);
        else
            basis = cat(1, parts{:});
        end
    end
end

function basis = enumerate_half_spin_basis(N, k)
%ENUMERATE_HALF_SPIN_BASIS  All N-bit integers with exactly k bits set.
%  Vectorized byte-LUT popcount; identical convention to release path.
    n_total = int64(2)^N;

    lut = uint8(zeros(256, 1));
    for i = 0 : 255
        lut(i+1) = uint8(sum(bitget(uint8(i), 1:8)));
    end

    dim_expected = nchoosek(N, k);
    basis        = zeros(dim_expected, 1, 'int64');
    idx          = 0;
    chunk_size   = int64(2^22);

    for cs = int64(0) : chunk_size : (n_total - 1)
        ce    = min(cs + chunk_size - 1, n_total - 1);
        range = (cs:ce)';

        b0 = uint8(bitand(range,                     int64(255)));
        b1 = uint8(bitand(bitshift(range,  -8),      int64(255)));
        b2 = uint8(bitand(bitshift(range, -16),      int64(255)));
        b3 = uint8(bitand(bitshift(range, -24),      int64(255)));

        pop = lut(double(b0)+1) + lut(double(b1)+1) + ...
              lut(double(b2)+1) + lut(double(b3)+1);

        valid   = pop == uint8(k);
        n_valid = sum(valid);
        if n_valid > 0
            basis(idx+1 : idx+n_valid) = range(valid);
            idx = idx + n_valid;
        end
    end
    basis = basis(1:idx);
end
