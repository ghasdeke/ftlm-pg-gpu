function W = spmv_pg_matlab(reps, orbit_lens, bonds, s_val, J, N, V)
%SPMV_PG_MATLAB  MATLAB reference of the per-thread PG-SpMV (k = 0).
%
%   W = SPMV_PG_MATLAB(REPS, ORBIT_LENS, BONDS, S_VAL, J, N, V)
%
%   computes the action of the symmetry-adapted Heisenberg Hamiltonian
%   for the trivial irrep p = 0 on a vector (or block of vectors) V in
%   the rep basis, using exactly the same per-output-element logic that
%   the CUDA kernel `cuda_lanczos_clut_block_pg.cu` will execute on the
%   GPU. This serves as a CPU-side reference used by `test_milestone_B`
%   to verify the kernel logic without ever touching the GPU.
%
%   Per output rep t the routine:
%       1. Decomposes basis(t) = reps(t) into N base-d_loc digits.
%       2. Adds the diagonal contribution c_diag * V(t, :) (unchanged
%          on the rep basis because H_diag commutes with translation).
%       3. For each bond and each of {S+_i S-_j, S-_i S+_j}:
%             a. Forms the spin-flipped state state_a.
%             b. Cyclic-shifts state_a to its orbit minimum rep_a and
%                looks idx_a up in a state -> rep index map.
%             c. Multiplies the ladder coefficient by sqrt(L_r / L_a)
%                and accumulates into W(t, :).
%
%   Inputs:
%       reps         sorted column of orbit representatives (int64)
%       orbit_lens   orbit length per rep (int32)
%       bonds        bond list [N_b x 2], 1-based site indices
%       s_val        local spin
%       J            Heisenberg coupling
%       N            number of sites (ring length)
%       V            input vector [dim x 1] or block [dim x B] (real)
%
%   Output:
%       W            same shape as V, real
%
%   Convention: the dense state -> rep-index lookup is built explicitly
%   inside the function. For testing on small rings this is fine; the
%   GPU kernel uses the compressed CLT in its place.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    d_loc   = round(2*s_val + 1);
    dim     = length(reps);
    n_b     = size(bonds, 1);
    powers  = int64(d_loc) .^ int64((0:N-1)');
    n_total = int64(d_loc)^int64(N);

    % state -> rep-index lookup (1-based; 0 means not a rep).
    assert(n_total <= 2^32, ...
        'spmv_pg_matlab: n_total too large for dense lookup.');
    lookup = zeros(double(n_total), 1, 'int32');
    lookup(double(reps) + 1) = int32(1:dim);

    % Digit decomposition of all reps in one pass.
    mi = zeros(dim, N);
    tmp = reps;
    for site = 1 : N
        dg = double(mod(tmp, int64(d_loc)));
        mi(:, site) = dg - s_val;
        tmp = (tmp - int64(dg)) / int64(d_loc);
    end

    % V must be supplied as [dim x B] (B == 1 for a single vector).
    % We deliberately do NOT auto-reshape with V(:), because for dim = 1
    % MATLAB's isvector heuristic would mistake a legitimate [1 x B]
    % block input for a length-B vector and collapse the columns into
    % rows, corrupting block-mode results in dim-1 sectors.
    assert(size(V, 1) == dim, ...
        'spmv_pg_matlab: V has %d rows, expected dim = %d.', size(V, 1), dim);
    W = zeros(size(V), 'like', V);

    % --- Diagonal: J * sum m_i*m_j on the rep basis (factor 1) ---
    diag_vals = zeros(dim, 1);
    for b = 1 : n_b
        diag_vals = diag_vals + J * mi(:, bonds(b,1)) .* mi(:, bonds(b,2));
    end
    W = W + diag_vals .* V;     % broadcasts across B columns when B > 1

    % --- Off-diagonal: per rep, per bond, two ladder branches ---
    for t = 1 : dim
        state = reps(t);
        L_r   = double(orbit_lens(t));
        sqrt_L_r = sqrt(L_r);

        for b = 1 : n_b
            si = bonds(b, 1); sj = bonds(b, 2);
            m_si = mi(t, si); m_sj = mi(t, sj);

            % S+_i S-_j
            if m_si < s_val - 1e-10 && m_sj > -s_val + 1e-10
                c_a = 0.5 * J ...
                    * sqrt(s_val*(s_val+1) - m_si*(m_si+1)) ...
                    * sqrt(s_val*(s_val+1) - m_sj*(m_sj-1));
                state_a = state + powers(si) - powers(sj);
                [rep_a, ~, L_a] = min_image_ring(state_a, N, d_loc);
                idx_a = lookup(double(rep_a) + 1);
                if idx_a > 0
                    norm_factor = sqrt_L_r / sqrt(double(L_a));
                    total = c_a * norm_factor;
                    W(t, :) = W(t, :) + total * V(idx_a, :);
                end
            end

            % S-_i S+_j
            if m_si > -s_val + 1e-10 && m_sj < s_val - 1e-10
                c_a = 0.5 * J ...
                    * sqrt(s_val*(s_val+1) - m_si*(m_si-1)) ...
                    * sqrt(s_val*(s_val+1) - m_sj*(m_sj+1));
                state_a = state - powers(si) + powers(sj);
                [rep_a, ~, L_a] = min_image_ring(state_a, N, d_loc);
                idx_a = lookup(double(rep_a) + 1);
                if idx_a > 0
                    norm_factor = sqrt_L_r / sqrt(double(L_a));
                    total = c_a * norm_factor;
                    W(t, :) = W(t, :) + total * V(idx_a, :);
                end
            end
        end
    end
end
