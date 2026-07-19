function Y = spmv_pg_Ih_matlab(super_reps, V_per_rep, eig_per_rep, ...
                                n_per_rep, bonds, s_val, J, ...
                                irrep_data, d_irrep, group, X)
%SPMV_PG_IH_MATLAB  Matrix-free SpMV on the I_h (M, Gamma) basis.
%
%   Y = SPMV_PG_IH_MATLAB(SUPER_REPS, V_PER_REP, EIG_PER_REP, N_PER_REP, ...
%                          BONDS, S_VAL, J, IRREP_DATA, D_IRREP, GROUP, X)
%
%   computes Y = H_block * X for the symmetry-adapted Heisenberg
%   Hamiltonian on a single (M, Gamma) block of the icosahedron under
%   I_h, using exactly the same per-output-block logic that the CUDA
%   kernel CUDA_LANCZOS_CLUT_BLOCK_PG_IH.CU will execute on the GPU.
%   This serves as a CPU-side reference used by TEST_MILESTONE_G_SPMV_IH
%   to verify the kernel logic without touching the GPU.
%
%   Per source rep i, the routine:
%       1. Adds the diagonal S^z * S^z contribution to the n_Gamma(i)
%          rows owned by rep i. The diagonal is identical for every
%          partner index k of the same rep, because H_diag commutes
%          with the group.
%       2. For each bond and each of {S+_i S-_j, S-_i S+_j}:
%             a. Forms the spin-flipped state state_a.
%             b. Calls min_image_Ih to obtain (rep_a, g_min) with
%                g_min(state_a) = rep_a. Skips if rep_a is not in the
%                (M, Gamma) basis.
%             c. Forms the small d_Gamma x d_Gamma matrix-element
%                kernel
%                    M_block = sqrt(lambda_a / lambda_r) * c_a *
%                              (V_a^dag * rho_Gamma(g_a)^T * V_r),
%                with rho_Gamma(g_a)^T = conj(rho_Gamma(g_min)) for
%                unitary irreps. Then SCATTERS
%                    Y(off_a + 1 : off_a + n_a, :) += M_block * X_r.
%
%   Convention: this is the "scatter from source rep" pattern. It is
%   the natural form of the matrix-element formula and the form used
%   by BUILD_HEISENBERG_SPARSE_IH_GAMMA2 internally. The CUDA port
%   will likely transpose this to a gather pattern (each thread owns
%   one output row) to avoid atomic adds; the gather kernel can be
%   verified bit-for-bit against this scatter reference.
%
%   Block-mode: X may be supplied as [n_basis x B] for block-Lanczos;
%   the SpMV broadcasts the matrix-vector kernel across the B columns.
%
%   Inputs:
%       super_reps    sorted column of orbit minima with n_Gamma > 0
%                     (int64), from ENUMERATE_SECTOR_WITH_IH_GAMMA2.
%       V_per_rep     cell array of d_irrep x n_Gamma(i) eigenvector
%                     matrices.
%       eig_per_rep   cell array of n_Gamma(i)-vectors of eigenvalues.
%       n_per_rep     int32 column, n_per_rep(i) = n_Gamma(super_reps(i)).
%       bonds         [30 x 2] 1-based vertex pairs.
%       s_val         local spin.
%       J             Heisenberg coupling.
%       irrep_data    1D characters [120 x 1] or [d x d x 120] irrep
%                     matrix array.
%       d_irrep       irrep dimension (1..5 for I_h).
%       group         struct from ICOSAHEDRON_IH_FULL.
%       X             [n_basis x B] input block (real or complex).
%
%   Output:
%       Y             same shape as X, of the same complexity as X.
%
%   See also BUILD_HEISENBERG_SPARSE_IH_GAMMA2,
%            ENUMERATE_SECTOR_WITH_IH_GAMMA2, MIN_IMAGE_IH,
%            SPMV_PG_MATLAB, SPMV_PG_CPLX_MATLAB.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    d_loc   = round(2*s_val + 1);
    N_sites = 12;
    n_total = double(d_loc)^N_sites;
    n_reps  = numel(super_reps);
    n_b     = size(bonds, 1);
    powers  = int64(d_loc) .^ int64((0 : N_sites - 1)');

    %% rep_offsets and n_basis
    rep_offsets = zeros(n_reps, 1, 'int32');
    n_basis = int32(0);
    for i = 1 : n_reps
        rep_offsets(i) = n_basis;
        n_basis = n_basis + n_per_rep(i);
    end
    n_basis = double(n_basis);

    if size(X, 1) ~= n_basis
        error('spmv_pg_Ih_matlab:DimMismatch', ...
            'X has %d rows, expected n_basis = %d.', size(X, 1), n_basis);
    end

    %% State -> rep-index lookup (1-based; 0 = not in basis)
    assert(n_total <= 2^32, ...
        'spmv_pg_Ih_matlab: n_total too large for dense lookup.');
    lookup = zeros(n_total, 1, 'int32');
    lookup(double(super_reps) + 1) = int32(1 : n_reps);

    %% Digit decomposition of each super-rep
    mi = zeros(n_reps, N_sites);
    tmp = super_reps;
    for site = 1 : N_sites
        dg = double(mod(tmp, int64(d_loc)));
        mi(:, site) = dg - s_val;
        tmp = (tmp - int64(dg)) / int64(d_loc);
    end

    %% Precompute sqrt(eigvals) per rep
    sqrt_eig_cache = cell(n_reps, 1);
    for i = 1 : n_reps
        sqrt_eig_cache{i} = sqrt(eig_per_rep{i});
    end

    %% Output Y. Preserve complex / real class of X (the SpMV writes
    %  complex values for higher-d irreps even if X is real, so we
    %  promote to complex when irrep is genuinely complex). We detect
    %  the complex case by checking if any rho_Gamma(g) has non-trivial
    %  imag part.
    % Real iff the irrep matrices themselves are real (realified FS=+1 irreps,
    % or 1D real characters). Do NOT key this off d_irrep > 1: a realified d>1
    % irrep (e.g. square C_4v) yields a genuinely real block.
    irrep_is_complex = ~isreal(irrep_data);
    if irrep_is_complex
        Y = complex(zeros(size(X)));
    else
        Y = zeros(size(X), 'like', X);
    end

    %% Diagonal: J * sum m_a * m_b applied to each rep's sub-block
    for i = 1 : n_reps
        diag_i = 0.0;
        for b = 1 : n_b
            diag_i = diag_i + J * mi(i, bonds(b, 1)) * mi(i, bonds(b, 2));
        end
        if diag_i ~= 0
            off_i = double(rep_offsets(i));
            n_i = double(n_per_rep(i));
            Y(off_i+1 : off_i+n_i, :) = Y(off_i+1 : off_i+n_i, :) ...
                                       + diag_i * X(off_i+1 : off_i+n_i, :);
        end
    end

    %% Off-diagonal: SCATTER from source reps via spin flips
    for i = 1 : n_reps
        r = super_reps(i);
        V_r = V_per_rep{i};
        sqrt_eig_r = sqrt_eig_cache{i};
        off_i = double(rep_offsets(i));
        n_i = double(n_per_rep(i));
        X_r = X(off_i+1 : off_i+n_i, :);

        for b = 1 : n_b
            si = bonds(b, 1); sj = bonds(b, 2);
            m_si = mi(i, si); m_sj = mi(i, sj);

            % S+_i S-_j
            if m_si < s_val - 1e-10 && m_sj > -s_val + 1e-10
                c_a = 0.5 * J ...
                    * sqrt(s_val*(s_val+1) - m_si*(m_si+1)) ...
                    * sqrt(s_val*(s_val+1) - m_sj*(m_sj-1));
                state_a = r + powers(si) - powers(sj);
                Y = scatter_block(Y, X_r, state_a, c_a, V_r, sqrt_eig_r, ...
                                  lookup, V_per_rep, sqrt_eig_cache, ...
                                  n_per_rep, rep_offsets, ...
                                  irrep_data, d_irrep, group, s_val);
            end

            % S-_i S+_j
            if m_si > -s_val + 1e-10 && m_sj < s_val - 1e-10
                c_a = 0.5 * J ...
                    * sqrt(s_val*(s_val+1) - m_si*(m_si-1)) ...
                    * sqrt(s_val*(s_val+1) - m_sj*(m_sj+1));
                state_a = r - powers(si) + powers(sj);
                Y = scatter_block(Y, X_r, state_a, c_a, V_r, sqrt_eig_r, ...
                                  lookup, V_per_rep, sqrt_eig_cache, ...
                                  n_per_rep, rep_offsets, ...
                                  irrep_data, d_irrep, group, s_val);
            end
        end
    end
end


% ----------------------------------------------------------------
function Y = scatter_block(Y, X_r, state_a, c_a, V_r, sqrt_eig_r, ...
                            lookup, V_per_rep, sqrt_eig_cache, n_per_rep, ...
                            rep_offsets, irrep_data, d_irrep, group, s_val)
% Single spin-flip contribution: scatters c_a-weighted M_block onto Y.
    [rep_a, g_min] = min_image_Ih(state_a, group, s_val);
    j = lookup(double(rep_a) + 1);
    if j == 0
        return;       % target rep not in this (M, Gamma) basis
    end
    n_j = double(n_per_rep(j));
    if n_j == 0
        return;
    end

    V_a        = V_per_rep{j};
    sqrt_eig_a = sqrt_eig_cache{j};
    off_j      = double(rep_offsets(j));

    % rho_Gamma(g_a)^T = conj(rho_Gamma(g_min)) for unitary irreps.
    rho_T = conj(irrep_matrix(irrep_data, g_min, d_irrep));     % d x d
    inner = V_a' * rho_T * V_r;                                 % n_j x n_r
    M_block = c_a * inner .* (sqrt_eig_a * (1 ./ sqrt_eig_r.'));% n_j x n_r

    Y(off_j+1 : off_j+n_j, :) = Y(off_j+1 : off_j+n_j, :) + M_block * X_r;
end


% ----------------------------------------------------------------
function M = irrep_matrix(irrep_data, g, d)
    % Preserve the real type of realified irreps (FS=+1); complex irreps keep
    % the complex path. See REALIFY_IRREPS.
    if d == 1
        v = irrep_data(g);
    else
        v = irrep_data(:, :, g);
    end
    if isreal(irrep_data)
        M = v;
    else
        M = complex(v);
    end
end
