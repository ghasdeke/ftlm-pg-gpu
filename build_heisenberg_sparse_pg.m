function H = build_heisenberg_sparse_pg(reps, orbit_lens, bonds, ...
                                        s_val, J, N, p_irrep, n_total)
%BUILD_HEISENBERG_SPARSE_PG  Heisenberg H on the (M, k) representative basis.
%
%   H = BUILD_HEISENBERG_SPARSE_PG(REPS, ORBIT_LENS, BONDS, S_VAL, J, ...
%                                   N, P_IRREP, N_TOTAL)
%
%   builds the sparse Heisenberg matrix for an N-site spin-S_VAL ring on
%   the symmetry-adapted (total-S^z, momentum) basis. The translation
%   irrep is k = 2*pi*P_IRREP/N.
%
%   Inputs:
%       reps         sorted column of orbit representatives (int64)
%       orbit_lens   orbit length per rep (int32)
%       bonds        bond list [N_b x 2], 1-based site indices
%       s_val        local spin
%       J            Heisenberg coupling, H = J * sum_{<ij>} S_i . S_j
%       N            number of sites (ring length)
%       p_irrep      irrep index in [0, N-1]
%       n_total      total Hilbert-space dimension d_loc^N (for safety check)
%
%   Output:
%       H            sparse [dim x dim] matrix.
%                    Real symmetric if p_irrep == 0 or 2*p_irrep == N;
%                    Hermitian complex otherwise.
%
%   Matrix element formula (off-diagonal):
%       M(r, a_R) = sqrt(L_r / L_{a_R}) * c_a * exp(-1i * k * h_minimg)
%   where:
%       a       = state generated from r by one bond spin-flip
%       c_a     = ladder-operator coefficient (S+_i S-_j or S-_i S+_j on r)
%       a_R     = orbit minimum of a (from min_image_ring)
%       L_{a_R} = orbit length of a_R
%       h_minimg = translation with T^{h_minimg}(a) = a_R
%
%   Sign convention: min_image_ring returns h such that T^h(a) = a_R, i.e.,
%   a = T^{N - h}(a_R). The exponent in M is +i*k*(N - h_minimg) =
%   -i*k*h_minimg (mod 2*pi, using k*N = 2*pi*P_IRREP), hence the negative
%   sign in the formula above.
%
%   Diagonal contribution is unchanged because [H_diag, T] = 0.
%
%   See also MIN_IMAGE_RING, ENUMERATE_SECTOR_WITH_TRANSLATION.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    d_loc   = round(2*s_val + 1);
    dim     = length(reps);
    n_bonds = size(bonds, 1);
    powers  = int64(d_loc) .^ int64((0:N-1)');
    k_phase = 2*pi*double(p_irrep)/double(N);
    is_real = (p_irrep == 0) || (2*p_irrep == N);

    if dim == 0
        if is_real
            H = sparse([], [], [], 0, 0);
        else
            H = sparse([], [], complex([]), 0, 0);
        end
        return;
    end

    %% Full state-to-rep-index lookup (rep states only).
    %  Valid only when n_total is small enough to allocate. For the
    %  Milestone A tests on small rings this is fine; production paths
    %  use the CLT bitmap instead.
    assert(n_total <= 2^32, ...
        'n_total = %d too large for dense lookup in build_heisenberg_sparse_pg.', n_total);
    lookup = zeros(double(n_total), 1, 'int32');
    lookup(double(reps) + 1) = int32(1:dim);

    %% Digit decomposition for each rep.
    mi = zeros(dim, N);
    tmp = reps;
    for site = 1 : N
        dg = double(mod(tmp, int64(d_loc)));
        mi(:, site) = dg - s_val;
        tmp = (tmp - int64(dg)) / int64(d_loc);
    end

    %% Pre-allocate row/col/val buffers.
    %  Upper bound: dim diagonal entries + 2*N_b*dim off-diagonal entries.
    cap      = dim * (1 + 2*n_bonds);
    row_list = zeros(cap, 1);
    col_list = zeros(cap, 1);
    if is_real
        val_list = zeros(cap, 1);
    else
        val_list = complex(zeros(cap, 1));
    end
    nz = 0;

    %% Diagonal: J * sum_{<i,j>} m_i * m_j.
    diag_vals = zeros(dim, 1);
    for b = 1 : n_bonds
        diag_vals = diag_vals + J * mi(:, bonds(b,1)) .* mi(:, bonds(b,2));
    end
    row_list(1:dim) = (1:dim)';
    col_list(1:dim) = (1:dim)';
    val_list(1:dim) = diag_vals;
    nz = dim;

    %% Off-diagonal: per bond, S+_i S-_j and S-_i S+_j contributions.
    for t = 1 : dim
        r   = reps(t);
        L_r = double(orbit_lens(t));
        for b = 1 : n_bonds
            si = bonds(b,1); sj = bonds(b,2);
            m_si = mi(t, si); m_sj = mi(t, sj);

            % --- S+_i S-_j ------------------------------------------
            if m_si < s_val - 1e-10 && m_sj > -s_val + 1e-10
                c_a = 0.5 * J ...
                    * sqrt(s_val*(s_val+1) - m_si*(m_si+1)) ...
                    * sqrt(s_val*(s_val+1) - m_sj*(m_sj-1));
                state_a = r + powers(si) - powers(sj);
                [rep_a, h_minimg, L_a] = min_image_ring(state_a, N, d_loc);
                idx_a = lookup(double(rep_a) + 1);
                if idx_a > 0
                    norm_factor = sqrt(L_r / double(L_a));
                    if is_real
                        phase = cos(k_phase * double(h_minimg));
                    else
                        phase = exp(-1i * k_phase * double(h_minimg));
                    end
                    nz = nz + 1;
                    row_list(nz) = double(idx_a);
                    col_list(nz) = t;
                    val_list(nz) = c_a * norm_factor * phase;
                end
            end

            % --- S-_i S+_j ------------------------------------------
            if m_si > -s_val + 1e-10 && m_sj < s_val - 1e-10
                c_a = 0.5 * J ...
                    * sqrt(s_val*(s_val+1) - m_si*(m_si-1)) ...
                    * sqrt(s_val*(s_val+1) - m_sj*(m_sj+1));
                state_a = r - powers(si) + powers(sj);
                [rep_a, h_minimg, L_a] = min_image_ring(state_a, N, d_loc);
                idx_a = lookup(double(rep_a) + 1);
                if idx_a > 0
                    norm_factor = sqrt(L_r / double(L_a));
                    if is_real
                        phase = cos(k_phase * double(h_minimg));
                    else
                        phase = exp(-1i * k_phase * double(h_minimg));
                    end
                    nz = nz + 1;
                    row_list(nz) = double(idx_a);
                    col_list(nz) = t;
                    val_list(nz) = c_a * norm_factor * phase;
                end
            end
        end
    end

    H = sparse(row_list(1:nz), col_list(1:nz), val_list(1:nz), dim, dim);
end
