function H = build_heisenberg_sparse_Ih(super_reps, orbit_lens, bonds, ...
                                          s_val, J, irrep_chars, group)
%BUILD_HEISENBERG_SPARSE_IH  Sparse Heisenberg H on the I_h (M, Gamma) basis.
%
%   H = BUILD_HEISENBERG_SPARSE_IH(SUPER_REPS, ORBIT_LENS, BONDS, ...
%                                   S_VAL, J, IRREP_CHARS, GROUP)
%
%   builds the Heisenberg Hamiltonian on the 1D-irrep symmetry-adapted
%   basis under I_h on the icosahedron. The matrix element formula is
%       M(r', r) = sqrt(L_r/L_{r'}) * c_a * chi_Gamma(g_a)
%   where:
%       r is a source super-rep,
%       a is the spin-flipped state from r via a bond operator,
%       r' is the I_h-orbit minimum of a,
%       g_a is the group element with g_a(r') = a,
%       L_r, L_{r'} are I_h orbit lengths (always divisors of 120).
%
%   For 1D real characters (A_g and A_u) we have chi(g_a) =
%   chi(g_min^{-1}) = chi(g_min) where g_min is the min-image group
%   element from MIN_IMAGE_IH applied to the spin-flipped state.
%
%   Inputs:
%       super_reps    [dim x 1] int64, sorted orbit minima from
%                     ENUMERATE_SECTOR_WITH_IH.
%       orbit_lens    [dim x 1] int32, I_h orbit length per super-rep.
%       bonds         [30 x 2] int, 1-based vertex pairs from
%                     ADJACENCY_ICOSAHEDRON_IH.
%       s_val         local spin.
%       J             Heisenberg coupling.
%       irrep_chars   [120 x 1] real, Gamma's character per group element.
%       group         struct from ICOSAHEDRON_IH_FULL.
%
%   Output:
%       H             sparse [dim x dim], Hermitian (here real because
%                     A_g and A_u characters are real). Densify and
%                     diagonalize for the (M, Gamma)-sector spectrum.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    d_loc = round(2*s_val + 1);
    N_sites = 12;
    dim = numel(super_reps);
    n_b = size(bonds, 1);
    powers = int64(d_loc) .^ int64((0 : N_sites - 1)');

    if dim == 0
        H = sparse([], [], [], 0, 0); return;
    end

    %% State -> super-rep-idx lookup over the rep state space.
    n_total = double(d_loc)^N_sites;
    assert(n_total <= 2^32, ...
        'build_heisenberg_sparse_Ih: n_total too large for dense lookup.');
    lookup = zeros(n_total, 1, 'int32');
    lookup(double(super_reps) + 1) = int32(1 : dim);

    %% Digit decomposition of each super-rep.
    mi = zeros(dim, N_sites);
    tmp = super_reps;
    for site = 1 : N_sites
        dg = double(mod(tmp, int64(d_loc)));
        mi(:, site) = dg - s_val;
        tmp = (tmp - int64(dg)) / int64(d_loc);
    end

    %% Diagonal contribution (unchanged by symmetry adaptation).
    diag_vals = zeros(dim, 1);
    for b = 1 : n_b
        diag_vals = diag_vals + J * mi(:, bonds(b, 1)) .* mi(:, bonds(b, 2));
    end

    %% Off-diagonal contributions.
    cap = dim * (1 + 2 * n_b);
    rows = zeros(cap, 1);
    cols = zeros(cap, 1);
    vals = zeros(cap, 1);
    rows(1 : dim) = (1 : dim)';
    cols(1 : dim) = (1 : dim)';
    vals(1 : dim) = diag_vals;
    nz = dim;

    for t = 1 : dim
        r = super_reps(t);
        L_r = double(orbit_lens(t));

        for b = 1 : n_b
            si = bonds(b, 1); sj = bonds(b, 2);
            m_si = mi(t, si); m_sj = mi(t, sj);

            % S+_i S-_j
            if m_si < s_val - 1e-10 && m_sj > -s_val + 1e-10
                c_a = 0.5 * J ...
                    * sqrt(s_val*(s_val+1) - m_si*(m_si+1)) ...
                    * sqrt(s_val*(s_val+1) - m_sj*(m_sj-1));
                state_a = r + powers(si) - powers(sj);
                [rep_a, g_min] = min_image_Ih(state_a, group, s_val);
                idx_a = lookup(double(rep_a) + 1);
                if idx_a > 0
                    L_a = double(orbit_lens(idx_a));
                    norm = sqrt(L_r / L_a);
                    chi = irrep_chars(g_min);
                    nz = nz + 1;
                    rows(nz) = idx_a;
                    cols(nz) = t;
                    vals(nz) = c_a * norm * chi;
                end
            end

            % S-_i S+_j
            if m_si > -s_val + 1e-10 && m_sj < s_val - 1e-10
                c_a = 0.5 * J ...
                    * sqrt(s_val*(s_val+1) - m_si*(m_si-1)) ...
                    * sqrt(s_val*(s_val+1) - m_sj*(m_sj+1));
                state_a = r - powers(si) + powers(sj);
                [rep_a, g_min] = min_image_Ih(state_a, group, s_val);
                idx_a = lookup(double(rep_a) + 1);
                if idx_a > 0
                    L_a = double(orbit_lens(idx_a));
                    norm = sqrt(L_r / L_a);
                    chi = irrep_chars(g_min);
                    nz = nz + 1;
                    rows(nz) = idx_a;
                    cols(nz) = t;
                    vals(nz) = c_a * norm * chi;
                end
            end
        end
    end

    H = sparse(rows(1 : nz), cols(1 : nz), vals(1 : nz), dim, dim);
end
