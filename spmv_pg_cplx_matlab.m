function [W_re, W_im] = spmv_pg_cplx_matlab(reps, orbit_lens, bonds, ...
                                            s_val, J, N, p_irrep, V_re, V_im)
%SPMV_PG_CPLX_MATLAB  MATLAB reference of the complex per-thread PG-SpMV.
%
%   [W_RE, W_IM] = SPMV_PG_CPLX_MATLAB(REPS, ORBIT_LENS, BONDS, ...
%                                       S_VAL, J, N, P_IRREP, V_RE, V_IM)
%
%   computes the action of the symmetry-adapted Heisenberg Hamiltonian
%   for an arbitrary translation irrep p_irrep on a complex vector (or
%   block of complex vectors) V = V_RE + 1i*V_IM in the rep basis, using
%   exactly the same per-output-element logic that the CUDA kernel
%   cuda_lanczos_clut_block_pg_cplx.cu will execute on the GPU.
%
%   Inputs:
%       reps         sorted column of orbit representatives (int64)
%       orbit_lens   orbit length per rep (int32)
%       bonds        bond list [N_b x 2], 1-based site indices
%       s_val        local spin
%       J            Heisenberg coupling
%       N            number of sites (ring length)
%       p_irrep      irrep index 0..N-1; k = 2*pi*p_irrep/N
%       V_re, V_im   real and imaginary parts as [dim x B] real arrays
%
%   Outputs:
%       W_re, W_im   real and imaginary parts of W, same shape as V_re
%
%   The forward-gather convention used here implements
%       H[t, idx_a] = c_a * sqrt(L_r/L_a) * exp(+1i*k*h_min)
%   where h_min is the min-image translation T^{h_min}(state_a) = rep_a.
%   This is the Hermitian conjugate of the build-time convention used in
%   build_heisenberg_sparse_pg.m, which is necessary because the SpMV
%   gathers from inputs at the output thread, not scatters from a
%   source-state perspective.

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

    k_phase = 2*pi*double(p_irrep)/double(N);
    cos_kh  = cos(k_phase * (0 : N-1));
    sin_kh  = sin(k_phase * (0 : N-1));

    assert(n_total <= 2^32, ...
        'spmv_pg_cplx_matlab: n_total too large for dense lookup.');
    lookup = zeros(double(n_total), 1, 'int32');
    lookup(double(reps) + 1) = int32(1:dim);

    assert(size(V_re, 1) == dim && isequal(size(V_re), size(V_im)), ...
        'spmv_pg_cplx_matlab: V_re/V_im must be [dim x B] with matching shape.');

    mi = zeros(dim, N);
    tmp = reps;
    for site = 1 : N
        dg = double(mod(tmp, int64(d_loc)));
        mi(:, site) = dg - s_val;
        tmp = (tmp - int64(dg)) / int64(d_loc);
    end

    W_re = zeros(size(V_re));
    W_im = zeros(size(V_im));

    %% Diagonal (real, no phase)
    diag_vals = zeros(dim, 1);
    for b = 1 : n_b
        diag_vals = diag_vals + J * mi(:, bonds(b,1)) .* mi(:, bonds(b,2));
    end
    W_re = W_re + diag_vals .* V_re;
    W_im = W_im + diag_vals .* V_im;

    %% Off-diagonal with complex phase
    for t = 1 : dim
        state = reps(t);
        L_r   = double(orbit_lens(t));
        sqrt_L_r = sqrt(L_r);

        for b = 1 : n_b
            si = bonds(b, 1); sj = bonds(b, 2);
            m_si = mi(t, si); m_sj = mi(t, sj);

            % S+_i S-_j branch
            if m_si < s_val - 1e-10 && m_sj > -s_val + 1e-10
                c_a = 0.5 * J ...
                    * sqrt(s_val*(s_val+1) - m_si*(m_si+1)) ...
                    * sqrt(s_val*(s_val+1) - m_sj*(m_sj-1));
                state_a = state + powers(si) - powers(sj);
                [rep_a, h_min, L_a] = min_image_ring(state_a, N, d_loc);
                idx_a = lookup(double(rep_a) + 1);
                if idx_a > 0
                    norm_fac = sqrt_L_r / sqrt(double(L_a));
                    cos_phi = cos_kh(h_min + 1);
                    sin_phi = sin_kh(h_min + 1);
                    alpha_re = c_a * norm_fac * cos_phi;
                    alpha_im = c_a * norm_fac * sin_phi;
                    vr = V_re(idx_a, :);
                    vi = V_im(idx_a, :);
                    W_re(t, :) = W_re(t, :) + alpha_re * vr - alpha_im * vi;
                    W_im(t, :) = W_im(t, :) + alpha_re * vi + alpha_im * vr;
                end
            end

            % S-_i S+_j branch
            if m_si > -s_val + 1e-10 && m_sj < s_val - 1e-10
                c_a = 0.5 * J ...
                    * sqrt(s_val*(s_val+1) - m_si*(m_si-1)) ...
                    * sqrt(s_val*(s_val+1) - m_sj*(m_sj+1));
                state_a = state - powers(si) + powers(sj);
                [rep_a, h_min, L_a] = min_image_ring(state_a, N, d_loc);
                idx_a = lookup(double(rep_a) + 1);
                if idx_a > 0
                    norm_fac = sqrt_L_r / sqrt(double(L_a));
                    cos_phi = cos_kh(h_min + 1);
                    sin_phi = sin_kh(h_min + 1);
                    alpha_re = c_a * norm_fac * cos_phi;
                    alpha_im = c_a * norm_fac * sin_phi;
                    vr = V_re(idx_a, :);
                    vi = V_im(idx_a, :);
                    W_re(t, :) = W_re(t, :) + alpha_re * vr - alpha_im * vi;
                    W_im(t, :) = W_im(t, :) + alpha_re * vi + alpha_im * vr;
                end
            end
        end
    end
end
