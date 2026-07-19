function [E_sec, w_sec, B_used] = run_ftlm_pg_sector_gpu_cplx( ...
    reps, orbit_lens, bonds, s_val, J, N, p_irrep, ...
    dim_sector, R, M_lz, B_gpu, L2_cache_bytes, gpu_h, parity)
%RUN_FTLM_PG_SECTOR_GPU_CPLX  Block-Lanczos FTLM on the GPU for complex p.
%
%   Complex-arithmetic GPU FTLM in one (M, p) sector, for any p in
%   [0, N-1]. Sectors with real H (p = 0 or 2*p = N) can be handled by
%   this kernel too, but the dedicated real kernel
%   RUN_FTLM_PG_SECTOR_GPU is faster and is the recommended path in
%   FTLM_OBSERVABLES_PG_GPU for the real cases.
%
%   The two real working buffers (V_re and V_im) are dispatched as
%   separate gpuArrays of class single. The Lanczos tridiagonal returned
%   is real, because for Hermitian H the diagonal coefficient
%   alpha = Re[<v|H|v>] is real and the subdiagonal beta = ||w|| is real
%   positive.
%
%   Inputs:
%       reps, orbit_lens   from enumerate_sector_with_translation
%       bonds, s_val, J, N from the Hamiltonian definition
%       p_irrep            irrep index for k = 2*pi*p/N
%       dim_sector         = length(reps)
%       R                  number of FTLM random vectors
%       M_lz               Lanczos steps per random vector
%       B_gpu              block size (0 = adaptive L2 heuristic)
%       L2_cache_bytes     used by the adaptive heuristic
%       gpu_h              gpuDevice handle
%
%   Outputs:
%       E_sec, w_sec, B_used  as in run_ftlm_pg_sector_gpu.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    assert(exist('cuda_lanczos_clut_block_pg_cplx', 'file') == 3, ...
        'MEX file cuda_lanczos_clut_block_pg_cplx not found. Run build_pg_kernels first.');

    if nargin < 14, parity = []; end
    if isempty(parity)
        parity_on = false;
        D_eff = dim_sector;
    else
        parity_on = true;
        D_eff     = parity.D_sigma;
        sigma     = parity.sigma;
        P_idx     = parity.P_idx;
        P_phase   = parity.P_phase;
        if D_eff <= 0
            E_sec = []; w_sec = []; B_used = 0; return;
        end
    end

    R_eff       = min(R, D_eff);
    M_lz_actual = min(M_lz, D_eff);

    % Complex case has 2 vectors per Krylov state -> twice the cache footprint.
    if B_gpu == 0
        mem_B8 = 3 * 2 * dim_sector * 8 * 4;   % 6 vectors x B=8 x FP32
        if mem_B8 <= L2_cache_bytes
            B_used = 8;
        else
            B_used = 4;
        end
    else
        B_used = B_gpu;
    end
    B_used = min(B_used, R_eff);

    n_total = double(round(2*s_val + 1))^N;
    [block_base, block_mask] = build_CLT_rep(reps, n_total);
    basis_int  = int32(reps);
    orbit_int  = int32(orbit_lens);
    bonds_flat = int32(reshape(bonds' - 1, [], 1));

    cuda_lanczos_clut_block_pg_cplx('init', ...
        gpuArray(block_base), gpuArray(block_mask), ...
        gpuArray(basis_int), gpuArray(orbit_int), ...
        bonds_flat, N, round(2*s_val + 1), s_val, J, ...
        dim_sector, B_used, p_irrep);
    wait(gpu_h);

    E_sec = zeros(M_lz_actual * R_eff, 1);
    w_sec = zeros(M_lz_actual * R_eff, 1);
    idx   = 0;

    if parity_on
        rng(double(dim_sector) + 100*double(p_irrep) + 1e6 * (sigma > 0));
    else
        rng(double(dim_sector) + 100*double(p_irrep));
    end

    for r_start = 1 : B_used : R_eff
        r_end = min(r_start + B_used - 1, R_eff);
        B_cur = r_end - r_start + 1;

        V0_re_d = double(randn(dim_sector, B_cur));
        V0_im_d = double(randn(dim_sector, B_cur));
        if parity_on
            V0_full = V0_re_d + 1i * V0_im_d;
            PV0     = apply_parity_pg(V0_full, P_idx, P_phase);
            V0_full = (V0_full + sigma * PV0) / sqrt(2);
            V0_re_d = real(V0_full);
            V0_im_d = imag(V0_full);
        end
        % Skip columns whose parity projection collapsed to zero.
        col_nrm = sqrt(sum(V0_re_d.^2 + V0_im_d.^2, 1));
        good_cols = col_nrm > 1e-10;
        if ~all(good_cols)
            V0_re_d = V0_re_d(:, good_cols);
            V0_im_d = V0_im_d(:, good_cols);
            B_cur   = size(V0_re_d, 2);
            if B_cur == 0, continue; end
        end
        V0_re = single(V0_re_d);
        V0_im = single(V0_im_d);

        [AL, BE] = cuda_lanczos_clut_block_pg_cplx('block_lanczos', ...
                       gpuArray(V0_re), gpuArray(V0_im), M_lz_actual);

        for b = 1 : B_cur
            [E_r, q1_r] = solve_tridiag(AL(:, b), BE(:, b));
            n_l = numel(E_r);
            E_sec(idx+1 : idx+n_l) = E_r;
            % FTLM single-vector weight uses D_eff = D_sigma (parity on)
            % or D_eff = dim_sector (parity off).
            w_sec(idx+1 : idx+n_l) = (D_eff / R_eff) * q1_r;
            idx = idx + n_l;
        end
    end

    cuda_lanczos_clut_block_pg_cplx('cleanup');

    E_sec = E_sec(1:idx);
    w_sec = w_sec(1:idx);
end

% ----------------------------------------------------------------
function [block_base, block_mask] = build_CLT_rep(reps, n_total)
    BLOCK_SIZE = 32;
    n_blocks   = ceil(n_total / BLOCK_SIZE);
    states     = double(reps(:));
    blks = floor(states / BLOCK_SIZE) + 1;
    bits = mod(states, BLOCK_SIZE);
    block_base      = int32(-ones(n_blocks, 1));
    [ub, fi]        = unique(blks, 'first');
    block_base(ub)  = int32(fi - 1);
    bit_vals   = pow2(bits);
    mask_sums  = accumarray(blks, bit_vals, [n_blocks, 1]);
    block_mask = uint32(mask_sums);
end

function [ep, q1] = solve_tridiag(alpha, beta)
    alpha = double(alpha(:));
    beta  = double(beta(:));
    n     = length(alpha);
    T     = diag(alpha(1:n));
    if n > 1
        T = T + diag(beta(1:n-1), 1) + diag(beta(1:n-1), -1);
    end
    [Q, D] = eig(T, 'vector');
    ep = D;
    q1 = abs(Q(1, :)').^2;
end
