function [E_sec, w_sec, B_used] = run_ftlm_pg_sector_gpu( ...
    reps, orbit_lens, bonds, s_val, J, N, ...
    dim_sector, R, M_lz, B_gpu, L2_cache_bytes, gpu_h, parity)
%RUN_FTLM_PG_SECTOR_GPU  Block-Lanczos FTLM on the GPU for the p = 0 sector.
%
%   Drop-in replacement for run_ftlm_pg_sector (CPU FP64) restricted to
%   the trivial irrep p = 0 (real arithmetic, real FP32 vectors). Calls
%   the matrix-free CUDA kernel cuda_lanczos_clut_block_pg.
%
%   For non-trivial irreps (p != 0) use the CPU path until the complex
%   GPU kernel from Milestone C is available.
%
%   Inputs:
%       reps             sorted column of orbit representatives (int64)
%       orbit_lens       orbit length per rep (int32)
%       bonds            bond list [N_b x 2], 1-based site indices
%       s_val, J         Heisenberg parameters
%       N                ring length
%       dim_sector       length(reps)
%       R                number of FTLM random vectors per sector
%       M_lz             Lanczos iterations per random vector
%       B_gpu            block-Lanczos block size (0 -> adaptive L2 heuristic)
%       L2_cache_bytes   used by the B_gpu = 0 heuristic
%       gpu_h            gpuDevice handle (for wait())
%
%   Outputs:
%       E_sec, w_sec     Ritz values and FTLM weights, length R_eff*M_lz_actual
%       B_used           effective block size

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    assert(exist('cuda_lanczos_clut_block_pg', 'file') == 3, ...
        'MEX file cuda_lanczos_clut_block_pg not found. Run build_pg_kernels first.');

    if nargin < 13, parity = []; end
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

    % Adaptive block size B (matches release heuristic).
    if B_gpu == 0
        mem_B8 = 3 * dim_sector * 8 * 4;
        if mem_B8 <= L2_cache_bytes
            B_used = 8;
        else
            B_used = 4;
        end
    else
        B_used = B_gpu;
    end
    B_used = min(B_used, R_eff);

    %% Build CLT over the representative basis
    n_total = double(round(2*s_val + 1))^N;
    [block_base, block_mask] = build_CLT_rep(reps, n_total);
    basis_int  = int32(reps);
    orbit_int  = int32(orbit_lens);
    bonds_flat = int32(reshape(bonds' - 1, [], 1));

    cuda_lanczos_clut_block_pg('init', ...
        gpuArray(block_base), gpuArray(block_mask), ...
        gpuArray(basis_int), gpuArray(orbit_int), ...
        bonds_flat, N, round(2*s_val + 1), s_val, J, dim_sector, B_used);
    wait(gpu_h);

    E_sec = zeros(M_lz_actual * R_eff, 1);
    w_sec = zeros(M_lz_actual * R_eff, 1);
    idx   = 0;

    % Deterministic seed per (sector, sigma sub-sector). Identical to the
    % release convention rng(dim_sector) when parity is off; offset by
    % a large constant for the +1 sub-sector so the parity-projected
    % vectors are stochastically independent.
    if parity_on
        rng(double(dim_sector) + 1e6 * (sigma > 0));
    else
        rng(double(dim_sector));
    end

    for r_start = 1 : B_used : R_eff
        r_end = min(r_start + B_used - 1, R_eff);
        B_cur = r_end - r_start + 1;

        V0_blk = double(randn(dim_sector, B_cur));
        if parity_on
            PV0 = apply_parity_pg(V0_blk, P_idx, P_phase);
            V0_blk = (V0_blk + sigma * real(PV0)) / sqrt(2);
            % P_phase is real (= +/- 1) for the p = 0 real-arithmetic path;
            % the real() guards against tiny imaginary roundoff if the
            % caller ever passes a complex P_phase here. Norm > 0 check
            % below catches the degenerate-projection corner case.
        end
        V0_blk = single(V0_blk);
        % Skip columns whose parity projection collapsed to zero.
        col_nrm = sqrt(sum(V0_blk .* V0_blk, 1));
        good_cols = col_nrm > 1e-10;
        if ~all(good_cols)
            V0_blk = V0_blk(:, good_cols);
            B_cur  = size(V0_blk, 2);
            if B_cur == 0, continue; end
        end
        V0_gpu = gpuArray(V0_blk);

        [AL, BE] = cuda_lanczos_clut_block_pg('block_lanczos', V0_gpu, M_lz_actual);

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

    cuda_lanczos_clut_block_pg('cleanup');

    E_sec = E_sec(1:idx);
    w_sec = w_sec(1:idx);
end

% ----------------------------------------------------------------
function [block_base, block_mask] = build_CLT_rep(reps, n_total)
%BUILD_CLT_REP  Compressed lookup table over the rep basis.
%  Identical to release/ftlm_observables.m build_CLT, just renamed.

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
