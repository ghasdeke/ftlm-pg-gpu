function [E_sec, w_sec] = run_ftlm_pg_sector(H_pg, dim_sector, ...
                                              R, M_lz, is_real, sector_seed, parity)
%RUN_FTLM_PG_SECTOR  FTLM in one symmetry-adapted (M, p) sector.
%
%   [E_SEC, W_SEC] = RUN_FTLM_PG_SECTOR(H_PG, DIM_SECTOR, R, M_LZ, ...
%                                        IS_REAL, SECTOR_SEED)
%
%   runs R independent Lanczos chains on the supplied symmetry-adapted
%   Hamiltonian H_PG, each starting from a fresh random vector of the
%   appropriate type (real Gaussian if IS_REAL, complex Gaussian
%   otherwise), and returns the concatenated Ritz values and FTLM
%   weights for use by the aggregator.
%
%   Inputs:
%       H_pg          sparse Hermitian operator on the (M, p) basis
%                     (real if is_real, complex otherwise)
%       dim_sector    size(H_pg, 1)
%       R             number of FTLM random vectors (clamped to
%                     min(R, dim_sector) internally)
%       M_lz          number of Lanczos steps per random vector
%                     (clamped to min(M_lz, dim_sector))
%       is_real       true for irreps p = 0 or 2p = N (real H);
%                     false otherwise
%       sector_seed   deterministic seed for this sector
%
%   Outputs:
%       E_sec         concatenated Ritz values, length R_eff * M_lz_eff
%       w_sec         FTLM weights (DIM_SECTOR / R_eff) * |q_k^(1)|^2,
%                     real, summing to DIM_SECTOR per random vector
%
%   The M <-> -M multiplicity factor (mult_M = 1 + (M > 0)) is NOT
%   applied here; the caller (ftlm_observables_pg) multiplies W_SEC by
%   mult_M after this function returns.
%
%   This is the pure-MATLAB CPU reference path (FP64). The GPU FP32
%   path will live in a CUDA kernel cuda_lanczos_clut_block_pg.cu
%   following Milestone B/C.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    if nargin < 7, parity = []; end

    if isempty(parity)
        parity_on = false;
        D_eff = dim_sector;
        sigma = 0;
        P_idx = [];
        P_phase = [];
    else
        parity_on = true;
        D_eff   = parity.D_sigma;
        sigma   = parity.sigma;
        P_idx   = parity.P_idx;
        P_phase = parity.P_phase;
        if D_eff <= 0
            % Empty parity subspace: nothing to do.
            E_sec = []; w_sec = []; return;
        end
    end

    R_eff       = min(R, D_eff);
    M_lz_actual = min(M_lz, D_eff);

    E_sec = zeros(M_lz_actual * R_eff, 1);
    w_sec = zeros(M_lz_actual * R_eff, 1);
    idx   = 0;

    rng(sector_seed);

    for r = 1 : R_eff
        if is_real
            v0 = randn(dim_sector, 1);
        else
            v0 = randn(dim_sector, 1) + 1i * randn(dim_sector, 1);
        end

        if parity_on
            Pv0 = apply_parity_pg(v0, P_idx, P_phase);
            v0  = (v0 + sigma * Pv0) / sqrt(2);
            nrm = norm(v0);
            if nrm < 1e-12
                continue;   % degenerate projection, vanishingly rare
            end
            % lanczos_recursion_pg normalizes internally; no explicit / nrm
        end

        [alpha, beta] = lanczos_recursion_pg(H_pg, v0, M_lz_actual);
        [E_r, q1_r]   = solve_tridiag(alpha, beta);

        n_l = numel(E_r);
        E_sec(idx+1 : idx+n_l) = E_r;
        % FTLM single-vector weight uses the effective subspace dimension:
        % D_eff = dim_sector when parity is off, D_sigma otherwise.
        w_sec(idx+1 : idx+n_l) = (D_eff / R_eff) * q1_r;
        idx = idx + n_l;
    end

    E_sec = E_sec(1:idx);
    w_sec = w_sec(1:idx);
end

% ----------------------------------------------------------------
function [ep, q1] = solve_tridiag(alpha, beta)
%SOLVE_TRIDIAG  Diagonalize the Lanczos tridiagonal matrix.
%
%   Returns the Ritz values EP and the squared first components of
%   the Ritz eigenvectors, Q1 = |Q(1, :)|^2. Identical convention to
%   the release/ftlm_observables.m helper.

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
