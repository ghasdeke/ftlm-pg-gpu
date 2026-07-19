function [alpha, beta] = lanczos_recursion_pg(H, v0, M_lz)
%LANCZOS_RECURSION_PG  Three-term Lanczos recursion for Hermitian H.
%
%   [ALPHA, BETA] = LANCZOS_RECURSION_PG(H, V0, M_LZ) runs the symmetric
%   Lanczos recursion without reorthogonalization starting from a (real
%   or complex) vector V0 with ||V0|| = 1 (or any positive norm; the
%   first step normalizes).
%
%   Inputs:
%       H      Hermitian operator (sparse or full, real symmetric or
%              complex Hermitian)
%       V0     starting vector (numel = size(H, 1))
%       M_lz   maximum number of Lanczos steps
%
%   Outputs:
%       alpha  diagonal coefficients (real column, length <= M_LZ)
%       beta   subdiagonal coefficients (real column, length <= numel(alpha)-1)
%
%   For Hermitian H, the inner products v'*H*v are real, so alpha is real
%   even when v is complex. The Lanczos tridiagonal T = diag(alpha) +
%   diag(beta, 1) + diag(beta, -1) is real symmetric and can be diagonalized
%   with standard solvers; its eigenvectors are real, so the FTLM weights
%   |q_k^(1)|^2 are also real.
%
%   Early termination: the recursion stops as soon as the next beta falls
%   below 1e-12 (effective convergence of the Krylov subspace).
%
%   This is the CPU-only reference implementation used inside
%   RUN_FTLM_PG_SECTOR.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    alpha = zeros(M_lz, 1);
    beta  = zeros(M_lz - 1, 1);

    v_prev = zeros(size(v0), 'like', v0);
    v      = v0 / norm(v0);

    n_alpha = 0;
    for j = 1 : M_lz
        w   = H * v;
        a_j = real(v' * w);          % real for Hermitian H
        alpha(j) = a_j;
        n_alpha  = j;

        w = w - a_j * v;
        if j > 1
            w = w - beta(j-1) * v_prev;
        end

        if j < M_lz
            b_j = norm(w);
            if b_j < 1e-12
                break;                % Krylov subspace converged
            end
            beta(j) = b_j;
            v_prev = v;
            v      = w / b_j;
        end
    end

    alpha = alpha(1:n_alpha);
    beta  = beta(1:max(0, n_alpha - 1));
end
