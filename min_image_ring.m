function [rep, h, L] = min_image_ring(n, N, d_loc)
%MIN_IMAGE_RING  Minimum-image and orbit length under cyclic translation C_N.
%
%   [REP, H, L] = MIN_IMAGE_RING(N_STATE, N, D_LOC) treats N_STATE as a
%   base-D_LOC integer with N digits and returns:
%
%       REP - the smallest integer in the orbit {T^g(N_STATE) : g = 0..N-1}
%       H   - smallest non-negative g such that T^H(N_STATE) = REP
%       L   - orbit length (smallest positive g with T^g(N_STATE) = N_STATE)
%
%   T denotes the cyclic right shift of digits:
%       T(n) = floor(n / D_LOC) + (n mod D_LOC) * D_LOC^(N-1).
%
%   N_STATE may be a scalar int64 (typical) or any integer-valued numeric
%   type that fits in int64. REP is returned in the same class as N_STATE.
%
%   Orbit length L always divides N. The largest possible value is N
%   (generic orbit); L < N occurs only for states with non-trivial
%   stabilizer subgroups (e.g., the fully polarized state has L = 1).
%
%   This is the MATLAB reference implementation used by
%   enumerate_sector_with_translation and build_heisenberg_sparse_pg.
%   The GPU device function in the CUDA kernel must mirror this convention.
%
%   See also ENUMERATE_SECTOR_WITH_TRANSLATION,
%            BUILD_HEISENBERG_SPARSE_PG.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    % Promote to int64 for safe arithmetic (covers d_loc^N up to ~9.2e18).
    n64    = int64(n);
    d_loc_ = int64(d_loc);
    d_top  = d_loc_ ^ int64(N - 1);

    rep   = n64;
    h     = int32(0);
    L     = int32(N);  % default: full orbit; overwritten if a shorter cycle is found
    n_cur = n64;

    for g = 1 : N - 1
        % Cyclic right shift of digits.
        n_cur = idivide(n_cur, d_loc_) + mod(n_cur, d_loc_) * d_top;
        if n_cur == n64
            L = int32(g);
            break;
        end
        if n_cur < rep
            rep = n_cur;
            h   = int32(g);
        end
    end

    % Cast rep back to the input class for convenience.
    if isa(n, 'int64') || isa(n, 'uint64')
        rep = int64(rep);
    else
        rep = cast(rep, 'like', n);
    end
end
