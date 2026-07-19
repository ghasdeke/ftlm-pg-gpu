function s_arr = apply_sigma_ring(n_arr, N, d_loc)
%APPLY_SIGMA_RING  Digit-reverse operator (ring reflection sigma).
%
%   S_ARR = APPLY_SIGMA_RING(N_ARR, N, D_LOC) treats each entry of N_ARR
%   as a base-D_LOC integer with N digits and returns the integer whose
%   digit string is reversed: a_k -> a_{N-1-k}.
%
%   This is the action of the ring reflection sigma on the state-integer
%   encoding used throughout the PG branch: sigma maps |m_0,...,m_{N-1}>
%   to |m_{N-1},...,m_0>, which in the integer encoding sends n to
%       sigma(n) = sum_k a_{N-1-k} * d_loc^k.
%
%   Vectorized over the input array; works on int64 inputs of any shape.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    d_loc_i = int64(d_loc);
    n_arr   = int64(n_arr);
    s_arr   = zeros(size(n_arr), 'int64');

    % Extract digits position by position and place them in reverse order
    tmp = n_arr;
    for k = 0 : N - 1
        dg = mod(tmp, d_loc_i);
        % digit at position k goes to position (N-1-k) in the reversed integer
        s_arr = s_arr + dg * int64(d_loc)^int64(N - 1 - k);
        tmp = idivide(tmp, d_loc_i);
    end
end
