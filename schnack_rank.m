function ranks = schnack_rank(states, D_cum, N_sites, two_s, d_loc, A_total)
%SCHNACK_RANK  Vectorised combinatorial ranking (Algorithm 1 in arxiv ms).
%
%   RANKS = SCHNACK_RANK(STATES, D_CUM, N_SITES, TWO_S, D_LOC, A_TOTAL)
%
%   For each integer-encoded state in STATES (digit base D_LOC = 2s+1,
%   N_SITES digits total, total digit sum A_TOTAL), returns its rank in
%   the M-sector in ascending integer-label order. NO basis array is
%   touched; the rank is computed purely from D_CUM (a few tens of KB,
%   built by BUILD_D_TABLE) and the state's digits.
%
%   Algorithm (Schnack/Hage/Schmidt 2007, GPU-adapted in arXiv 2506.xxxxx):
%       rank ← 0,    A ← A_total
%       for p = N-1 down to 0:
%           a_p ← digit p of state
%           rank ← rank + D_cum(p, A, a_p)
%           A   ← A - a_p
%       return rank
%
%   For ranking states that DO satisfy sum(digits) == A_TOTAL, the
%   result is in [0, D(N_SITES, A_TOTAL)). For states with the wrong
%   digit sum the function returns garbage; the caller is responsible
%   for filtering (this matters when the lookup is used for spin-flip
%   targets that may fall outside the M-sector, though for
%   M-conserving Hamiltonians like the Heisenberg model this never
%   happens by construction).
%
%   Performance: vectorised over STATES; the inner loop runs N_SITES
%   times regardless of input value, hence is data-independent. For
%   icosidodecahedron-scale batches (~ 1M states) the cost is
%   dominated by N_SITES * batch * vector-load throughput.
%
%   See also BUILD_D_TABLE, BUILD_LOOKUP_SCHNACK, QUERY_LOOKUP_SCHNACK.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    states = int64(states(:));
    n      = numel(states);
    if n == 0
        ranks = zeros(0, 1, 'int64');
        return;
    end

    N_sites = double(N_sites);
    A_total = double(A_total);
    d_loc_i64 = int64(d_loc);

    %% Pre-extract all digits: digits(p+1, i) = a_p of states(i).
    %  int8 since digits are in {0, ..., 4} for s up to 2.
    digits = zeros(N_sites, n, 'int8');
    tmp    = states;
    for p = 1 : N_sites
        d = mod(tmp, d_loc_i64);
        digits(p, :) = int8(d).';
        tmp = idivide(tmp, d_loc_i64);
    end

    %% Pre-compute linear-index strides into D_cum, which has shape
    %  (N_sites+1) x (A_total+1) x (2s+1). MATLAB column-major.
    A_stride = int64(N_sites + 1);
    a_stride = int64((N_sites + 1) * (A_total + 1));

    %% Main loop (Algorithm 1). All operations vectorised over STATES.
    ranks = zeros(n, 1, 'int64');
    A_vec = int64(A_total) * ones(n, 1, 'int64');

    for p = N_sites - 1 : -1 : 0
        a_p = int64(digits(p + 1, :)).';           % [n x 1]
        % Linear index into D_cum for each state:
        %   row    = p + 1
        %   col    = A_vec + 1
        %   slab   = a_p + 1
        lin_idx = int64(p + 1) + A_vec * A_stride + a_p * a_stride;
        ranks   = ranks + D_cum(lin_idx);
        A_vec   = A_vec - a_p;
    end
end
