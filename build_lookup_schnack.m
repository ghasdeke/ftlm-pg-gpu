function lookup = build_lookup_schnack(super_reps, s_val, N_sites)
%BUILD_LOOKUP_SCHNACK  Schnack-CR state -> super_rep_idx structure.
%
%   LOOKUP = BUILD_LOOKUP_SCHNACK(SUPER_REPS, S_VAL, N_SITES)
%
%   Produces a lookup struct that maps any M-sector state to its
%   super-rep index (or 0 if not a super-rep). Memory cost is
%   essentially zero: the D_cum table is ~ a few KB, and the only
%   per-rep storage is the sorted-ranks array of size n_super_reps x 8 B
%   (5 MB at N=30, ~ 300 MB at N=36 -- versus 256 MB / 17 GB for the
%   bitmap CLT at the same sizes).
%
%   Mechanism. For PG-symmetry-adapted bases, the basis (super-reps)
%   is a SUBSET of the M-sector. Schnack's combinatorial ranking
%   provides state -> M-sector-rank in O(N) without storage. We then
%   resolve M-sector-rank -> super_rep_idx via binary search in
%   LOOKUP.SUPER_REPS_RANK, which we precompute (and which is
%   automatically sorted because SUPER_REPS is sorted by state-integer
%   and the Schnack ranking is monotonic in state-integer).
%
%   Inputs:
%       SUPER_REPS  [n_reps x 1] int64, sorted ascending (orbit minima
%                   in this M sector)
%       S_VAL       local spin value
%       N_SITES     number of sites
%
%   Output struct fields:
%       type             'schnack'
%       N_sites          echoed
%       s_val            echoed
%       two_s            int = round(2 * s_val)
%       d_loc            int = 2s+1
%       A_total          int = sum(digits) of any super-rep
%                              (equivalently N*s + M_target)
%       D                from BUILD_D_TABLE; D(N, A_total) is the M-sector dim
%       D_cum            from BUILD_D_TABLE; used for ranking
%       super_reps       echoed (kept for verification / unranking)
%       super_reps_rank  [n_reps x 1] int64, ranks in M-sector (sorted)
%       n_reps           numel(super_reps)
%
%   See also SCHNACK_RANK, QUERY_LOOKUP_SCHNACK, BUILD_CLT_LOOKUP.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    super_reps = int64(super_reps(:));
    n_reps     = numel(super_reps);

    two_s = round(2 * s_val);
    d_loc = two_s + 1;

    %% A_total from a single super-rep (or 0 for empty basis).
    if n_reps == 0
        A_total = round(N_sites * s_val);   % default for empty: use M=0 sector
    else
        A_total = sum_digits(super_reps(1), d_loc, N_sites);
    end

    %% Tables (tiny, ~ KB).
    [D, D_cum] = build_D_table(N_sites, two_s, A_total);

    %% Per-rep ranks. SUPER_REPS is sorted ascending; Schnack ranking is
    %  monotonic in state-integer, so SUPER_REPS_RANK is also sorted.
    if n_reps == 0
        super_reps_rank = zeros(0, 1, 'int64');
    else
        super_reps_rank = schnack_rank(super_reps, D_cum, ...
                                        N_sites, two_s, d_loc, A_total);
    end

    %% Pack
    lookup.type            = 'schnack';
    lookup.N_sites         = double(N_sites);
    lookup.s_val           = s_val;
    lookup.two_s           = two_s;
    lookup.d_loc           = d_loc;
    lookup.A_total         = A_total;
    lookup.D               = D;
    lookup.D_cum           = D_cum;
    lookup.super_reps      = super_reps;
    lookup.super_reps_rank = super_reps_rank;
    lookup.n_reps          = n_reps;
end


% ----------------------------------------------------------------
function A = sum_digits(state, d_loc, N_sites)
%SUM_DIGITS  Total digit sum of a single state in base d_loc.
    A = 0;
    s = int64(state);
    di64 = int64(d_loc);
    for k = 1 : N_sites
        d = mod(s, di64);
        A = A + double(d);
        s = idivide(s, di64);
    end
end
