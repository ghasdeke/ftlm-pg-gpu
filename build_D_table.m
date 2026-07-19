function [D, D_cum] = build_D_table(N_sites, two_s, A_total)
%BUILD_D_TABLE  Schnack's dimension recursion and cumulative table.
%
%   [D, D_CUM] = BUILD_D_TABLE(N_SITES, TWO_S, A_TOTAL)
%
%   Builds the tables used by the combinatorial-ranking (CR) scheme of
%   Schnack/Hage/Schmidt (2007). They define
%
%     D(m, A) = # of length-m strings with digits in {0, ..., 2s}
%               and total digit sum A
%
%   via the recursion
%
%     D(m, A) = sum_{q = 0}^{2s} D(m - 1, A - q),    D(0, 0) = 1
%
%   (with D(m, A) = 0 outside the allowed range). For ranking it is
%   convenient to also precompute the cumulative table
%
%     D_cum(m, A, a) = sum_{q = 0}^{a - 1} D(m, A - q),    a = 0, ..., 2s
%
%   so that the per-digit increment in the rank loop is a single table
%   lookup instead of a data-dependent inner sum.
%
%   Inputs:
%       N_SITES   number of digit positions
%       TWO_S     2*s (so digits in {0, ..., TWO_S})
%       A_TOTAL   total digit sum of the target sector
%                 (A_TOTAL = N_SITES * s + M_target by digit convention)
%
%   Outputs:
%       D       [N_SITES+1 x A_TOTAL+1] int64; D(m+1, A+1) = D(m, A)
%       D_CUM   [N_SITES+1 x A_TOTAL+1 x TWO_S+1] int64;
%               D_CUM(m+1, A+1, a+1) = D_cum(m, A, a)
%
%   Sizes are TINY: for N=30, s=1/2, M=0 -> A_TOTAL=15, TWO_S=1,
%   D and D_cum each ~ 500 int64 entries -> ~ 4 KB total. Stays in L1.
%
%   See also SCHNACK_RANK, BUILD_LOOKUP_SCHNACK.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% Source method:
%   J. Schnack, P. Hage, H.-J. Schmidt,
%   "Efficient implementation of the Lanczos method for magnetic systems",
%   J. Comput. Phys. 227 (2008), 4512.
% ================================================================

    N_sites = double(N_sites);
    two_s   = double(two_s);
    A_total = double(A_total);

    %% D(m, A) via the Schnack recursion. int64 throughout; D values can
    %  exceed 2^31 (e.g., D(36, 18) = C(36,18) ~ 9.1e9 for s=1/2 / N=36).
    D = zeros(N_sites + 1, A_total + 1, 'int64');
    D(1, 1) = 1;                          % D(0, 0) = 1
    for m = 1 : N_sites
        for A = 0 : A_total
            acc = int64(0);
            for q = 0 : two_s
                A_prev = A - q;
                if A_prev >= 0
                    acc = acc + D(m, A_prev + 1);
                end
            end
            D(m + 1, A + 1) = acc;
        end
    end

    %% D_cum(m, A, a) = sum_{q = 0}^{a - 1} D(m, A - q). The summation
    %  is a running prefix in q, so we build it incrementally.
    D_cum = zeros(N_sites + 1, A_total + 1, two_s + 1, 'int64');
    for m = 0 : N_sites
        for A = 0 : A_total
            cum = int64(0);
            for a = 0 : two_s
                D_cum(m + 1, A + 1, a + 1) = cum;
                A_prev = A - a;
                if A_prev >= 0
                    cum = cum + D(m + 1, A_prev + 1);
                end
            end
        end
    end
end
