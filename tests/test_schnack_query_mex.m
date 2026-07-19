function test_schnack_query_mex(N_list, batch)
%TEST_SCHNACK_QUERY_MEX  Correctness + throughput of the fused Schnack query.
%
%   TEST_SCHNACK_QUERY_MEX()
%   TEST_SCHNACK_QUERY_MEX(N_LIST, BATCH)
%
%   (A) CORRECTNESS: builds a real BUILD_LOOKUP_SCHNACK over a sampled set
%       of M=0 super-reps and checks that SCHNACK_QUERY_MEX reproduces
%       QUERY_LOOKUP_SCHNACK bit-for-bit on a mix of in-basis and
%       out-of-basis queries (and that the optional rank output matches
%       SCHNACK_RANK).
%
%   (B) THROUGHPUT: for each N, times the production host query path
%       (QUERY_LOOKUP_SCHNACK = schnack_rank + ismember) against the fused
%       MEX on size-faithful fixtures (true n_reps ~ C(N,N/2)/120), and on
%       identical inputs also re-checks equality -- so N=36 is verified
%       against the reference too.
%
%   (C) AMORTIZATION: throughput of the threaded MEX vs batch size at
%       N=36. The std::thread spawn is per-call, so tiny batches under-
%       report; production collect queries ~n_reps/4 (~1.9e7) states per
%       (bond,sign), well into the amortized regime.
%
%   Defaults: N_LIST = [30 36], BATCH = 2000000 (production-representative).
%
%   See also SCHNACK_QUERY_MEX, BUILD_SCHNACK_QUERY_MEX,
%            QUERY_LOOKUP_SCHNACK, BENCH_SCHNACK_SCALING.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    if nargin < 1 || isempty(N_list), N_list = [30 36];  end
    if nargin < 2 || isempty(batch),  batch  = 2000000;  end

    s_val = 0.5; two_s = 1; d_loc = 2; order = 120; min_t = 0.30;
    rng(20260531);

    if exist('schnack_query_mex', 'file') ~= 3
        fprintf('schnack_query_mex not built; building now...\n');
        build_schnack_query_mex();
    end

    %% ---------- (A) Correctness against a real lookup (N=30) ----------
    fprintf('\n=== (A) Correctness vs query_lookup_schnack (N=30, real lookup) ===\n');
    Nc = 30;
    pool       = make_M0_states(Nc, 300000);
    super_reps = unique(pool);                       % sorted ascending int64
    lk = build_lookup_schnack(super_reps, s_val, Nc);

    n_hit  = 5000;
    hits   = super_reps(randperm(numel(super_reps), min(n_hit, numel(super_reps))));
    misses = make_M0_states(Nc, 5000);               % mostly not super-reps
    q = [hits; misses];
    q = q(randperm(numel(q)));

    ref = query_lookup_schnack(lk, q);
    [got, ranks_mex] = schnack_query_mex(int64(q), lk.D_cum, Nc, lk.two_s, ...
                                         lk.A_total, lk.super_reps_rank);
    ranks_ref = schnack_rank(q, lk.D_cum, lk.N_sites, lk.two_s, lk.d_loc, lk.A_total);

    ok_idx  = isequal(int32(ref(:)), int32(got(:)));
    ok_rank = isequal(int64(ranks_ref(:)), int64(ranks_mex(:)));
    n_in    = nnz(got > 0);
    fprintf('  queries=%d (in-basis=%d), idx match=%d, rank match=%d\n', ...
            numel(q), n_in, ok_idx, ok_rank);
    assert(ok_idx,  'schnack_query_mex idx mismatch vs query_lookup_schnack');
    assert(ok_rank, 'schnack_query_mex rank mismatch vs schnack_rank');
    fprintf('  PASS\n');

    %% ---------- (B) Throughput + N=36 equality ----------
    fprintf('\n=== (B) Throughput: host query vs fused MEX ===\n');
    fprintf('  N | n_reps     | host q/s   | mex q/s    | speedup | idx match\n');
    fprintf('  --+------------+------------+------------+---------+----------\n');
    host36 = NaN;
    for N = N_list(:).'
        A_total = N / 2;
        n_reps  = round(nchoosek(N, N/2) / order);

        states   = make_M0_states(N, batch);
        [~, Dcum] = build_D_table(N, two_s, A_total);
        srr = (int64(0):int64(n_reps - 1)).' * int64(order);   % sorted search array

        % minimal lookup struct for the reference host path
        lkN = struct('D_cum', Dcum, 'N_sites', N, 'two_s', two_s, ...
                     'd_loc', d_loc, 'A_total', A_total, 'super_reps_rank', srr);

        % equality on identical inputs (verifies N=36 against the reference)
        ref_b = query_lookup_schnack(lkN, states);
        got_b = schnack_query_mex(states, Dcum, N, two_s, A_total, srr);
        ok_b  = isequal(int32(ref_b(:)), int32(got_b(:)));
        assert(ok_b, 'N=%d: mex disagrees with query_lookup_schnack', N);

        tp_host = timed_tp(@() query_lookup_schnack(lkN, states), batch, min_t);
        tp_mex  = timed_tp(@() schnack_query_mex(states, Dcum, N, two_s, A_total, srr), ...
                           batch, min_t);

        fprintf('  %2d | %10s | %6.2e | %6.2e | %6.1fx | %d\n', ...
                N, human_count(n_reps), tp_host, tp_mex, tp_mex/tp_host, ok_b);
        if N == 36, host36 = tp_host; end

        clear srr states Dcum lkN ref_b got_b;
    end

    %% ---------- (C) Batch-size amortization at N=36 ----------
    if any(N_list(:) == 36)
        if isnan(host36), host36 = 1.0e6; end     % measured N=36 host baseline
        fprintf('\n=== (C) Threaded MEX throughput vs batch size (N=36) ===\n');
        N = 36; A_total = N / 2; n_reps = round(nchoosek(N, N/2) / order);
        srr = (int64(0):int64(n_reps - 1)).' * int64(order);
        [~, Dcum] = build_D_table(N, two_s, A_total);
        Kmax = max(4000000, batch);
        big  = make_M0_states(N, Kmax);
        fprintf('  batch K   | mex q/s    | speedup vs host\n');
        fprintf('  ----------+------------+----------------\n');
        for K = unique([100000, 500000, 2000000, Kmax])
            sK = big(1:K);
            tp = timed_tp(@() schnack_query_mex(sK, Dcum, N, two_s, A_total, srr), ...
                          K, min_t);
            fprintf('  %9d | %6.2e | %5.1fx\n', K, tp, tp / host36);
        end
        clear srr Dcum big sK;
    end

    fprintf('\nAll checks passed.\n');
end


% ----------------------------------------------------------------
function states = make_M0_states(N, B)
    half = N / 2;
    [~, ord] = sort(rand(B, N), 2);
    sel = ord(:, 1:half);
    states = zeros(B, 1, 'int64');
    two = int64(2);
    for c = 1 : half
        states = states + two .^ int64(sel(:, c) - 1);
    end
end


% ----------------------------------------------------------------
function tp = timed_tp(fn, n_per_call, min_t)
    fn();                                    % warm-up
    k = 0; t0 = tic;
    while toc(t0) < min_t
        fn(); k = k + 1;
    end
    tp = k * n_per_call / toc(t0);
end


% ----------------------------------------------------------------
function s = human_count(n)
    if n >= 1e6, s = sprintf('%.2e', n); else, s = sprintf('%d', n); end
end
