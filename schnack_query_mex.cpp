/*
 * schnack_query_mex.cpp
 * ================================================================
 * Fused Schnack-CR state -> super_rep_idx query in a SINGLE pass,
 * parallelised with std::thread.
 *
 * Replaces the two-stage host path
 *     ranks = schnack_rank(states, ...);            % N vectorised passes
 *     [~, idx] = ismember(ranks, super_reps_rank);  % sort + binary search
 * (i.e. QUERY_LOOKUP_SCHNACK) with one loop per state that
 *   (1) decomposes the state into base-d_loc digits,
 *   (2) accumulates the Schnack rank from D_cum in registers, and
 *   (3) binary-searches the (already sorted) super_reps_rank array,
 * returning the 1-based super-rep index (0 = not a super-rep).
 *
 * The N-position rank loop runs ONCE in registers instead of N times
 * over the whole state array, and the binary search never re-sorts the
 * search array -- the two costs BENCH_SCHNACK_SCALING flags as ~73% of
 * the per-entry collect time at N=36. At large N the binary search into
 * the big super_reps_rank array is memory-latency bound, which threads
 * hide well, so the work is split across std::thread::hardware_concurrency
 * lanes. Memory is neutral: D_cum is a few KB and super_reps_rank is the
 * same array QUERY_LOOKUP_SCHNACK already holds.
 *
 * Threading note: we deliberately use std::thread, NOT OpenMP. An MSVC
 * /openmp MEX links vcomp140.dll, which clashes with MATLAB's own
 * libiomp5 and faults at process teardown (0xC0000005). std::thread uses
 * the OS scheduler directly -- no second OpenMP runtime, no conflict.
 *
 * INTERFACE
 *   idx               = schnack_query_mex(states, D_cum, N, two_s, A_total, srr)
 *   [idx, ranks]      = schnack_query_mex(...)
 *   [threaded, hwc]   = schnack_query_mex('info')
 *
 *     states   int64 [n x 1]  integer-encoded states (digit sum = A_total)
 *     D_cum    int64 [(N+1) x (A_total+1) x (two_s+1)] cumulative table
 *                              from BUILD_D_TABLE (column-major, as MATLAB)
 *     N        scalar          number of sites
 *     two_s    scalar          2s (digits in 0..two_s, d_loc = two_s+1)
 *     A_total  scalar          digit sum of the sector (= N*s + M)
 *     srr      int64 [m x 1]   super_reps_rank, sorted ascending
 *
 *     idx      int32 [n x 1]   1-based position in srr, or 0 if absent
 *     ranks    int64 [n x 1]   (optional) the computed M-sector ranks
 *
 * Bit-for-bit equivalent to QUERY_LOOKUP_SCHNACK for states whose digit
 * sum equals A_total (guaranteed for M-conserving Heisenberg flips).
 *
 * Build:  build_schnack_query_mex   (or  mex -R2018a -O schnack_query_mex.cpp)
 *
 * See also QUERY_LOOKUP_SCHNACK, SCHNACK_RANK, BUILD_D_TABLE,
 *          BENCH_SCHNACK_SCALING, COLLECT_CLT_ENTRIES_IH.
 * ================================================================
 * Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
 * and Helmholtz-Zentrum Dresden-Rossendorf e.V.
 * Licensed under the Apache License, Version 2.0.
 * ================================================================
 */

#include "mex.h"
#include <cstdint>
#include <thread>
#include <vector>
#include <algorithm>

#define MAX_SITES 64

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* schnack_query_mex('info') -> [threaded, hardware_concurrency] */
    if (nrhs == 1 && mxIsChar(prhs[0])) {
        unsigned hwc = std::thread::hardware_concurrency();
        plhs[0] = mxCreateDoubleMatrix(1, 2, mxREAL);
        double *o = mxGetPr(plhs[0]);
        o[0] = 1.0; o[1] = (double)(hwc ? hwc : 1u);
        return;
    }

    if (nrhs != 6)
        mexErrMsgIdAndTxt("schnack_query_mex:nrhs",
            "Six inputs required: states, D_cum, N, two_s, A_total, srr.");
    if (!mxIsInt64(prhs[0]) || !mxIsInt64(prhs[1]) || !mxIsInt64(prhs[5]))
        mexErrMsgIdAndTxt("schnack_query_mex:type",
            "states, D_cum and srr must be int64.");

    const int64_t *states = (const int64_t *) mxGetInt64s(prhs[0]);
    const int64_t *Dcum   = (const int64_t *) mxGetInt64s(prhs[1]);
    const int64_t *srr    = (const int64_t *) mxGetInt64s(prhs[5]);

    const mwSize n      = mxGetNumberOfElements(prhs[0]);
    const mwSize n_reps = mxGetNumberOfElements(prhs[5]);

    const int N       = (int) mxGetScalar(prhs[2]);
    const int two_s   = (int) mxGetScalar(prhs[3]);
    const int A_total = (int) mxGetScalar(prhs[4]);
    const int64_t d_loc = (int64_t)(two_s + 1);

    if (N < 1 || N > MAX_SITES)
        mexErrMsgIdAndTxt("schnack_query_mex:N", "N must be in 1..%d.", MAX_SITES);

    /* D_cum strides (column-major): element (p, A, d) [0-based] is at
     *   p + A*(N+1) + d*(N+1)*(A_total+1). */
    const mwSize Astride = (mwSize)(N + 1);
    const mwSize dstride = (mwSize)(N + 1) * (mwSize)(A_total + 1);

    plhs[0] = mxCreateNumericMatrix(n, 1, mxINT32_CLASS, mxREAL);
    int32_t *idx = (int32_t *) mxGetInt32s(plhs[0]);

    int64_t *ranks_out = nullptr;
    if (nlhs >= 2) {
        plhs[1] = mxCreateNumericMatrix(n, 1, mxINT64_CLASS, mxREAL);
        ranks_out = (int64_t *) mxGetInt64s(plhs[1]);
    }

    if (n == 0) return;

    /* Per-state kernel over a half-open index range [a, b). */
    auto worker = [&](mwSize a, mwSize b) {
        for (mwSize i = a; i < b; i++) {
            int64_t s = states[i];

            /* (1) digits, low position first */
            int digit[MAX_SITES];
            for (int p = 0; p < N; p++) {
                digit[p] = (int)(s % d_loc);
                s /= d_loc;
            }

            /* (2) Schnack rank: high position first, one D_cum read per pos */
            int64_t rank = 0;
            int Ak = A_total;
            for (int p = N - 1; p >= 0; p--) {
                int d = digit[p];
                rank += Dcum[(mwSize)p + (mwSize)Ak * Astride + (mwSize)d * dstride];
                Ak -= d;
            }
            if (ranks_out) ranks_out[i] = rank;

            /* (3) binary search in the sorted super_reps_rank */
            mwSize lo = 0, hi = n_reps;     /* [lo, hi) */
            int32_t found = 0;
            while (lo < hi) {
                mwSize mid = lo + ((hi - lo) >> 1);
                int64_t v = srr[mid];
                if (v == rank) { found = (int32_t)(mid + 1); break; }
                else if (v < rank) lo = mid + 1;
                else hi = mid;
            }
            idx[i] = found;
        }
    };

    /* Thread count: scale with work, cap at hardware concurrency. */
    const mwSize MIN_CHUNK = 4096;
    unsigned hwc = std::thread::hardware_concurrency(); if (hwc == 0) hwc = 1;
    mwSize want = n / MIN_CHUNK; if (want < 1) want = 1;
    unsigned nthreads = (unsigned) std::min<mwSize>(want, (mwSize)hwc);
    if (nthreads <= 1) { worker(0, n); return; }

    const mwSize chunk = (n + nthreads - 1) / nthreads;
    std::vector<std::thread> pool;
    pool.reserve(nthreads - 1);
    for (unsigned t = 0; t < nthreads; t++) {
        mwSize a = (mwSize)t * chunk;
        if (a >= n) break;
        mwSize b = std::min<mwSize>(a + chunk, n);
        if (t + 1 < nthreads && b < n)
            pool.emplace_back(worker, a, b);   /* helper lanes */
        else { worker(a, b); break; }          /* last lane on this thread */
    }
    for (auto &th : pool) th.join();
}
