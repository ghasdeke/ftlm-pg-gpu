function cache = enumerate_M_orbits_Ih_gpu(s_val, M_target, group)
%ENUMERATE_M_ORBITS_IH_GPU  GPU-accelerated drop-in for ENUMERATE_M_ORBITS_IH.
%
%   CACHE = ENUMERATE_M_ORBITS_IH_GPU(S_VAL, M_TARGET, GROUP)
%
%   Same contract as ENUMERATE_M_ORBITS_IH, but the heaviest steps on a
%   large M sector are restructured for speed and bounded memory:
%
%   (A) M-sector filter + orbit-minimum test, FUSED and STREAMED
%       (Steps 1+2). For s >= 1 the full state space d_loc^12 grows
%       quickly (5^12 ~ 2.4e8 for s=2, 6^12 ~ 2.2e9 for s=5/2); for
%       s=1/2 at N=30 the M sector alone is ~1.55e8 states (1.24 GB).
%       Rather than materialise the whole M-basis and then run a
%       full-length MIN_IMAGE_IH over it (a ~5 GB host peak whose only
%       product is the ~1.3M super-reps), we walk the integer range in
%       chunks (on a gpuArray for the s>=1 digit decomposition; CPU
%       popcount for s=1/2), filter each chunk to M_target, and call
%       MIN_IMAGE_IH right away with a SINGLE output (no g_min array),
%       keeping only the states equal to their own orbit minimum. The
%       full basis list is never resident. Falls back transparently to
%       the CPU path if no CUDA device is available.
%
%   (B) Stabiliser flags (Steps 3+4). The original routine calls
%       APPLY_PERM_TO_STATE 120 times in sequence on the super-rep
%       vector. Each call is itself a 12-iteration BLAS reduction, so
%       the cost is O(120 * 12 * n_reps). We replace the outer 120-loop
%       with one matmul:
%
%           states_g_all(g, i) = sum_k perm_powers(g, k) * digits(k, i)
%
%       The stabiliser flag matrix is then a single elementwise compare
%       against the super-rep vector. This is the same BLAS trick used
%       inside MIN_IMAGE_IH and runs in ~50 ms on a 3 M-rep s=2 M=0
%       sector vs. several seconds for the per-g loop. The matmul is
%       chunked over reps so its [order x n_reps] working set never
%       exceeds ~150 MB at N=30; the result is identical to the
%       monolithic product.
%
%   Output CACHE has the same fields as ENUMERATE_M_ORBITS_IH:
%       M_target, s_val, super_reps, orbit_lens, and the stabilisers in
%       CSR form stab_flat (uint8) + stab_ptr (int64): rep i's stabiliser
%       g-indices are stab_flat(stab_ptr(i):stab_ptr(i+1)-1).
%
%   See also ENUMERATE_M_ORBITS_IH, MIN_IMAGE_IH, APPLY_IRREP_TO_ORBITS.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    d_loc   = round(2*s_val + 1);
    N_sites = double(group.N);     % was hardcoded 12; now driven by group
    n_total = double(d_loc)^N_sites;

    %% Probe for a CUDA device. Failure -> CPU fallback.
    use_gpu = false;
    try
        gd = gpuDevice();
        if gd.DeviceSupported
            use_gpu = true;
        end
    catch
        use_gpu = false;
    end

    %% Steps 1+2 FUSED (streaming): enumerate the M sector and keep ONLY
    %  orbit minima, without ever materialising the full M-basis list.
    %
    %  Old flow: build the complete basis_M (~1.55e8 int64 = 1.24 GB at
    %  N=30 / s=1/2 / M=0), run MIN_IMAGE_IH over all of it (a full-length
    %  reps array plus a DISCARDED full-length g_min array), then mask
    %  down to the ~1.3M super-reps. Three full-length passes whose only
    %  product is the 1.3M reps -- a host peak of order 5 GB.
    %
    %  New flow: "a state is a super-rep iff it equals its own orbit
    %  minimum" is a LOCAL test, so we stream the M sector in chunks, run
    %  MIN_IMAGE_IH (single-output, so no g_min) on each chunk, and keep
    %  only the orbit minima. The full basis_M is never resident; peak
    %  host transient is one enumeration chunk plus MIN_IMAGE_IH's own
    %  internal working set. super_reps stays sorted ascending (chunks
    %  processed in increasing order, order preserved within each chunk),
    %  which BUILD_LOOKUP_SCHNACK and the bitmap CLT both rely on.
    if s_val == 0.5
        k_up = N_sites/2 + M_target;
        if k_up < 0 || k_up > N_sites || abs(k_up - round(k_up)) > 0.01
            cache = empty_cache(s_val, M_target); return;
        end
        k_up = round(k_up);
        super_reps = stream_super_reps_spin_half(N_sites, k_up, group, s_val, use_gpu);
    else
        super_reps = stream_super_reps_general(n_total, d_loc, N_sites, ...
                                               s_val, M_target, use_gpu, group);
    end

    n_reps = numel(super_reps);
    if n_reps == 0
        cache = empty_cache(s_val, M_target); return;
    end

    %% Step 3+4: stabiliser flags via a CHUNKED BLAS matmul.
    %  perm_powers * digits gives the image of every super-rep under
    %  every group element; the stabiliser flag is then
    %  states_g(g, i) == super_reps(i). The monolithic [order x n_reps]
    %  product would be a ~1.25 GB transient at N=30; we chunk over reps
    %  so the working set is bounded to [order x REP_CHUNK] (~150 MB).
    %  The matmul has a 12-element inner dimension and is memory-bound,
    %  so chunking costs nothing measurable. The per-rep find/orbit-len
    %  bookkeeping is byte-identical to the monolithic version.
    %  Spin-flip Z2 (ADD_SPIN_FLIP_Z2): the matmul runs over the PERMUTATION
    %  half only; a flip element (g, P) stabilises rep r iff g*r == C - r
    %  with C = d_loc^N - 1 (flip(g*r) = C - g*r). The flip-stabiliser flags
    %  are stacked BELOW the plain ones, so the column-major find yields
    %  ascending combined indices 1..2|G| per rep, and orbit_lens =
    %  group.order ./ counts stays exact (group.order is already 2|G|).
    has_flip    = isfield(group, 'flip') && any(group.flip);
    if has_flip
        perms_d = double(group.perms(1:double(group.n_g0), :));
        C_flip  = double(d_loc)^N_sites - 1;
    else
        perms_d = double(group.perms);          % [order x N_sites]
    end
    perm_powers = d_loc .^ (perms_d - 1);       % [order(/2) x N_sites]
    d_loc_i64   = int64(d_loc);

    %% Stabilisers in CSR form (replaces the n_reps-cell stab_lists, whose
    %  per-cell overhead is ~9.4 GB at N=36). rep i's stabiliser g-indices
    %  are stab_flat(stab_ptr(i):stab_ptr(i+1)-1), uint8 (g in 1..120 < 256).
    %  stab_flat is ~ n_reps long (avg |Stab| ~ 1) -> ~76 MB; stab_ptr is
    %  n_reps+1 int64 -> ~0.6 GB. APPLY_IRREP_TO_ORBITS reads it only for the
    %  (rare) non-trivial-stabiliser reps.
    orbit_lens = zeros(n_reps, 1, 'int32');
    % Bound the [order x REP_CHUNK] matmul transient to ~0.25 GB: for the
    % shipped geometries (|G| <= 432) this reproduces the historical fixed
    % 131072, but a large USER group (|G| up to the 65535 cap) would have
    % turned the fixed chunk into a ~68 GB host transient. Chunking is
    % byte-identical for any REP_CHUNK.
    REP_CHUNK  = min(131072, max(2048, floor(0.25e9 / (8 * max(1, size(perms_d, 1))))));
    n_chunks   = ceil(n_reps / REP_CHUNK);
    flat_cells = cell(n_chunks, 1);
    cc = 0;
    for cs = 1 : REP_CHUNK : n_reps
        cc  = cc + 1;
        ce  = min(cs + REP_CHUNK - 1, n_reps);
        idx = cs : ce;
        sr_chunk = super_reps(idx);

        % digits(k, j) = k-th base-d_loc digit of sr_chunk(j).
        digits_chunk = zeros(N_sites, numel(idx));
        tmp = sr_chunk;
        for k = 1 : N_sites
            digits_chunk(k, :) = double(mod(tmp, d_loc_i64)).';
            tmp = idivide(tmp, d_loc_i64);
        end

        states_g_chunk  = perm_powers * digits_chunk;             % [order(/2) x len]
        stab_flag_chunk = (states_g_chunk == double(sr_chunk).'); % [order(/2) x len] logical
        if has_flip
            % Flip-stabiliser flags (g, P): g*r == C - r, stacked below ->
            % combined index n_g0 + g, ascending within each column.
            stab_flag_chunk = [stab_flag_chunk; ...
                               (states_g_chunk == C_flip - double(sr_chunk).')];
        end

        % Column-major find groups the g-indices by rep, ascending in g --
        % identical order to the old per-rep find(stab_flag_chunk(:,jj)).
        [g_rows, ~]      = find(stab_flag_chunk);                 % flat, rep order
        flat_cells{cc}   = uint16(g_rows);                        % g in 1..order (uint16: |G| up to 65535)
        counts_chunk     = sum(stab_flag_chunk, 1).';            % [len x 1]
        orbit_lens(idx)  = int32(group.order ./ counts_chunk);

        clear digits_chunk states_g_chunk stab_flag_chunk g_rows counts_chunk;
    end

    stab_flat = vertcat(flat_cells{:});
    stab_ptr  = [int64(1); int64(1) + cumsum(int64(group.order) ./ int64(orbit_lens))];
    clear flat_cells;

    %% Pack
    cache.M_target   = M_target;
    cache.s_val      = s_val;
    cache.super_reps = super_reps;
    cache.orbit_lens = orbit_lens;
    cache.stab_flat  = stab_flat;     % uint8  [sum|Stab| x 1]
    cache.stab_ptr   = stab_ptr;      % int64  [n_reps+1 x 1], CSR row pointers
end


% ----------------------------------------------------------------
function super_reps = stream_super_reps_spin_half(N_sites, k_up, group, s_val, use_gpu)
%STREAM_SUPER_REPS_SPIN_HALF  Combinatorial M-sector super-rep collector (s=1/2).
%
%   Generates the popcount == k_up sector DIRECTLY by combinatorial
%   unranking (in ascending integer order) instead of scanning all 2^N
%   integers and popcount-filtering. Only C(N, k_up) states are produced
%   (1.55e8 at N=30, vs scanning 2^30 ~ 1.07e9), in chunks; each chunk
%   goes through MIN_IMAGE_IH / MIN_IMAGE_IH_GPU and only orbit minima
%   are kept. This both removes the popcount scan and -- crucially --
%   scales: the 2^N scan becomes infeasible at larger N (2^36 ~ 6.9e10),
%   the direct generation does not.
%
%   UNRANK_POPCOUNT is the exact inverse of SCHNACK_RANK for two_s = 1,
%   so the produced sequence is byte-identical (same set, same ascending
%   order) to the old popcount scan; super_reps stays sorted ascending,
%   as BUILD_LOOKUP_SCHNACK and the bitmap CLT require.
%
%   When USE_GPU is true the unranking and the orbit-minimum search both
%   run on the GPU (states never leave VRAM until the small kept-rep list
%   is gathered); otherwise everything runs on the host.

    [D, ~] = build_D_table(N_sites, 1, k_up);     % D(p+1,A+1) = C(p,A)
    total  = D(N_sites + 1, k_up + 1);            % = C(N, k_up)
    if total <= 0
        super_reps = int64([]);
        return;
    end

    % Cache perm_powers on the GPU so MIN_IMAGE_IH_GPU reuses it per chunk.
    ppg = [];
    if use_gpu
        ppg = gpuArray(2 .^ (double(group.perms) - 1));   % d_loc = 2
    end

    % Spin-flip Z2: no G x Z2 super-rep can exceed C/2 = (2^N - 1)/2 (r <= C - r
    % is necessary: the pure flip maps r to C - r). The unranking walks states
    % ascending, so the scan can STOP at the first candidate above C/2 -- the
    % candidate volume halves, the kept set is unchanged.
    has_flip = isfield(group, 'flip') && any(group.flip);
    C_half   = floor((2^N_sites - 1) / 2);

    % VRAM-adaptive chunk (shared helper: sized to gpu_free_bytes, hard-
    % capped at the 2^31-elements-per-gpuArray limit; test-swept across
    % fake card sizes by test_gpu_sizing_invariants). Chunking only
    % partitions the scan -- kept set and order are byte-identical.
    CHUNK = adaptive_enum_chunk(N_sites, true);
    out_cells = {};
    for r0 = int64(0) : CHUNK : (total - 1)
        r1   = min(r0 + CHUNK - 1, total - 1);
        cand = unrank_popcount(r0, r1, N_sites, k_up, D, use_gpu);  % ascending int64
        if has_flip
            if gather(cand(1)) > C_half, break; end       % all later chunks higher
            cand = cand(cand <= C_half);                  % tail of the boundary chunk
            if isempty(cand), break; end
        end

        % Local super-rep test (R2: staged early-reject on GPU -- most non-minima
        % die after a cheap block; only survivors get the full orbit-min test).
        if use_gpu
            keep = gather(cand(is_super_rep_Ih_gpu(cand, group, s_val, ppg)));
        else
            reps_c = min_image_Ih(cand, group, s_val);            % single output
            keep   = cand(reps_c == cand);
        end
        if ~isempty(keep)
            out_cells{end+1} = keep(:); %#ok<AGROW>
        end
        clear cand reps_c keep;
    end

    if isempty(out_cells)
        super_reps = int64([]);
    else
        super_reps = vertcat(out_cells{:});
    end
end


% ----------------------------------------------------------------
function states = unrank_popcount(r0, r1, N_sites, k_up, D, use_gpu)
%UNRANK_POPCOUNT  Vectorised combinatorial unranking for the s=1/2 sector.
%
%   Returns the int64 states whose ranks (in ascending-integer order
%   among the popcount == k_up integers) are r0 .. r1 inclusive. Walking
%   bit positions p from high to low with A = ones still to place, the
%   bit p is set iff the rank still owes at least C(p, A) = D(p+1, A+1)
%   lower states (those with bit p = 0). Exact inverse of the two_s = 1
%   SCHNACK_RANK, hence the output is in ascending integer order. The
%   whole routine is data-independent (N_sites vectorised passes) and
%   runs transparently on host arrays or gpuArrays.

    nrows = int64(size(D, 1));                 % N_sites + 1
    r = (r0 : r1).';                           % int64 column of ranks
    if use_gpu
        r = gpuArray(r);
        D = gpuArray(D);
        states = gpuArray.zeros(numel(r), 1, 'int64');
        A      = k_up * gpuArray.ones(numel(r), 1, 'int64');
    else
        states = zeros(numel(r), 1, 'int64');
        A      = int64(k_up) * ones(numel(r), 1, 'int64');
    end

    one64 = int64(1);
    for p = N_sites - 1 : -1 : 0
        lin  = int64(p + 1) + A * nrows;        % linear index of D(p+1, A+1)
        c    = D(lin);                          % = C(p, A), per element
        si   = int64(r >= c);                   % 1 if bit p is set, else 0
        states = states + si * bitshift(one64, p);
        r      = r - si .* c;
        A      = A - si;
    end
end


% ----------------------------------------------------------------
function super_reps = stream_super_reps_general(n_total, d_loc, N_sites, ...
                                                s_val, M_target, use_gpu, group)
%STREAM_SUPER_REPS_GENERAL  Streaming super-rep collector for s >= 1 (R3).
%
%   General-d_loc analog of STREAM_SUPER_REPS_SPIN_HALF. Rather than
%   scanning all d_loc^N integers and digit-sum filtering down to the
%   M = M_target sector (only ~5% survive at s=7/2 -- 6.87e10 scanned for
%   3.4e9 kept), the sector is generated DIRECTLY by combinatorial
%   unranking (UNRANK_COMPOSITION) in ascending integer order. Each chunk
%   of ranks is unranked to states, reduced by MIN_IMAGE_IH(_GPU), and only
%   orbit minima are kept. The unranked states are already exactly the
%   M-sector, so no S^z computation / float rounding is needed. super_reps
%   stays sorted ascending (ranks walked in increasing order), as
%   BUILD_LOOKUP_SCHNACK and the bitmap CLT rely on.

    two_s = round(d_loc - 1);

    % Digit sum of the M = M_target sector: sum(digit) = N*s + M.
    A_total = N_sites*s_val + M_target;
    if abs(A_total - round(A_total)) > 1e-6 || A_total < 0 || A_total > N_sites*two_s
        super_reps = int64([]); return;
    end
    A_total = round(A_total);

    % Bounded-composition count tables (int64). D(p+1,a+1) = #{p digits in
    % [0,two_s] summing to a}; total = D(N+1,A_total+1) = exact sector dim.
    % D_cum is the cumulative table the R5 unranking reads directly.
    [D, D_cum] = build_D_table(N_sites, two_s, A_total);
    total  = D(N_sites + 1, A_total + 1);
    if total <= 0
        super_reps = int64([]); return;
    end

    % Cache perm_powers on the GPU so MIN_IMAGE_IH_GPU reuses it per chunk
    % (mirrors STREAM_SUPER_REPS_SPIN_HALF's ppg; R1).
    ppg = [];
    if use_gpu
        ppg = gpuArray(double(d_loc) .^ (double(group.perms) - 1));
    end

    % Spin-flip Z2: stop the ascending scan at C/2 (see the s=1/2 collector).
    has_flip = isfield(group, 'flip') && any(group.flip);
    C_half   = floor((double(d_loc)^N_sites - 1) / 2);

    % VRAM-adaptive (shared helper, 2^31-capped); byte-identical results.
    CHUNK = idivide(adaptive_enum_chunk(N_sites, use_gpu), int64(2));   % s>=1: heavier per state
    if ~use_gpu, CHUNK = int64(4e6); end       % gentler on CPU RAM

    out_cell = {};
    for r0 = int64(0) : CHUNK : (total - 1)
        r1   = min(r0 + CHUNK - 1, total - 1);
        % R3: generate the M-sector states directly (ascending) by unranking.
        cand = unrank_composition(r0, r1, N_sites, two_s, d_loc, D_cum, use_gpu);
        if has_flip
            if gather(cand(1)) > C_half, break; end       % all later chunks higher
            cand = cand(cand <= C_half);                  % tail of the boundary chunk
            if isempty(cand), break; end
        end

        % Local super-rep test (R2: staged early-reject on GPU -- most non-minima
        % die after a cheap block; only survivors get the full orbit-min test).
        if use_gpu
            keep = gather(cand(is_super_rep_Ih_gpu(cand, group, s_val, ppg)));
        else
            reps_c = min_image_Ih(cand, group, s_val);            % single output
            keep   = cand(reps_c == cand);
        end
        if ~isempty(keep)
            out_cell{end+1} = keep(:); %#ok<AGROW>
        end
        clear cand reps_c keep;
    end

    if isempty(out_cell)
        super_reps = int64([]);
    else
        super_reps = vertcat(out_cell{:});
    end
end


% ----------------------------------------------------------------
function states = unrank_composition(r0, r1, N_sites, two_s, d_loc, D_cum, use_gpu)
%UNRANK_COMPOSITION  Vectorised unranking of the digit-sum sector (s >= 1, R5).
%
%   Returns the int64 states whose ranks (ascending integer order among the
%   base-d_loc N-digit numbers with digit sum A_total) are r0..r1 inclusive.
%   General-d_loc analog of UNRANK_POPCOUNT. Walking position p from high to
%   low with budget A still to place, the digit is read straight off the
%   CUMULATIVE completion-count table D_cum from BUILD_D_TABLE:
%
%       D_cum(p,A,a) = #completions of the lower p positions with digit < a,
%       digit_p      = #{ a in 1..two_s : D_cum(p,A,a) <= r },
%       r            = r - D_cum(p,A,digit_p),   A = A - digit_p.
%
%   (R5) This replaces the previous per-digit subtract-and-guard inner loop
%   (no done/valid/max bookkeeping), cutting the per-element vectorised-op
%   count ~3x. Byte-identical to the old version: the cumulative subtracted
%   before choosing digit d is exactly sum_{v<d} D(p,A-v) = D_cum(p,A,d).
%   Reduces to UNRANK_POPCOUNT for two_s == 1. A_total recovered from D_cum's
%   2nd dim. Runs on host arrays or gpuArrays (states stays in VRAM for the
%   fused IS_SUPER_REP_IH_GPU / MIN_IMAGE_IH_GPU).

    nrows   = int64(size(D_cum, 1));                          % N_sites + 1
    nplane  = int64(size(D_cum, 1)) * int64(size(D_cum, 2));  % (N+1)*(A_total+1)
    dpow    = int64(d_loc) .^ int64(0 : N_sites - 1);         % [1 x N_sites] place values
    A_total = int64(size(D_cum, 2) - 1);
    r       = (r0 : r1).';
    n       = numel(r);

    if use_gpu
        r      = gpuArray(int64(r));
        D_cum  = gpuArray(D_cum);
        dpow   = gpuArray(dpow);
        states = gpuArray.zeros(n, 1, 'int64');
        A      = A_total * gpuArray.ones(n, 1, 'int64');
    else
        states = zeros(n, 1, 'int64');
        A      = A_total * ones(n, 1, 'int64');
    end

    for p = N_sites - 1 : -1 : 0
        % Linear index of D_cum(p+1, A+1, a+1) = (p+1) + A*nrows + a*nplane.
        % The (p+1)+A*nrows part is shared across a, so build it once; then
        % step the a-plane by nplane. digit = count of cumulative thresholds
        % the rank still covers.
        base  = int64(p + 1) + A .* nrows;       % [n x 1] -> a=0 plane (D_cum=0)
        digit = zeros(n, 1, 'like', states);     % int64
        idx   = base;
        for a = 1 : two_s
            idx   = idx + nplane;                % -> D_cum(p+1, A+1, a+1)
            digit = digit + int64(D_cum(idx) <= r);
        end
        % r_consumed = D_cum(p, A, digit): cumulative count for digits < digit.
        r      = r - D_cum(base + digit .* nplane);
        states = states + digit .* dpow(p + 1);
        A      = A - digit;
    end
end


% ----------------------------------------------------------------
function cache = empty_cache(s_val, M_target)
    cache.M_target   = M_target;
    cache.s_val      = s_val;
    cache.super_reps = int64([]);
    cache.orbit_lens = int32([]);
    cache.stab_flat  = zeros(0, 1, 'uint16');
    cache.stab_ptr   = int64(1);
end

