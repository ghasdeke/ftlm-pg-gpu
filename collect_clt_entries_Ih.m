function entries = collect_clt_entries_Ih(super_reps, bonds, s_val, J, group, lookup_method, entries_storage, ondisk_dir)
%COLLECT_CLT_ENTRIES_IH  Irrep-independent CLT entry collection (Phase 1).
%
%   ENTRIES = COLLECT_CLT_ENTRIES_IH(SUPER_REPS, BONDS, S_VAL, J, GROUP)
%
%   Performs the irrep-INDEPENDENT half of the CLT construction once per
%   M sector. The split is motivated by the fact that, within BUILD_CLT_PG_IH,
%   the 60 vectorised MIN_IMAGE_IH calls used to determine spin-flip
%   targets, the surviving-entry indexing, and the c_a coefficients
%   depend only on SUPER_REPS, BONDS, S_VAL, J and GROUP. They do NOT
%   depend on the chosen I_h irrep. Calling the monolithic builder 10
%   times per M sector therefore redoes ~30-50% of its total cost
%   needlessly. This function lifts that work to a single per-M call
%   and BUILD_CLT_FROM_ENTRIES_IH consumes the result for each irrep.
%
%   Outputs (packed into ENTRIES struct, sorted by target rep so the
%   gather SpMV can iterate contiguous slices per output rep):
%       super_reps        echoed (int64)
%       n_reps            numel(super_reps)
%       diag_vals         [n_reps x 1] Heisenberg diagonal Sz Sz sum
%                         (the internal digit matrix and the lookup struct
%                         are NOT exported -- no downstream reader)
%       src_sorted        [n_entries x 1] int32 source rep index (1-based)
%       tgt_sorted        [n_entries x 1] int32 target rep index (1-based)
%       g_sorted          [n_entries x 1] uint16 group element index (1-based;
%                         covers |G|<=65535 -> 2 B/entry vs int32's 4)
%       c_sorted          [n_entries x 1] double off-diagonal coefficient,
%                         OR empty (0x1) when c_is_const (s=1/2) -- then the
%                         single value lives in c_const (saves n_entries*8 B)
%       c_is_const        logical; true when every off-diagonal c equals
%                         c_const (the s=1/2 Heisenberg case)
%       c_const           double scalar 0.5*J (valid iff c_is_const)
%       n_entries         double, == numel(src_sorted)
%
%   Memory: the CLT lookup uses n_total / 32 * 8 bytes (exact 16x
%   compression vs the old dense int32 lookup). For s = 1/2 on the
%   icosidodecahedron (n_total = 2^30) this is 256 MB instead of 4 GB;
%   for s = 2 on the icosahedron (n_total = 5^12) it is 60 MB instead
%   of ~ 1 GB.
%
%   See also BUILD_CLT_LOOKUP, QUERY_CLT_LOOKUP,
%            BUILD_CLT_FROM_ENTRIES_IH, BUILD_CLT_PG_IH, MIN_IMAGE_IH.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    if nargin < 6 || isempty(lookup_method)
        lookup_method = 'bitmap';   % backward-compatible default
    end
    if nargin < 7 || isempty(entries_storage)
        entries_storage = 'host';   % backward-compatible default
    end
    if nargin < 8 || isempty(ondisk_dir), ondisk_dir = ''; end
    use_ondisk = ~isempty(ondisk_dir);    % out-of-core: stream entries to disk
    lookup_method   = lower(lookup_method);
    entries_storage = lower(entries_storage);
    % entries_storage='gpu' accumulates the FULL entry table as gpuArrays:
    % gate it early against the 2^31 elements-per-variable cap (upper bound
    % n_reps * 2*n_bonds entries) instead of crashing in the final vertcat
    % after the whole multi-minute collect (portability audit 2026-07-03).
    if strcmp(entries_storage, 'gpu')
        assert(double(numel(super_reps)) * 2 * size(bonds, 1) < 2^31, ...
            ['collect_clt_entries_Ih: entries_storage=''gpu'' cannot hold the ', ...
             'projected entry table (> 2^31 elements per gpuArray). Use ', ...
             'entries_storage=''host'' (optionally with entries_on_disk=true).']);
    end
    if ~ismember(entries_storage, {'host', 'gpu'})
        error('collect_clt_entries_Ih:storage', ...
            'Unknown entries_storage "%s". Expected ''host'' or ''gpu''.', entries_storage);
    end

    d_loc   = round(2*s_val + 1);
    N_sites = double(group.N);     % was hardcoded 12; now driven by group
    n_total = double(d_loc)^N_sites;
    n_reps  = numel(super_reps);
    n_b     = size(bonds, 1);
    powers  = int64(d_loc) .^ int64((0 : N_sites - 1)');

    %% State -> super_rep-index backend. Two coexisting options:
    %    'bitmap'  : 32-state bitmap CLT (16x compression vs dense
    %                int32 lookup, but still scales with n_total).
    %                Use for N <= ~30 where n_total / 32 * 8 B is OK.
    %    'schnack' : combinatorial-ranking (Schnack/Hage/Schmidt 2007),
    %                state -> M-sector-rank computed from digits in O(N)
    %                with a few-KB D_cum table; the per-rep auxiliary
    %                is sorted ranks (n_reps * 8 B). For N >= ~34 the
    %                bitmap becomes infeasible (17 GB at N=36) and
    %                Schnack-CR is mandatory.
    %  Both backends return queries via the same int32-or-zero vector
    %  contract (0 = state is not a super-rep, otherwise 1-based index).
    switch lookup_method
        case 'bitmap'
            assert(n_total <= 2^32, ...
                'collect_clt_entries_Ih: n_total too large for the 32-state bitmap CLT.');
            lookup_clt = build_clt_lookup(super_reps, n_total);
        case 'schnack'
            lookup_clt = build_lookup_schnack(super_reps, s_val, N_sites);
        otherwise
            error('collect_clt_entries_Ih:lookup_method', ...
                'Unknown lookup_method "%s". Expected ''bitmap'' or ''schnack''.', ...
                lookup_method);
    end

    %% Digit decomposition for diagonal Sz Sz (cached for the bond loop).
    %  int8 DIGITS instead of an n_reps x N_sites double m-matrix: the m
    %  values are exact half-integers digit - s, reconstructed per use as
    %  double(mi8) - s_val, so every downstream double is BIT-IDENTICAL --
    %  while the resident cost drops 8 -> 1 B per (rep, site). At the
    %  dodecahedron s=3/2 M=0 flip sector (n_reps ~ 3.6e8, N=20) this was a
    %  58 GB double held through the whole bond loop: the single largest
    %  collect-resident array, and what OOM-killed cgroup-limited SLURM
    %  allocations (2026-07 incident) despite entries_on_disk.
    mi8 = zeros(n_reps, N_sites, 'int8');
    if n_total <= 2^52
        % Float-exact digit path (proof in MIN_IMAGE_IH_GPU: the floor of a
        % correctly rounded x/p cannot cross an integer for x < 2^53); the
        % int64 mod/divide loop below is several-fold slower on the host.
        tmp = double(super_reps);
        for site = 1 : N_sites
            dg = rem(tmp, d_loc);
            mi8(:, site) = int8(dg);
            tmp = (tmp - dg) / d_loc;
        end
    else
        tmp = super_reps;
        for site = 1 : N_sites
            dg = double(mod(tmp, int64(d_loc)));
            mi8(:, site) = int8(dg);
            tmp = (tmp - int64(dg)) / int64(d_loc);
        end
    end
    m_col = @(site) double(mi8(:, site)) - s_val;   % exact half-integer m values

    diag_vals = zeros(n_reps, 1);
    for b = 1 : n_b
        diag_vals = diag_vals + J * m_col(bonds(b, 1)) .* m_col(bonds(b, 2));
    end

    %% Phase 1: vectorised spin-flip entry collection (streaming).
    %  We do NOT preallocate the upper bound n_reps * n_b * 2 (which at
    %  N=36 scale would be 9e9 entries -> 100+ GB). Each (bond, sign)
    %  contributes a small chunk; we accumulate in a cell list and
    %  concatenate at the end. Per-chunk transient is bounded by
    %  n_reps * 16 bytes (~ 20 MB at N=30, ~ 1.2 GB at N=36 -- still OK).
    src_cells  = cell(0, 1);
    tgt_cells  = cell(0, 1);
    g_cells    = cell(0, 1);
    cidx_cells = cell(0, 1);     % uint8 c-index per entry (non-const path)
    c_table    = zeros(0, 1);    % distinct c values, grown across chunks

    %% Off-diagonal coefficient constancy. For the s=1/2 Heisenberg model
    %  the spin-flip coefficient 0.5*J*sqrt(s(s+1)-m(m+1))*sqrt(s(s+1)-m(m-1))
    %  is EXACTLY 0.5*J for every allowed flip: m in {-1/2,+1/2} makes both
    %  sqrt factors equal sqrt(1)=1. Storing one scalar instead of an
    %  n_entries array of identical values saves ~n_entries*8 B (22.6 GB at
    %  N=36). For s>=1 the coefficient genuinely varies, so the full
    %  per-entry array is kept. When constant we still VERIFY it per chunk
    %  below, so a future model change can never silently corrupt results.
    %  For s>1/2 c varies but takes only a few distinct values (the spin
    %  matrix elements), bit-identical for identical (m_i,m_j,sign). We then
    %  store a uint8 INDEX into a tiny table of those values instead of a
    %  per-entry double: 8 B -> 1 B/entry, and no 10 GB double sort-copy.
    c_is_const  = (abs(s_val - 0.5) < 1e-12);
    c_const_val = 0.5 * J;

    % g (group element index, 1..|G|) is stored as uint16: covers |G| up to
    % 65535 (was uint8 for |I_h|=120; widened for space groups with |G|>255,
    % e.g. triangular N=36 |G|=432 / square 6x6 C_4v |G|=288). 2 B/entry vs
    % int32's 4. The CUDA kernel reads g as unsigned short (matching).
    assert(double(group.order) <= intmax('uint16'), ...
        'collect_clt_entries_Ih: |G|=%d exceeds uint16; widen the g type.', ...
        double(group.order));

    %% Out-of-core accumulation: stream entries to disk bucket files instead of
    %  RAM cells so collect never holds the full entry table (the Stage-2 lever
    %  vs the ~260 GB in-RAM sort peak). s=1/2 streams [src][g] (6 B/entry);
    %  s >= 1 additionally streams the uint8 c-index ([src][g][c_idx], 7 B/entry
    %  -- the dodecahedron s=3/2 companion-A extension). The bond loop below
    %  pushes each (bond,sign) chunk; the tail finalises an on-disk struct.
    if use_ondisk
        if ~exist(ondisk_dir, 'dir'), mkdir(ondisk_dir); end
        % Bucket count: bound the finalize in-RAM peak (~23 B x entries per
        % bucket) to ~2-3 GB using the PROJECTED entry count -- the old
        % reps-only rule gave ~8 GB buckets at dodec s=3/2 M=1 scale. Capped
        % at 192 buckets (4 open files each; stays well under fd limits).
        % SHARED with the estimate_feasibility od floor via EBS_BUCKET_COUNT
        % (single source; the floor must see the exact real bucket size).
        proj_e = 2 * double(n_reps) * n_b;
        ebs_n_buckets = ebs_bucket_count(n_reps, proj_e);
        ebs_h = ebs_open(ondisk_dir, n_reps, ebs_n_buckets, ~c_is_const);
    end

    %% R4: run the per-(bond,sign) orbit-minimum search on the GPU when a
    %  CUDA device is present. min_image is the dominant collect cost once
    %  the lookup is MEX-accelerated; MIN_IMAGE_IH_GPU returns BYTE-IDENTICAL
    %  rep + g_min (the perm_powers*digits matmul is exact in double < 2^53,
    %  so its min and argmin are identical to the CPU path) -- the two small
    %  outputs are gathered, leaving the downstream lookup/accumulation
    %  unchanged. perm_powers is cached once and reused across all 2*n_b calls.
    use_gpu_mi = false;
    ppg_mi     = [];
    try
        if gpuDeviceCount > 0
            use_gpu_mi = true;
            ppg_mi = gpuArray(double(d_loc) .^ (double(group.perms) - 1));
        end
    catch
        use_gpu_mi = false;
    end

    % Env-gated build diagnostics (FTLM_BUILD_DIAG=1): accumulate the wall
    % time of the four collect sub-phases so the parallelization target is
    % data-based (min_image GPU vs lookup vs c-index vs push-I/O).
    build_diag = ~isempty(getenv('FTLM_BUILD_DIAG'));
    bd_t_mi = 0; bd_t_lu = 0; bd_t_ci = 0; bd_t_push = 0; bd_t0 = tic;

    for b = 1 : n_b
        si = bonds(b, 1); sj = bonds(b, 2);
        m_si_col = m_col(si);           % transient doubles (2 columns per bond),
        m_sj_col = m_col(sj);           % freed each iteration -- not resident

        for sign_dir = 1 : 2
            if sign_dir == 1
                can = (m_si_col < s_val - 1e-10) & (m_sj_col > -s_val + 1e-10);
                if ~any(can), continue; end
                m_si_can = m_si_col(can);
                m_sj_can = m_sj_col(can);
                c_a_vec  = 0.5 * J ...
                    .* sqrt(s_val*(s_val+1) - m_si_can.*(m_si_can+1)) ...
                    .* sqrt(s_val*(s_val+1) - m_sj_can.*(m_sj_can-1));
                state_a_vec = super_reps(can) + powers(si) - powers(sj);
            else
                can = (m_si_col > -s_val + 1e-10) & (m_sj_col < s_val - 1e-10);
                if ~any(can), continue; end
                m_si_can = m_si_col(can);
                m_sj_can = m_sj_col(can);
                c_a_vec  = 0.5 * J ...
                    .* sqrt(s_val*(s_val+1) - m_si_can.*(m_si_can-1)) ...
                    .* sqrt(s_val*(s_val+1) - m_sj_can.*(m_sj_can+1));
                state_a_vec = super_reps(can) - powers(si) + powers(sj);
            end
            src_idx_can = int32(find(can));

            bd_tp = tic;
            if use_gpu_mi
                % CHUNKED (GPU-portability audit 2026-07-03): the un-chunked
                % call hands min_image_Ih_gpu the full candidate vector, whose
                % ~36-44 B/state device working set OOMs small cards at large
                % sectors, and whose internal [N_sites x n] digit matrix can
                % exceed the 2^31 elements-per-gpuArray cap. min_image is
                % elementwise per state -> chunking is byte-identical.
                n_can    = numel(state_a_vec);
                mi_chunk = min([64e6, ...
                                max(4e6, floor(gpu_free_bytes() * 0.25 / 44)), ...
                                floor(2^31 / N_sites * 0.9)]);
                rep_a_vec = zeros(n_can, 1, 'int64');
                g_min_vec = zeros(n_can, 1, 'int32');
                for mc0 = 1 : mi_chunk : n_can
                    mc1 = min(mc0 + mi_chunk - 1, n_can);
                    [rep_g, gmin_g] = min_image_Ih_gpu(state_a_vec(mc0:mc1), ...
                                                       group, s_val, ppg_mi);
                    rep_a_vec(mc0:mc1) = gather(rep_g);
                    g_min_vec(mc0:mc1) = gather(gmin_g);
                    clear rep_g gmin_g;
                end
            else
                [rep_a_vec, g_min_vec] = min_image_Ih(state_a_vec, group, s_val);
            end
            bd_t_mi = bd_t_mi + toc(bd_tp);
            bd_tp = tic;
            switch lookup_method
                case 'bitmap'
                    t_idx_vec = query_clt_lookup(lookup_clt, rep_a_vec);
                case 'schnack'
                    % MEX-accelerated single-pass rank+search when the
                    % kernel is built (SCHNACK_QUERY_MEX); auto-falls back
                    % to the MATLAB query_lookup_schnack otherwise.
                    % Bit-identical either way (gated by
                    % TEST_LOOKUP_SCHNACK_VS_BITMAP).
                    t_idx_vec = query_lookup_schnack_fast(lookup_clt, rep_a_vec);
            end
            bd_t_lu = bd_t_lu + toc(bd_tp);
            in_basis  = t_idx_vec > 0;
            if ~any(in_basis), continue; end

            % Stufe 6b: when entries_storage == 'gpu', upload each
            % chunk immediately to gpuArray cells. The cells hold
            % gpuArray references on host (~ few bytes each), the
            % actual entries live only in VRAM. Per-chunk host RAM
            % usage is bounded by the chunk size (~ few MB per bond),
            % not by the cumulative entry count.
            src_chunk = src_idx_can(in_basis);
            tgt_chunk = t_idx_vec(in_basis);
            g_chunk   = uint16(g_min_vec(in_basis));    % 1..|G| fits uint16

            % c handling: const (s=1/2) -> verify only; else map each value
            % to a uint8 index into a tiny growing table of distinct c's.
            % (Computed BEFORE the storage dispatch below so the on-disk path
            % can stream the c-index alongside src/g; values are identical to
            % the previous after-the-dispatch placement.)
            bd_tp = tic;
            cidx_chunk = zeros(0, 1, 'uint8');
            if c_is_const
                if any(abs(c_a_vec - c_const_val) > 1e-9 * max(abs(c_const_val), 1))
                    error('collect_clt_entries_Ih:cNotConst', ...
                          ['Off-diagonal c expected constant (=0.5*J) for ', ...
                           's=1/2 but a varying value was found.']);
                end
            else
                c_chunk = c_a_vec(in_basis);
                [tf, loc] = ismember(c_chunk, c_table);   % exact (deterministic floats)
                if ~all(tf)
                    c_table = [c_table; unique(c_chunk(~tf))]; %#ok<AGROW>
                    if numel(c_table) > 255
                        error('collect_clt_entries_Ih:cTooMany', ...
                              ['c has >255 distinct values; widen the c-index ', ...
                               'to uint16 (spin too large for a uint8 index).']);
                    end
                    [~, loc] = ismember(c_chunk, c_table);
                end
                cidx_chunk = uint8(loc);                  % 1-based index
            end

            bd_t_ci = bd_t_ci + toc(bd_tp);
            if use_ondisk
                bd_tp = tic;
                if c_is_const
                    ebs_h = ebs_push(ebs_h, src_chunk, tgt_chunk, g_chunk);
                else
                    ebs_h = ebs_push(ebs_h, src_chunk, tgt_chunk, g_chunk, cidx_chunk);
                end
                bd_t_push = bd_t_push + toc(bd_tp);
            else
                if strcmp(entries_storage, 'gpu')
                    src_chunk = gpuArray(src_chunk);
                    tgt_chunk = gpuArray(tgt_chunk);
                    g_chunk   = gpuArray(g_chunk);
                end
                src_cells{end+1, 1} = src_chunk; %#ok<AGROW>
                tgt_cells{end+1, 1} = tgt_chunk; %#ok<AGROW>
                g_cells{end+1,   1} = g_chunk;   %#ok<AGROW>
                if ~c_is_const
                    if strcmp(entries_storage, 'gpu'), cidx_chunk = gpuArray(cidx_chunk); end
                    cidx_cells{end+1, 1} = cidx_chunk; %#ok<AGROW>
                end
            end
        end
    end

    if build_diag
        fprintf(['    [build-diag] collect Bond-Loop %.1f s: min_image %.1f s | ', ...
                 'lookup %.1f s | c-idx %.1f s | ebs_push %.1f s | Rest %.1f s\n'], ...
            toc(bd_t0), bd_t_mi, bd_t_lu, bd_t_ci, bd_t_push, ...
            toc(bd_t0) - bd_t_mi - bd_t_lu - bd_t_ci - bd_t_push);
    end

    % Free the per-collect working sets before the final sort/concat peak.
    % `mi8` (n_reps x N_sites int8: ~7.3 GB at dodec s=3/2, ~2.3 GB at N=36)
    % and `lookup_clt` are used ONLY inside the bond loop above and are NOT
    % consumed downstream (verified: no reader anywhere), so we drop them here
    % instead of carrying them resident across the irrep loop.
    clear mi8 lookup_clt;

    %% On-disk finalisation: bucket-sort to one [src][g] (s=1/2) or
    %  [src][g][c_idx] (s>=1) file + the per-rep histogram, return a lightweight
    %  on-disk struct (no in-RAM entry arrays). For the indexed-c case the small
    %  c_table (grown in the bond loop above, EXACTLY as in the in-RAM path)
    %  rides in the struct -- the file holds only the per-entry uint8 indices.
    if use_ondisk
        sorted_path = fullfile(ondisk_dir, 'entries_sorted.bin');
        bd_tp = tic;
        [sorted_path, epr_od, n_coll_od] = ebs_finalize(ebs_h, sorted_path);
        if build_diag
            fprintf('    [build-diag] ebs_finalize %.1f s\n', toc(bd_tp));
        end
        entries = struct('super_reps', super_reps, 'n_reps', n_reps, ...
            'diag_vals', diag_vals, 'on_disk', true, 'sorted_path', sorted_path, ...
            'entries_per_rep', epr_od, 'n_entries', n_coll_od, ...
            'c_is_const', c_is_const, 'c_const', c_const_val, ...
            'c_is_indexed', ~c_is_const, 'c_table', c_table);
        return;
    end

    if isempty(src_cells)
        % Use the destination storage for the empty placeholders.
        if strcmp(entries_storage, 'gpu')
            all_src_flat = gpuArray(zeros(0, 1, 'int32'));
            all_tgt_flat = gpuArray(zeros(0, 1, 'int32'));
            all_g_flat   = gpuArray(zeros(0, 1, 'uint16'));
        else
            all_src_flat = zeros(0, 1, 'int32');
            all_tgt_flat = zeros(0, 1, 'int32');
            all_g_flat   = zeros(0, 1, 'uint16');
        end
    else
        all_src_flat = vertcat(src_cells{:});
        all_tgt_flat = vertcat(tgt_cells{:});
        all_g_flat   = vertcat(g_cells{:});
    end
    clear src_cells tgt_cells g_cells;
    n_coll = numel(all_src_flat);

    %% Sort entries by target rep (MATLAB's sort handles gpuArray transparently).
    %  Free each input the moment it has been reindexed, and narrow the
    %  permutation to uint32 (n_coll < 2^32), so this gather never holds all
    %  three inputs + all three outputs + a double `order` live simultaneously.
    %  That caps the peak of this sub-phase to ~one input + one output + the
    %  uint32 order; the output is byte-identical (same permutation values).
    [tgt_sorted, order] = sort(all_tgt_flat);
    clear all_tgt_flat;
    assert(numel(order) == n_coll, ...
        'collect_clt_entries_Ih: sort permutation length %d ~= n_coll %d', ...
        numel(order), n_coll);
    if n_coll <= intmax('uint32')
        order = uint32(order);             % halve the permutation footprint
    end
    src_sorted = all_src_flat(order);   clear all_src_flat;
    g_sorted   = all_g_flat(order);     clear all_g_flat;

    %% c representation. const (s=1/2) -> scalar c_const; s>1/2 -> uint8
    %  index c_idx into the distinct-value table c_table. c_sorted stays
    %  empty in both cases: no per-entry double (10 GB at s=7/2 / N=12) and
    %  no double-sized copy during the sort.
    c_sorted = zeros(0, 1);
    if c_is_const
        c_is_indexed = false;
        c_idx        = zeros(0, 1, 'uint8');
        c_table      = zeros(0, 1);
    else
        c_is_indexed = true;
        if isempty(cidx_cells)
            if strcmp(entries_storage, 'gpu')
                all_c_idx = gpuArray(zeros(0, 1, 'uint8'));
            else
                all_c_idx = zeros(0, 1, 'uint8');
            end
        else
            all_c_idx = vertcat(cidx_cells{:});
        end
        c_idx = all_c_idx(order);          % reorder to match src/tgt/g_sorted
        clear all_c_idx;
    end
    clear cidx_cells order;

    %% Pack
    %  NB: entries.mi and entries.lookup_clt were removed (2026-06-03) -- they
    %  were stored but never read by any downstream consumer, and entries.mi
    %  alone is ~70 GB at N=36. Both are now cleared after the bond loop.
    entries.super_reps   = super_reps;
    entries.n_reps       = n_reps;
    entries.diag_vals    = diag_vals;
    entries.src_sorted   = src_sorted;
    entries.tgt_sorted   = tgt_sorted;
    entries.g_sorted     = g_sorted;
    entries.c_sorted     = c_sorted;      % [] (0x1); kept for back-compat
    entries.c_is_const   = c_is_const;    % true for s=1/2
    entries.c_const      = c_const_val;   % scalar 0.5*J (valid iff c_is_const)
    entries.c_is_indexed = c_is_indexed;  % true for s>1/2 (uint8 index path)
    entries.c_table      = c_table;       % distinct c values (double), iff indexed
    entries.c_idx        = c_idx;         % [n_entries x 1] uint8, 1-based into c_table
    entries.n_entries    = n_coll;
end
