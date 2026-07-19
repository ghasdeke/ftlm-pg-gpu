function eskel = build_entry_skeleton_Ih(entries, force_b2, tile_cap, force_unpacked)
%   *** PRODUCTION ***  irrep-INDEPENDENT skeleton half, built ONCE per M sector
%   by ftlm_observables_pg_gpu_Ih and shared across irreps. Paired with the
%   per-irrep build_clt_skeleton_from_entries_Ih. (Builder roles: see that file.)
%BUILD_ENTRY_SKELETON_IH  Irrep-INDEPENDENT half of the skeleton CLT (A1-ph2).
%
%   ESKEL = BUILD_ENTRY_SKELETON_IH(ENTRIES)
%
%   Builds, ONCE per M sector, the parts of the skeleton CLT that depend only
%   on ENTRIES and NOT on the irrep: the per-entry index/coefficient arrays
%   (src_idx, g_idx, and either the uint8 c_idx + c_table or the per-entry
%   c_a, with constant-c detection), the per-target-rep entry counts/offsets,
%   and the diagonal Sz.Sz vector -- all returned as gpuArrays.
%
%   BUILD_CLT_SKELETON_FROM_ENTRIES_IH then takes this ESKEL plus the per-irrep
%   data (reps, V, eig, n_per_rep, irrep matrices) and assembles the full skel
%   by REUSING these gpuArray handles, so the ~8 GB of per-entry arrays are
%   uploaded ONCE instead of once per irrep. Across the 10 I_h irreps of one M
%   sector this removes ~9 redundant builds+uploads of the entry data.
%
%   Uses the A1 full-rep indexing: entries reference full super-rep indices
%   (1..n_reps_full) and are kept AS-IS (no active-rep compaction). Inactive
%   reps get n_per_rep = 0 in the per-irrep skeleton and are skipped by the
%   kernel; entries referencing them contribute exactly zero.
%
%   The caller MUST keep ESKEL alive for as long as any skeleton built from it
%   is in use (the CUDA init_skel_ref path borrows these device pointers).
%
%   See also BUILD_CLT_SKELETON_FROM_ENTRIES_IH, COLLECT_CLT_ENTRIES_IH,
%            RUN_FTLM_PG_SECTOR_GPU_IH.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    %% Out-of-core: entries already bucket-sorted on disk (collect_clt_entries_Ih
    %  with ondisk_dir). Build the skeleton from the provided per-rep histogram +
    %  a memory-mapping of the [src][g] file -- no in-RAM entry arrays ever exist.
    if isfield(entries, 'on_disk') && entries.on_disk
        if nargin < 3 || isempty(tile_cap), tile_cap = 64e6; end
        eskel = entry_skeleton_ondisk(entries, tile_cap);
        return;
    end

    c_is_const   = isfield(entries, 'c_is_const')   && entries.c_is_const;
    c_is_indexed = isfield(entries, 'c_is_indexed') && entries.c_is_indexed;
    n_reps_full  = numel(entries.super_reps);
    n_coll       = numel(entries.src_sorted);

    %% B2 entry-tiling gate: keep the per-ENTRY arrays on the HOST when either
    %  (a) n_entries exceeds the ~2^31 max-elements cap of ONE MATLAB gpuArray
    %      (the kernel's init_skel_b2 cudaMalloc's >2^31 fine), or
    %  (b) the PROJECTED resident table would not comfortably fit the card
    %      (> 0.5 x free VRAM): small GPUs then take the bit-identical
    %      B2 -> streaming degradation ladder in run_ftlm instead of OOMing
    %      at the upload. (GPU-portability audit 2026-07-03: the old pure
    %      element-count gate crashed 8-16 GB cards on tables the streaming
    %      path handles, while estimate_feasibility even PREDICTED streaming.)
    %  force_b2=true forces it (for small-case tests).
    if nargin < 2 || isempty(force_b2),        force_b2 = false;        end
    if nargin < 4 || isempty(force_unpacked),  force_unpacked = false;  end

    %  G2 packing decision (host-decidable up front; also feeds the byte
    %  projection): when the full-rep count fits in 25 bits AND g fits in
    %  7 bits, pack src(25b)|g(7b) into one uint32 -> 4 B for src+g instead
    %  of 4+2. Else separate arrays (uint16 g); mutually exclusive.
    PACK_BITS = 25;
    if isempty(entries.g_sorted)
        g_max = 0;     % no off-diagonal entries (e.g. fully polarised M=M_max)
    else
        g_max = double(max(entries.g_sorted(:)));
    end
    is_packed = ~force_unpacked && ...
                (n_reps_full < bitshift(1, PACK_BITS)) && (g_max < bitshift(1, 32 - PACK_BITS));

    %  Projected resident-table bytes/entry: g projected as uint16 (the uint8
    %  narrowing exists only ON the B2 path); a non-const/non-indexed c is
    %  projected at 4 B/entry (conservative -- the near-constant collapse
    %  below may still drop it, which only pushes borderline cases toward
    %  B2, never toward an OOM).
    if is_packed, bpe_proj = 4; else, bpe_proj = 4 + 2; end
    if c_is_indexed
        bpe_proj = bpe_proj + 1;
    elseif ~c_is_const
        bpe_proj = bpe_proj + 4;
    end
    is_b2 = force_b2 || (n_coll > 2e9) || ...
            (n_coll > 0 && n_coll * bpe_proj > 0.5 * gpu_free_bytes());
    if is_b2, wrap = @(x) x;            % keep on host
    else,     wrap = @(x) gpuArray(x);  % upload (resident-when-fits fast path)
    end

    %% Per-entry index arrays (full-rep, AS-IS -> no keep/remap).
    g_is_u8 = false;
    if is_packed
        srcg      = bitor(uint32(entries.src_sorted), ...
                          bitshift(uint32(entries.g_sorted), PACK_BITS));
        skel_srcg = wrap(srcg);
        skel_src  = wrap(zeros(0, 1, 'int32'));    % empty -> kernel reads srcg
        skel_g    = wrap(zeros(0, 1, 'uint16'));
    else
        skel_srcg = wrap(zeros(0, 1, 'uint32'));
        skel_src  = wrap(entries.src_sorted);      % int32, 1-based full-rep
        % G1b: on the B2 (host-resident / streaming) path, narrow g to uint8 when
        % |G| <= 255 (e.g. kagome N=36 |G|=144). Halves the ~4.6 GB host g table
        % AND the per-SpMV streamed g-tile. The CUDA init_skel_b2/stream modes
        % detect the uint8 class and the OTF kernel reads it via g_idx8. Resident
        % non-B2 paths (init_skel/init_skel_ref) are untouched -> stay uint16.
        g_is_u8 = is_b2 && (g_max > 0) && (g_max < 256);
        if g_is_u8
            skel_g = wrap(uint8(entries.g_sorted));
        else
            skel_g = wrap(uint16(entries.g_sorted)); % uint16 (G1)
        end
    end

    %% c representation + constant-c detection (all irrep-independent).
    if c_is_const
        c_a_const    = single(entries.c_const);
        c_a_is_const = true;
        skel_c_a     = wrap(single(zeros(0, 1)));
    elseif c_is_indexed
        c_a_const    = single(0);
        c_a_is_const = true;                           % indexed -> empty c_a
        skel_c_a     = wrap(single(zeros(0, 1)));
    else
        c_sorted_h = single(entries.c_sorted);
        if isempty(c_sorted_h)
            c_a_const    = single(0);
            c_a_is_const = true;
            skel_c_a     = wrap(single(zeros(0, 1)));
        else
            c_min = gather(min(c_sorted_h));
            c_max = gather(max(c_sorted_h));
            c_rel = single(c_max - c_min) / max(abs(single(c_max)), single(1e-30));
            c_a_is_const = (c_rel < single(1e-5));
            if c_a_is_const
                c_a_const = single(gather(c_sorted_h(1)));
                skel_c_a  = wrap(single(zeros(0, 1)));
            else
                c_a_const = single(0);
                skel_c_a  = wrap(c_sorted_h);
            end
        end
    end
    if c_is_indexed
        skel_c_idx   = wrap(entries.c_idx);                    % uint8
        skel_c_table = gpuArray(single(entries.c_table(:)));   % small -> stays gpuArray
    else
        skel_c_idx   = wrap(zeros(0, 1, 'uint8'));
        skel_c_table = gpuArray(single(zeros(0, 1)));
    end

    %% Per-target-rep entry counts + offsets over the FULL rep set.
    %  CHUNKED accumulate: a single accumarray(double(tgt_sorted),...) would
    %  materialise an 18.4 GB transient double at N=36 (2.3e9 entries) on top of
    %  the ~31 GB resident entries -> a ~49 GB host spike. Counting in chunks
    %  bounds the transient to ~CH*8 B (~1.6 GB) with no change to the result
    %  (counts are additive). Bit-identical to the single accumarray.
    n_coll_tg = numel(entries.tgt_sorted);
    entries_per_rep_h = zeros(n_reps_full, 1, 'int32');
    CH = 2e8;
    for a = 1 : CH : n_coll_tg
        b = min(a + CH - 1, n_coll_tg);
        c = accumarray(double(gather(entries.tgt_sorted(a:b))), 1, [n_reps_full, 1]);
        entries_per_rep_h = entries_per_rep_h + int32(c);
    end
    clear c;
    % int64: the cumulative entry offset reaches n_entries, which exceeds
    % int32 (2^31) for large sectors (e.g. N=36 kagome n_entries ~2.3e9). The
    % CUDA kernel reads entry_offsets as int64 (B2). Per-rep counts stay int32.
    entry_offsets_h   = int64([0; cumsum(double(entries_per_rep_h(1:end-1)))]);

    %% Streaming-B2 rep-tile partition (irrep-INDEPENDENT, built once per M when
    %  is_b2). Entries are sorted by target rep, so a contiguous rep range maps
    %  to a contiguous entry slice. We greedily cut tiles at rep boundaries so
    %  each tile holds <= ~tile_cap entries AND every rep is wholly inside one
    %  tile (-> the per-rep OTF kernel writes W[t] once, no atomics). RUN_FTLM
    %  uses this to STREAM the host entry table tile-by-tile per SpMV for the
    %  large-d blocks that cannot keep the ~11.7 GB table resident.
    if is_b2 && n_coll > 0
        % Default ~64M entries/tile -> ~320 MB device buffer (src+g 5 B/entry),
        % ~36 tiles at N=36 (n_entries ~2.3e9). Lowered from 200e6 (June 2026):
        % a 320 MB chunk still saturates PCIe (copy efficiency unchanged; more
        % tiles only means more cheap launches), while the finer granularity
        % (a) shrinks the per-SpMV device tile buffer(s) -- x2 with Lever A's
        % ping-pong pair -- and (b) lets the RESIDENT-PREFIX (init_skel_stream)
        % pack the leftover VRAM more tightly, since the prefix is cut at tile
        % boundaries. Both effects free VRAM for the prefix on the big blocks.
        if nargin < 3 || isempty(tile_cap), tile_cap = 64e6; end
        cum    = double(entry_offsets_h);              % n_reps, 0-based first-entry offset
        tof    = floor(cum / double(tile_cap));        % tile index per rep (monotone non-decr.)
        starts = [1; find(diff(tof) > 0) + 1];         % 1-based rep starting each tile
        e_end  = [cum(starts(2:end)); double(n_coll)]; % entry end of each tile
        eskel.tile_rep_ptr = int32([starts - 1; n_reps_full]);   % 0-based rep bounds, nt+1
        eskel.tile_e_start = int64(cum(starts));                 % entry base per tile, nt
        eskel.tile_e_count = int64(e_end - cum(starts));         % entry count per tile, nt
    end

    %% Diagonal Sz.Sz per rep (full, irrep-independent).
    skel_diag = gpuArray(single(entries.diag_vals));

    %% Pack
    eskel.is_entry_skeleton = true;
    eskel.is_b2             = is_b2;    % per-entry arrays kept on HOST (B2 tiling)
    eskel.n_reps_full       = n_reps_full;
    eskel.n_coll            = n_coll;
    eskel.src_idx           = skel_src;
    eskel.g_idx             = skel_g;
    eskel.srcg              = skel_srcg;    % packed src|g uint32 (G2), empty if not packed
    eskel.is_packed         = is_packed;
    eskel.g_is_u8           = g_is_u8;      % G1b: g_idx stored as uint8 (|G|<=255, B2 path)
    eskel.c_a               = skel_c_a;
    eskel.c_a_const         = c_a_const;
    eskel.c_a_is_const      = c_a_is_const;
    eskel.c_is_indexed      = c_is_indexed;
    eskel.c_idx             = skel_c_idx;
    eskel.c_table           = skel_c_table;
    eskel.entries_per_rep   = gpuArray(entries_per_rep_h);
    eskel.entry_offsets     = gpuArray(entry_offsets_h);
    eskel.diag_vals         = skel_diag;
end

% ----------------------------------------------------------------
function eskel = entry_skeleton_ondisk(entries, tile_cap)
%ENTRY_SKELETON_ONDISK  Skeleton for the out-of-core path: no in-RAM entries.
    assert(exist('mmap_file', 'file') == 3, ...
        'build_entry_skeleton_Ih: on-disk path needs the mmap_file MEX (build_all).');
    n_reps_full = double(numel(entries.super_reps));
    n_coll      = double(entries.n_entries);
    epr_h           = int32(entries.entries_per_rep(:));
    entry_offsets_h = int64([0; cumsum(double(epr_h(1:end-1)))]);
    c_is_indexed = isfield(entries, 'c_is_indexed') && entries.c_is_indexed;

    %% Diagonal-only sector (n_entries == 0, e.g. the fully polarised M = M_max
    %  of a full sweep): the sorted file is EMPTY and cannot be memory-mapped
    %  (POSIX mmap fails with EINVAL; Windows CreateFileMapping also rejects
    %  zero-length files). Return a normal RESIDENT empty-entry skeleton
    %  instead (is_b2 = false, no tile partition) -- run_ftlm then takes the
    %  standard init_skel_ref path, which already handles zero entries; the
    %  driver's force_stream is moot without a partition.
    if n_coll == 0
        eskel.is_entry_skeleton = true;
        eskel.is_b2        = false;
        eskel.on_disk      = true;
        eskel.mmap_handle  = [];                 % nothing mapped; end-of-M guard skips
        eskel.n_reps_full  = n_reps_full;
        eskel.n_coll       = 0;
        eskel.src_idx      = gpuArray(zeros(0, 1, 'int32'));
        eskel.g_idx        = gpuArray(zeros(0, 1, 'uint16'));
        eskel.srcg         = gpuArray(zeros(0, 1, 'uint32'));
        eskel.is_packed    = false;
        eskel.g_is_u8      = false;
        eskel.c_a          = gpuArray(zeros(0, 1, 'single'));
        if c_is_indexed, eskel.c_a_const = single(0);
        else,            eskel.c_a_const = single(entries.c_const); end
        eskel.c_a_is_const = true;
        eskel.c_is_indexed = c_is_indexed;
        eskel.c_idx        = gpuArray(zeros(0, 1, 'uint8'));
        eskel.c_table      = gpuArray(single(zeros(0, 1)));   % nothing to index
        eskel.entries_per_rep = gpuArray(epr_h);
        eskel.entry_offsets   = gpuArray(entry_offsets_h);
        eskel.diag_vals       = gpuArray(single(entries.diag_vals));
        return;
    end

    % Memory-map the sorted [ src int32 ][ g uint16 ] file (6 B/entry, s=1/2)
    % or [ src ][ g ][ c_idx uint8 ] (7 B/entry, s>=1 indexed c) -> raw host ptrs.
    bpe_file = 6 + double(c_is_indexed);
    [base, nb] = mmap_file('open', entries.sorted_path);
    assert(double(nb) == n_coll * bpe_file, ...
        'build_entry_skeleton_Ih: on-disk file %d B != expected %d B', ...
        double(nb), n_coll * bpe_file);

    % Rep-tile partition (same greedy cut at rep boundaries as the resident B2 path).
    cum    = double(entry_offsets_h);
    tof    = floor(cum / double(tile_cap));
    starts = [1; find(diff(tof) > 0) + 1];
    e_end  = [cum(starts(2:end)); n_coll];
    eskel.tile_rep_ptr = int32([starts - 1; n_reps_full]);
    eskel.tile_e_start = int64(cum(starts));
    eskel.tile_e_count = int64(e_end - cum(starts));

    eskel.is_entry_skeleton = true;
    eskel.is_b2        = true;
    eskel.on_disk      = true;
    eskel.mmap_handle  = base;                               % close at end of M
    eskel.mmap_src     = base;                               % src block at offset 0
    eskel.mmap_g       = base + uint64(n_coll) * uint64(4);  % g block after the int32 src block
    if c_is_indexed                                          % c_idx block after the uint16 g block
        eskel.mmap_cidx = base + uint64(n_coll) * uint64(6);
    end
    eskel.n_reps_full  = n_reps_full;
    eskel.n_coll       = n_coll;
    eskel.src_idx      = zeros(0, 1, 'int32');               % streamed from the mapping
    eskel.g_idx        = zeros(0, 1, 'uint16');
    eskel.srcg         = zeros(0, 1, 'uint32');
    eskel.is_packed    = false;
    eskel.g_is_u8      = false;             % on-disk mmap g block is uint16 (6/7 B/entry layout)
    eskel.c_a          = zeros(0, 1, 'single');
    if c_is_indexed
        eskel.c_a_const = single(0);
        eskel.c_table   = gpuArray(single(entries.c_table(:)));   % small, resident
    else
        eskel.c_a_const = single(entries.c_const);
        eskel.c_table   = gpuArray(single(zeros(0, 1)));
    end
    eskel.c_a_is_const = true;              % no per-entry single c_a either way
    eskel.c_is_indexed = c_is_indexed;
    eskel.c_idx        = zeros(0, 1, 'uint8');   % streamed from the mapping (mmap_cidx)
    eskel.entries_per_rep = gpuArray(epr_h);
    eskel.entry_offsets   = gpuArray(entry_offsets_h);
    eskel.diag_vals       = gpuArray(single(entries.diag_vals));
end
