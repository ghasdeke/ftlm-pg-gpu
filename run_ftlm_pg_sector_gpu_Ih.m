function [E_sec, w_sec, B_used, vram_peak_gb] = run_ftlm_pg_sector_gpu_Ih( ...
                                                            clt, R, M_lz, ...
                                                            B_gpu, seed, gpu_h, ...
                                                            mem_diag, lz_diag)
%RUN_FTLM_PG_SECTOR_GPU_IH  Block-Lanczos FTLM on the GPU for one
%                          (M, Gamma) sector of the icosahedron under I_h.
%
%   [E_SEC, W_SEC, B_USED] = RUN_FTLM_PG_SECTOR_GPU_IH(CLT, R, M_LZ, ...
%                                                      B_GPU, SEED, GPU_H)
%
%   runs FTLM with R random complex starting vectors on the
%   symmetry-adapted Hamiltonian for one (M, Gamma) block, using the
%   precomputed compressed lookup table CLT from BUILD_CLT_PG_IH. The
%   FP32 complex block-Lanczos kernel
%   CUDA_LANCZOS_CLUT_BLOCK_PG_IH executes the SpMV by gather over
%   precomputed d_Gamma x d_Gamma matrix-element kernels, with no
%   runtime min-image search and no atomicAdds.
%
%   The FTLM weight per Ritz value is the standard
%       w_k = (D_eff / R_eff) * |q_k^(1)|^2
%   where D_eff = clt.n_basis is the (M, Gamma) block dimension at the
%   chosen Gamma row alpha = 1. The d_Gamma partner-row multiplicity
%   that aggregates the (M, Gamma) result into the full Hilbert
%   spectrum is applied OUTSIDE this routine by the caller
%   (FTLM_OBSERVABLES_PG_GPU_IH).
%
%   Inputs:
%       clt        struct from BUILD_CLT_PG_IH
%       R          number of FTLM random vectors
%       M_lz       Lanczos steps per random vector
%       B_gpu      block size (0 = adaptive)
%       seed       deterministic seed for this sector
%       gpu_h      gpuDevice handle
%
%   Outputs:
%       E_sec      column of concatenated Ritz values
%       w_sec      column of FTLM weights (without the d_Gamma factor)
%       B_used     actual block size used (clamped to R_eff and MAX_B)
%
%   See also BUILD_CLT_PG_IH, SPMV_PG_IH_CLT_MATLAB,
%            CUDA_LANCZOS_CLUT_BLOCK_PG_IH, RUN_FTLM_PG_SECTOR_GPU_CPLX.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    %% 64-bit basis-offset ABI handshake (v2, 2026-07), once per session: a
    %  stale pre-audit MEX would read the int64 offsets as int32 and compute
    %  silent garbage. Shared helper (also used by the standalone kernel
    %  tests); rethrows genuine GPU errors instead of blaming the build.
    assert_kernel_abi();

    if nargin < 7, mem_diag = false; end
    if nargin < 8, lz_diag  = false; end

    n_basis   = clt.n_basis;
    n_reps    = clt.n_reps;
    d_irrep   = clt.d_irrep;
    % Sum in DOUBLE: entries_per_rep is int32 and MATLAB's native integer
    % sum SATURATES at 2^31-1 (N=36 kagome B2 already has 2.3e9 entries).
    n_entries = gather(double(sum(clt.entries_per_rep(:), 'double')));

    D_eff       = n_basis;
    R_eff       = min(R, D_eff);
    M_lz_actual = min(M_lz, D_eff);

    %% Decide B_used: VRAM-ADAPTIVE block size. The block-Lanczos holds ~12
    %  fp32 vector buffers of n_basis*B (raw cudaMalloc, outside MATLAB's
    %  gpuArray pool), so we size B to the card's FREE VRAM with margin -- the
    %  SAME code then runs on an 8 GB or an 80 GB GPU, and on large n_basis,
    %  without manual tuning. R random vectors are processed in ceil(R/B_used)
    %  blocks (the loop below), so a smaller B only adds blocks, never drops
    %  random vectors. MAX_B must match the kernel's #define MAX_B: it is the
    %  HARD cap and is set by register pressure, NOT VRAM -- raising it past 8
    %  re-introduces the local-memory spills on the d=4,5 irreps that made it
    %  slower, so a bigger card cannot use a bigger block; VRAM here only
    %  matters for DOWN-scaling B (portability / very large sectors).
    MAX_B = 8;                                           % == cuda_lanczos_clut_block_pg_Ih #define MAX_B
    % Real FP32 path (clt.is_real, June 2026): the kernel holds only the three
    % re Krylov buffers and block_lanczos takes ONE V0 RNG buffer -> the
    % per-column footprint HALVES (BUF_FACTOR 8 -> 4). This is what fits the
    % dodecahedron s=3/2 d=4/d=5 blocks on a 48 GB card (complex needs 46/58 GB).
    is_real_clt = isfield(clt, 'is_real') && clt.is_real;
    % Bytes per Krylov column: the block-Lanczos holds 8 fp32 buffers of n_basis
    % (6 cudaMalloc'd in init_skel_* -- v/vp/w re+im -- plus the 2 V0 RNG buffers
    % in 'block_lanczos'); reduction workspace is covered by the absolute
    % 0.5 GB slack in vram_budget. BUF_FACTOR is the REAL count (8, down from 9
    % after the s_d_tmp_col staging buffer was removed -- V0 is now transposed
    % straight from the borrowed gpuArray). A larger factor here falsely refused
    % B=1 on the big N=36 blocks (e.g. the d=6 streaming block fits but an inflated
    % factor over-reported). The VRAM pre-check below uses 9x (real: 5x) as the
    % final gate.
    if is_real_clt, BUF_FACTOR = 4; else, BUF_FACTOR = 8; end
    % B_vram MUSS mit dem Skeleton-Pre-Flight (need_bytes-Gate unten, Z.~353:
    % real fp32 20 B/Elem-Spalte, FTLM_FP16 12.5, komplex 36) konsistent
    % sein: die alte, optimistischere Formel (16/10/32 B) liess auf der
    % 20-GB-Karte B=8 zu, das der Pre-Flight dann zu Recht ablehnte
    % (kagome-N=36-Benchmark, 2026-07-11: 13.1 GB Bedarf vs 12.0 GB frei,
    % harter Abbruch statt kleinerem B). Frueher verdeckte der Gather-Cap
    % (B=1) die Luecke. Invariante jetzt: jedes B, das B_vram zulaesst,
    % passiert auch das Gate. FP16 halbiert die drei Kernel-Puffer ->
    % 12.5 B/Elem (V0-RNG bleibt fp32; im Segmented-Regime konservativ).
    if is_real_clt
        per_elem = 20;                                        % == 5*vec-Gate
        if strcmp(getenv('FTLM_FP16'), '1'), per_elem = 12.5; end
        bytes_per_col = double(n_basis) * per_elem + 4e6;
    else
        bytes_per_col = double(n_basis) * 36 + 4e6;           % == 9*vec-Gate
    end

    %% Resident-vs-STREAMING decision + VRAM-adaptive B (B2-aware).
    %  init_skel_b2 uploads the ~11.7 GB per-entry table to VRAM and keeps it
    %  RESIDENT, then allocates the n_basis*B Lanczos buffers -- so for the large
    %  -d N=36 blocks (d>=3) the table + buffers do not both fit. STREAMING-B2
    %  instead keeps the table on the HOST and streams it in rep-tiles per SpMV,
    %  so VRAM only reserves ONE tile buffer (a few GB) alongside the buffers.
    %  We pick streaming when the resident table + a single Lanczos column would
    %  not fit (or when a test forces it). `reserved` is what stays resident on
    %  the GPU for the SpMV: the full table (resident B2) vs one tile (streaming)
    %  vs 0 (non-B2: entries are already resident gpuArrays, counted in Avail).
    is_b2_clt    = isfield(clt, 'is_b2') && clt.is_b2;
    has_part     = isfield(clt, 'tile_rep_ptr') && ~isempty(clt.tile_rep_ptr);
    force_stream = isfield(clt, 'force_stream') && clt.force_stream;
    g_bytes      = 2;                                % uint16 g (default)
    if isfield(clt, 'g_is_u8') && clt.g_is_u8, g_bytes = 1; end   % uint8 g (|G|<=255, B2)
    table_bytes  = 0;
    if is_b2_clt
        table_bytes = numel(clt.src_idx) * 4 ...       % int32 src (empty if packed)
                    + numel(clt.g_idx)   * g_bytes ...  % uint16/uint8 g (empty if packed)
                    + numel(clt.c_a)     * 4;           % single per-entry c (empty if const/indexed)
        if isfield(clt, 'srcg'),  table_bytes = table_bytes + numel(clt.srcg)  * 4; end  % uint32 packed src|g
        if isfield(clt, 'c_idx'), table_bytes = table_bytes + numel(clt.c_idx) * 1; end  % uint8 c-index (s>1/2)
    end
    % min() with gpu_free_bytes: identical to gpu_h.AvailableMemory in
    % production; under FTLM_FAKE_FREE_VRAM_GB it lets tests emulate a
    % SMALLER card safely (never inflates beyond the real free memory).
    avail = min(double(gpu_h.AvailableMemory), gpu_free_bytes());
    % Resident-vs-streaming need for B=1: GATE-KONSISTENT zur B_vram-Formel
    % unten (Invariante: resident gewaehlt => B_vram >= 1). Die alte Form
    % table + (bytes_per_col-4e6) + 0.7e9 liess bei grossem n_basis ein
    % GB-Fenster zu, in dem resident gewaehlt, aber B_vram = 0 wurde ->
    % harter Abbruch statt Streaming auf Karten mittlerer Groesse
    % (Robustheits-Audit 2026-07-11). Sichere Richtung: kippt minimal
    % frueher auf Streaming; FP16-Residency-Synergie bleibt erhalten.
    resident_b1_need = table_bytes + (bytes_per_col + 0.5e9) / 0.90;
    use_stream = (is_b2_clt && has_part && resident_b1_need > avail) || (has_part && force_stream);

    if use_stream
        bpe        = 4 + g_bytes;                        % src(4) + g(g_bytes) per entry
        if isfield(clt, 'srcg') && ~isempty(clt.srcg), bpe = 4; end   % packed uint32
        has_cidx_arr  = isfield(clt, 'c_idx')     && ~isempty(clt.c_idx);
        has_cidx_mmap = isfield(clt, 'mmap_cidx') && ~isempty(clt.mmap_cidx);
        assert(~(has_cidx_arr && has_cidx_mmap), ...
            'run_ftlm_pg_sector_gpu_Ih: clt carries BOTH a resident c_idx and a mapped mmap_cidx.');
        if has_cidx_arr || has_cidx_mmap
            bpe = bpe + 1;                               % + streamed uint8 c-index tile (s>=1)
        end
        reserved   = double(max(clt.tile_e_count)) * bpe;             % one device tile buffer
    else
        reserved   = table_bytes;                        % resident table (B2) or 0
    end
    free_for_block = avail - reserved;
    vram_budget    = 0.90 * free_for_block - 0.5e9;      % margin + V0 RNG
    B_vram         = floor(vram_budget / bytes_per_col);

    %% Gather-aware cap on the adaptive block size (B_GATHER_CAP).
    %  A larger block B ONLY amortises the resident entry-table reread: the SpMV
    %  makes ceil(R/B) passes over the table, so B>1 cuts that pass count. But
    %  the dominant SpMV cost is the RANDOM V[src] gather, and that is
    %  B-INVARIANT in total (R columns are gathered either way). When one reduced
    %  Krylov column V_per_col = n_basis*8 B does NOT fit the L2 cache, the gather
    %  is DRAM-random-bound and dominates the time; then a larger B buys almost no
    %  speed yet costs B*bytes_per_col of VRAM (this is what pushed the N=36 d=6
    %  block to ~19.4/20.5 GB at B=2) and inflates the random working set. So once
    %  clearly gather-bound we cap B=1 -> frees VRAM (headroom / larger problems)
    %  at ~no speed cost. Cache-resident / small blocks (V_per_col below the
    %  threshold) keep the full MAX_B so their relatively expensive table reread
    %  is still amortised. An explicit B_gpu>0 overrides the cap (benchmark/force).
    %  NB: the s=1 mid-block oversubscription was NOT a B problem (B was always
    %  picked = MAX_B); its true cause was device-heap fragmentation from the
    %  ascending-size per-irrep alloc/free cycles. (A speculative largest-first
    %  irrep reorder in the driver was chased and REVERTED on 2026-06-06, see
    %  Irreps run in TABLE ORDER, ascending d,
    %  in ftlm_observables_pg_gpu_Ih.)
    %  K2d/K7b amendments to the cap:
    %    (a) FP16-aware working set: the fp16 gather reads __half, so one
    %        reduced column is 2 B/element there (blocks with n_basis in
    %        (64e6,128e6] were needlessly capped to B=1 under FTLM_FP16).
    %    (b) L2-size from the device via the kernel's 'l2_size' query mode
    %        (parallel.gpu.GPUDevice exposes no L2 size). gather_cap is
    %        MONOTONE: never below the calibrated 256e6 (~5x the AD104
    %        48 MB L2) -> dev-box path byte-identical; a B200 (126 MB L2)
    %        gets 630e6. try/catch keeps stale MEX builds working.
    %    (c) STREAMED tables are exempt: there B amortises the full
    %        host->device table re-read per pass (ceil(R/B) passes, see the
    %        note above) which dominates over the gather -- capping B=1 on
    %        a streamed table is actively harmful (2.27x measured lever).
    V_per_col = double(n_basis) * 8;                     % one complex reduced column
    if is_real_clt
        V_per_col = double(n_basis) * 4;                 % real path: re only
        if strcmp(getenv('FTLM_FP16'), '1')
            V_per_col = double(n_basis) * 2;             % fp16 gather reads __half
        end
    end
    B_cap = MAX_B;
    % L2-Groesse EINMAL pro Session abfragen (persistent): der MEX-Mode
    % ruft cudaGetDeviceProperties (10-40 ms) -- pro Block gerufen summierte
    % sich das bei vielen kleinen Bloecken zu Sekunden (icosido-Benchmark:
    % 158 Bloecke; Regressions-Nachlese 2026-07-11).
    persistent l2b_cached
    if isempty(l2b_cached)
        l2b_cached = 48e6;                               % AD104 fallback (stale MEX)
        try, l2b_cached = cuda_lanczos_clut_block_pg_Ih('l2_size'); catch, end
    end
    l2b = l2b_cached;
    gather_cap = max(256e6, 5 * double(l2b));            % monotone: never below the calibrated AD104 value
    if V_per_col > gather_cap && ~use_stream
        B_cap = 1;
    end
    %% gpuArray element cap on the V0 draw: V0_re is ONE gpuArray of
    %  n_basis*B elements, and MATLAB caps a single gpuArray variable at
    %  2^31-1 elements REGARDLESS of VRAM. B is clamped so the one-shot
    %  draw fits; blocks with n_basis beyond the cap itself (dodec s=3/2
    %  M=1..5 d>=4, icosahedron s=5 d=5) run at B = 1 through the SPLIT-V0
    %  path below (chunked draws handed to the kernel via 'set_v0').
    B_elem_cap = max(1, floor((2^31 - 1) / max(n_basis, 1)));
    if is_real_clt
        % SEGMENTED-V0 B-UNLOCK (2026-07-10): the real path draws
        % V0 per COLUMN in chunks under the 2^31 cap and scatters into the
        % kernel's interleaved buffer ('set_v0' column arg) -- the one-shot
        % gpuArray cap no longer clamps B. Configs that fit the one-shot
        % draw keep using it (bit-identical to before); the segmented draw
        % engages ONLY beyond the cap and DEFINES its RNG baseline there
        % (chunk pattern is deterministic: fixed SPLIT + ascending columns).
        % This is what unlocks B = 2..MAX_B on the dodec s=3/2 blocks
        % (n_basis 0.7-1.8e9). Gate: test_split_v0 case C.
        B_elem_cap = MAX_B;
    end
    if B_gpu == 0
        B_used = min([R_eff, MAX_B, B_vram, B_cap, B_elem_cap]);
    else
        if B_gpu > B_elem_cap
            fprintf(['    [run_ftlm] B_gpu=%d clamped to %d (n_basis*B must stay ', ...
                     'below the 2^31 gpuArray element cap).\n'], B_gpu, B_elem_cap);
        end
        B_used = min([B_gpu, R_eff, MAX_B, B_vram, B_elem_cap]);   % force bypasses B_cap only
    end
    %% Residency-vor-B (K1c; User-Politik: resident-when-fits ist der schnelle
    %  Pfad, pro Irrep gegated). Passt die VOLLE Streaming-Tabelle als
    %  residenter Prefix neben B>=1 Spalten, klemme B so, dass sie resident
    %  BLEIBT: gestreamt kostet die Tabelle ~einen vollen Host->Device-Pass
    %  pro SpMV (gemessen 1.34 vs 0.39 s/Spalte resident, 2026-07), waehrend
    %  B 4->2 die Passzahl nur x2 erhoeht. Ohne die Klemme waehlt B_vram die
    %  Lanczos-Puffer so gross, dass der Kernel-AUTO-Prefix (bzw. der via
    %  keep_table gehaltene Prefix) verdraengt wird -- prod_m0-Hergang:
    %  A-Bloecke klein -> Prefix 196/196 resident; T1g+ mit B=4 -> Alloc-Fail
    %  -> Self-Heal droppt 87.7 GB Prefix. `avail` (oben) ist pro Irrep LIVE
    %  gemessen (nach dem Irrep-Upload), nicht ein Kampagnen-Nominalwert.
    %  Nur AUTO-Prefix: explizites prefix_budget 0 (force_stream-Tests
    %  behalten Streaming-Coverage) oder >0 (Partial-Prefix-Experimente)
    %  klemmt nie; explizites B_gpu>0 bleibt Force-Override (Benchmarks).
    %  NB: -1 zaehlt als AUTO, auch wenn explizit gesetzt -- der
    %  entries_on_disk-Produktionspfad setzt clt.prefix_budget = -1
    %  ausdruecklich (ftlm_observables_pg_gpu_Ih) und MUSS klemmen.
    if use_stream && B_gpu == 0
        pref_auto = ~isfield(clt, 'prefix_budget') || double(clt.prefix_budget) == -1;
        if pref_auto
            table_full = sum(double(clt.tile_e_count)) * bpe;   % volle Tabelle in Prefix-Bytes
            budget_res = 0.90 * (avail - table_full) - 0.5e9;
            B_res      = floor(budget_res / bytes_per_col);
            if B_res >= 1 && B_res < B_used
                fprintf(['    [run_ftlm] B %d -> %d: volle Tabellen-Residency (%.1f GB Prefix) ', ...
                         'vor groesserem B (Streaming vermieden).\n'], B_used, B_res, table_full/1e9);
                B_used = B_res;
            end
        end
    end
    %% Zweierpotenz-B bei adaptiver Wahl (Kampagnen-Befund 2026-07-11, M=6):
    %  ungerades B zerstoert die Sektor-Koaleszenz der interleaved Gathers
    %  (B x 2/4 Byte pro Zeile) -- gemessen B=7: 1596 s vs B=1: 564 s auf
    %  praktisch identischen T-Bloecken (~3x pro Spalte). Explizites B_gpu
    %  bleibt unangetastet (Benchmarks duerfen jeden Wert erzwingen).
    if B_gpu == 0 && B_used > 1
        B_p2 = 2^floor(log2(double(B_used)));
        if B_p2 < B_used
            fprintf('    [run_ftlm] B %d -> %d (Zweierpotenz fuer koaleszierte Gathers).\n', ...
                    B_used, B_p2);
            B_used = B_p2;
        end
    end
    if B_used < 1 && isfield(clt, 'keep_table') && clt.keep_table
        % Keep-table/pre-flight interplay (M=1 H_g postmortem, 2026-07-04):
        % the kernel may still hold the PREVIOUS irrep's kept entry table --
        % its VRAM-resident prefix can exceed 100 GB (dodec s=3/2 M=1: 107 GB)
        % -- which this pre-flight counts as "used" although the kernel would
        % drop it on demand. That self-heal lives in init_skel_stream and is
        % unreachable once we error out HERE. So reclaim the kept state now
        % and re-run the sizing on the true free VRAM; the upcoming init then
        % re-sizes a smaller prefix AFTER the Lanczos buffers, exactly like a
        % fresh first irrep (bit-identical either way).
        ks = [0; 0; 0];
        try, ks = cuda_lanczos_clut_block_pg_Ih('keep_stats'); catch, end
        if ks(2) == 1
            fprintf(['    [run_ftlm] pre-flight short on VRAM with a KEPT entry ', ...
                     'table -> dropping kept state and re-sizing.\n']);
            cuda_lanczos_clut_block_pg_Ih('cleanup');
            wait(gpu_h);
            avail          = min(double(gpu_h.AvailableMemory), gpu_free_bytes());
            free_for_block = avail - reserved;
            vram_budget    = 0.90 * free_for_block - 0.5e9;
            B_vram         = floor(vram_budget / bytes_per_col);
            if B_gpu == 0
                B_used = min([R_eff, MAX_B, B_vram, B_cap, B_elem_cap]);
                if B_used > 1   % Zweierpotenz-Klemme (s. oben)
                    B_used = 2^floor(log2(double(B_used)));
                end
            else
                B_used = min([B_gpu, R_eff, MAX_B, B_vram, B_elem_cap]);
            end
        end
    end
    if B_used < 1
        error('run_ftlm_pg_sector_gpu_Ih:VRAM', ...
            ['Not enough free VRAM for even one Lanczos column: ~%.2f GB/col ', ...
             'needed, %.2f GB free (n_basis=%d, reserved %.2f GB, stream=%d). ', ...
             'Use a larger GPU or shrink the skeleton.'], ...
            bytes_per_col/1e9, free_for_block/1e9, n_basis, reserved/1e9, use_stream);
    end
    if mem_diag
        if use_stream, mstr = sprintf('STREAM(%d tiles)', numel(clt.tile_e_count));
        elseif is_b2_clt, mstr = 'resident-B2'; else, mstr = 'resident'; end
        fprintf(['    [run_ftlm] n_basis=%d  mode=%s  reserved=%.2f GB' ...
                 '  free-for-block=%.2f GB  -> B_used=%d (B_vram=%d)\n'], ...
                n_basis, mstr, reserved/1e9, free_for_block/1e9, B_used, B_vram);
    end

    %% Dispatch: three CLT layouts are supported.
    %    (a) skeleton path (init_skel):  M built on the GPU per SpMV from
    %         per-entry V_t / V_r / rho / sqrt_eig. Currently archived
    %         (register spilling on d >= 4 on the RTX 4000 SFF Ada).
    %    (b) streamed-M-on-GPU (init):   M was streamed into VRAM directly
    %         by BUILD_CLT_FROM_ENTRIES_IH_STREAMED. NOTHING materialises
    %         on the host. THIS IS THE PRODUCTION PATH at icosidodecahedron
    %         scale.
    %    (c) legacy host-M (init):       M is a host complex double tensor;
    %         we convert + upload here. Kept for backward compatibility
    %         on tiny systems where the host allocation is harmless.
    is_skeleton = isfield(clt, 'is_skeleton') && clt.is_skeleton;
    is_M_on_gpu = isfield(clt, 'M_on_gpu')    && clt.M_on_gpu;

    %% VRAM pre-check (skeleton path). init_skel_ref cudaMalloc's ~7-10
    %  Lanczos vector buffers of n_basis*B*4 B as RAW cudaMalloc (outside
    %  MATLAB's gpuArray pool). On a near-full GPU that alloc can fail; the
    %  kernel now checks it (clean error), but we also refuse here so the
    %  caller gets an actionable message before touching the GPU. The 2x
    %  margin guards against fragmentation of the free block (the skeleton
    %  already holds a large contiguous gpuArray pool, so the raw cudaMalloc
    %  may not find a contiguous chunk even when the total looks sufficient).
    if is_skeleton
        vec_bytes  = double(n_basis) * double(B_used) * 4;    % one fp32 block
        if is_real_clt
            need_bytes = 5 * vec_bytes + 0.5e9;               % 3 re bufs + V0 + slack (gate ~ count 4x)
            if strcmp(getenv('FTLM_FP16'), '1')
                % R1: Kernel-Puffer halbiert (3 x 2 B) + fp32-V0 (4 B) +
                % ~2.5 B/Element Marge = 12.5 B/Element (fp32-Gate: 20 B).
                need_bytes = double(n_basis) * double(B_used) * 12.5 + 0.5e9;
            end
        else
            need_bytes = 9 * vec_bytes + 0.5e9;               % bufs + V0 + margin (gate ~ real 8x)
        end
        % The buffers must fit in free_for_block = AvailableMemory - reserved
        % (reserved = resident B2 table, or one streaming tile, or 0).
        free_bytes = free_for_block;
        if need_bytes > free_bytes && isfield(clt, 'keep_table') && clt.keep_table
            % SECOND keep-table/pre-flight gate (M=3 H_g postmortem,
            % 2026-07-05): the B<1 gate above reclaims the kept table, but a
            % block can pass it (Lanczos columns alone fit) and still trip
            % THIS total-need check because the previous irrep's kept prefix
            % (>100 GB at dodec s=3/2 big-M scale) is counted as used. Same
            % remedy: drop the kept state, re-measure, re-check once.
            ks = [0; 0; 0];
            try, ks = cuda_lanczos_clut_block_pg_Ih('keep_stats'); catch, end
            if ks(2) == 1
                fprintf(['    [run_ftlm] VRAM refusal with a KEPT entry table ', ...
                         '-> dropping kept state and re-checking.\n']);
                cuda_lanczos_clut_block_pg_Ih('cleanup');
                wait(gpu_h);
                avail          = min(double(gpu_h.AvailableMemory), gpu_free_bytes());
                free_for_block = avail - reserved;
                free_bytes     = free_for_block;
            end
        end
        if need_bytes > free_bytes
            error('run_ftlm_pg_sector_gpu_Ih:VRAM', ...
                ['Block-Lanczos needs ~%.1f GB free VRAM (n_basis=%d, B=%d) ', ...
                 'but only %.1f GB is free. Lower B (B_gpu) or shrink the ', ...
                 'skeleton (c-index). Refusing to avoid a CUDA OOM crash.'], ...
                need_bytes/1e9, n_basis, B_used, free_bytes/1e9);
        elseif 1.5 * need_bytes > free_bytes
            % 1.5x (was 2x): the 2x margin fired routinely on HEALTHY
            % production runs (kagome N=36 spin-flip d=6: 7.3 GB needed vs
            % 12.2 GB free ran fine every block) -- warning noise that
            % drowned real signals. 1.5x still flags genuinely tight cases
            % before the hard refusal above.
            warning('run_ftlm_pg_sector_gpu_Ih:VRAMtight', ...
                ['Block-Lanczos VRAM margin is tight: ~%.1f GB needed, %.1f GB ', ...
                 'free. Fragmentation may still cause a cudaMalloc failure; ', ...
                 'consider a smaller B or the c-index skeleton.'], ...
                need_bytes/1e9, free_bytes/1e9);
        end
    end

    if mem_diag, mem_snapshot('enter run_ftlm_pg_sector', gpu_h); end

    %% Legacy path uploads (only needed for is_M_on_gpu / non-skeleton).
    %  For the skeleton path (Stufe 5a), clt.* large fields are already
    %  gpuArrays and are passed directly to init_skel_ref below; we
    %  don't allocate src_idx_gpu (= clt.src_idx - 1) here, because
    %  the subtraction would create a duplicate gpuArray.
    if ~is_skeleton
        % Cap-closure 2026-07-03: the legacy precomputed-M path uploads the
        % full per-entry src column (n_entries) and later the M tensor
        % (d^2 * n_entries) as SINGLE gpuArrays. Fine for the small systems
        % this archived path serves; impossible beyond MATLAB's 2^31 cap --
        % fail early with the remedy.
        assert(n_entries < 2^31 && d_irrep^2 * n_entries < 2^31, ...
            ['run_ftlm_pg_sector_gpu_Ih: the legacy precomputed-M path cannot ', ...
             'hold n_entries = %.3g (M tensor d^2*n_entries = %.3g) within the ', ...
             '2^31 gpuArray element cap. Use the production skeleton path ', ...
             '(build_entry_skeleton_Ih + build_clt_skeleton_from_entries_Ih).'], ...
            n_entries, d_irrep^2 * n_entries);
        diag_gpu       = gpuArray(single(clt.diag_vals));
        % int64 offsets (2026-07 64-bit basis-offset ABI). NB: entry_offsets
        % was silently uploaded as int32 here while the kernel has read it as
        % long long since the June B2 upgrade -- a latent legacy-path bug
        % (production uses the skeleton path); both casts are now correct.
        rep_offs_gpu   = gpuArray(int64(clt.rep_offsets));
        n_per_rep_gpu  = gpuArray(int32(clt.n_per_rep));
        entries_n_gpu  = gpuArray(int32(clt.entries_per_rep));
        entry_offs_gpu = gpuArray(int64(clt.entry_offsets));
        src_idx_gpu    = gpuArray(int32(clt.src_idx - 1));   % 0-based for C
    end

    if is_skeleton
        % Skeleton: kernel computes M on-the-fly per SpMV.
        %
        % Stufe 5a (May 2026): clt.* large fields arrive as gpuArrays
        % from BUILD_CLT_SKELETON_FROM_ENTRIES_IH; the host never
        % carried a shadow. No gpuArray() upload necessary here; we
        % just rename for clarity. clt.c_a_const carries the constant
        % when c_a is empty (s = 1/2 case, Stufe 1b).
        %
        % Stufe 2: 'init_skel_ref' takes raw device pointers into our
        % gpuArrays. clt and its fields MUST stay alive until cleanup
        % is called at the end of this function; do NOT `clear` them
        % between init and cleanup, that crashes.
        if mem_diag, mem_snapshot('before init_skel_ref', gpu_h); end

        % Indexed-c arrays (s>1/2). Empty -> kernel uses the per-entry/const
        % c_a path. isfield guard keeps old skeletons (no c_idx) working.
        if isfield(clt, 'c_idx'),   c_idx_arg = clt.c_idx;
        else,                       c_idx_arg = gpuArray(zeros(0, 1, 'uint8')); end
        if isfield(clt, 'c_table'), c_table_arg = clt.c_table;
        else,                       c_table_arg = gpuArray(zeros(0, 1, 'single')); end
        % G2: packed src|g uint32 (empty -> kernel uses src_idx/g_idx).
        if isfield(clt, 'srcg'),    srcg_arg = clt.srcg;
        else,                       srcg_arg = gpuArray(zeros(0, 1, 'uint32')); end
        % #1: trivial-trivial fast-path arrays (empty -> kernel uses the full
        % V' rho V contraction for every entry).
        if isfield(clt, 'triv'),    triv_arg = clt.triv;
        else,                       triv_arg = gpuArray(zeros(0, 1, 'uint8'));  end
        if isfield(clt, 'Qbar_re'), qre_arg = clt.Qbar_re; qim_arg = clt.Qbar_im;
        else, qre_arg = gpuArray(zeros(0,1,'single')); qim_arg = gpuArray(zeros(0,1,'single')); end
        % D2: per-rep V/sqrt slot map (compact-V). Empty -> kernel full mode.
        if isfield(clt, 'v_slot'), vslot_arg = clt.v_slot;
        else,                      vslot_arg = gpuArray(zeros(0, 1, 'int32')); end
        % Dispatch the three on-the-fly init modes (same leading 26 args):
        %   init_skel_stream : STREAM the host entry table in rep-tiles per SpMV
        %                      (large-d N=36; +3 trailing tile-partition args).
        %   init_skel_b2     : per-entry HOST arrays uploaded + kept RESIDENT
        %                      (n_entries > 2^31 gpuArray cap, fits VRAM).
        %   init_skel_ref    : per-entry gpuArrays borrowed (normal path).
        if use_stream
            % OUT-OF-CORE: when the entry table was spilled to a memory-mapped
            % file (spill_entries_mmap), pass the raw uint64 mapped pointers in
            % place of the resident host arrays; the kernel's host_ptr_arg reads
            % tile slices straight from the mapped pages (NVMe-paged). Otherwise
            % pass the resident host arrays (unchanged).
            if isfield(clt, 'mmap_srcg') && ~isempty(clt.mmap_srcg)
                src_pa = uint64(0); g_pa = uint64(0); srcg_pa = clt.mmap_srcg;   % packed mmap
            elseif isfield(clt, 'mmap_src') && ~isempty(clt.mmap_src)
                src_pa = clt.mmap_src; g_pa = clt.mmap_g; srcg_pa = srcg_arg;     % unpacked mmap (srcg_arg empty)
            else
                src_pa = clt.src_idx; g_pa = clt.g_idx; srcg_pa = srcg_arg;       % resident host arrays
            end
            % s>=1 out-of-core: the uint8 c-index block lives in the mapped
            % file too -> pass its raw uint64 pointer as the c_idx arg (the
            % kernel's host_ptr_arg treats a uint64 SCALAR as a mapped ptr).
            if isfield(clt, 'mmap_cidx') && ~isempty(clt.mmap_cidx)
                c_idx_arg = clt.mmap_cidx;
            end
            % Guard: an indexed-c block (non-empty c_table) must come with a
            % c-index source (host/gpu c_idx array OR the mmap pointer);
            % otherwise the kernel would silently fall back to constant c.
            if ~isempty(c_table_arg) && isempty(c_idx_arg)
                error('run_ftlm_pg_sector_gpu_Ih:cidx', ...
                    'indexed-c clt (non-empty c_table) without c_idx/mmap_cidx.');
            end
            % Resident-prefix budget (bytes), trailing init arg:
            %   -1 = AUTO  (kernel sizes the prefix of leading tiles to the VRAM
            %               left free after its Lanczos buffers; tail is streamed)
            %    0 = OFF   (full streaming)
            %   >0 = explicit byte cap (tests exercise partial prefixes)
            % K2e: the default is AUTO even under force_stream. The old
            % force_stream->OFF default was a latent trap: any NEW
            % force_stream caller silently re-streamed the FULL entry table
            % on EVERY SpMV (~86 GB at dodec s=3/2). Tests/benchmarks that
            % must exercise the streaming machinery set
            % clt.prefix_budget = 0 EXPLICITLY.
            if isfield(clt, 'prefix_budget'), pref_budget = double(clt.prefix_budget);
            else,                             pref_budget = -1;
            end
            cuda_lanczos_clut_block_pg_Ih('init_skel_stream', ...
                clt.diag_vals, clt.rep_offsets, clt.n_per_rep, ...
                clt.entries_per_rep, clt.entry_offsets, ...
                src_pa, g_pa, clt.c_a, ...
                clt.V_re, clt.V_im, clt.rho_re, clt.rho_im, clt.sqrt_eig, ...
                n_basis, n_reps, n_entries, d_irrep, B_used, clt.c_a_const, ...
                c_idx_arg, c_table_arg, srcg_pa, triv_arg, qre_arg, qim_arg, vslot_arg, ...
                int32(clt.tile_rep_ptr), int64(clt.tile_e_start), int64(clt.tile_e_count), ...
                pref_budget);
        else
            if is_b2_clt
                init_mode = 'init_skel_b2';
            else
                init_mode = 'init_skel_ref';
            end
            if isfield(clt, 'run_ptr') && ~isempty(clt.run_ptr) ...
                    && strcmp(init_mode, 'init_skel_ref')
                % R2-v1: getilte Entry-Struktur durchreichen (Entry-Arrays im
                % clt sind bereits PERMUTIERT; der Kernel kopiert run_ptr/
                % run_tgt in eigene Device-Puffer).
                cuda_lanczos_clut_block_pg_Ih(init_mode, ...
                    clt.diag_vals, clt.rep_offsets, clt.n_per_rep, ...
                    clt.entries_per_rep, clt.entry_offsets, ...
                    clt.src_idx, clt.g_idx, clt.c_a, ...
                    clt.V_re, clt.V_im, clt.rho_re, clt.rho_im, clt.sqrt_eig, ...
                    n_basis, n_reps, n_entries, d_irrep, B_used, clt.c_a_const, ...
                    c_idx_arg, c_table_arg, srcg_arg, triv_arg, qre_arg, qim_arg, vslot_arg, ...
                    clt.run_ptr, clt.run_tgt, clt.tile_run_ptr);
            else
                cuda_lanczos_clut_block_pg_Ih(init_mode, ...
                    clt.diag_vals, clt.rep_offsets, clt.n_per_rep, ...
                    clt.entries_per_rep, clt.entry_offsets, ...
                    clt.src_idx, clt.g_idx, clt.c_a, ...
                    clt.V_re, clt.V_im, clt.rho_re, clt.rho_im, clt.sqrt_eig, ...
                    n_basis, n_reps, n_entries, d_irrep, B_used, clt.c_a_const, ...
                    c_idx_arg, c_table_arg, srcg_arg, triv_arg, qre_arg, qim_arg, vslot_arg);
            end
        end

        if mem_diag, mem_snapshot('after init_skel_ref', gpu_h); end
    elseif is_M_on_gpu
        % Streamed M layout: M_re_flat_gpu / M_im_flat_gpu are already
        % gpuArrays. Use 'init_ref' so the MEX skips its internal
        % cudaMalloc + cudaMemcpyDeviceToDevice for M, which would
        % otherwise momentarily DOUBLE the M-tensor VRAM footprint
        % (e.g. 32 GB instead of 16 GB on s = 1/2 icosidodecahedron
        % H_g, exceeding the 20 GB on the RTX 4000 SFF Ada).
        %
        % CRITICAL: with 'init_ref' the kernel holds raw device
        % pointers into clt.M_re_flat_gpu / clt.M_im_flat_gpu. The
        % caller MUST keep CLT alive until 'cleanup' has been called
        % at the end of this function; otherwise MATLAB's GC would
        % free the underlying VRAM while the kernel still uses it.
        cuda_lanczos_clut_block_pg_Ih('init_ref', ...
            diag_gpu, rep_offs_gpu, n_per_rep_gpu, ...
            entries_n_gpu, entry_offs_gpu, src_idx_gpu, ...
            clt.M_re_flat_gpu, clt.M_im_flat_gpu, ...
            n_basis, n_reps, n_entries, d_irrep, B_used);
    else
        % Legacy: M precomputed on host as 3D complex double tensor.
        % Kept for backward compatibility on small systems; do NOT use
        % at icosidodecahedron scale or above (32 GB host allocation).
        M_re_flat = single(real(clt.M));
        M_im_flat = single(imag(clt.M));
        M_re_gpu  = gpuArray(M_re_flat(:));
        M_im_gpu  = gpuArray(M_im_flat(:));

        cuda_lanczos_clut_block_pg_Ih('init', ...
            diag_gpu, rep_offs_gpu, n_per_rep_gpu, ...
            entries_n_gpu, entry_offs_gpu, src_idx_gpu, ...
            M_re_gpu, M_im_gpu, ...
            n_basis, n_reps, n_entries, d_irrep, B_used);
        % Same reverted clear as above; see init_skel branch comment.
    end
    wait(gpu_h);

    %% Main loop over random vectors in blocks of B_used
    E_sec = zeros(M_lz_actual * R_eff, 1);
    w_sec = zeros(M_lz_actual * R_eff, 1);
    idx   = 0;

    % Use the GPU RNG so the random start vectors are created DIRECTLY
    % in VRAM. The previous code:
    %     randn(n_basis, B) on host (double)  -> 416 MB for H_g
    %     single() copy on host               -> 208 MB extra
    %     gpuArray() copy to device           -> 208 MB on GPU
    % allocated ~ 1.25 GB of host transient per chain. MATLAB's
    % gpuArray pool then cached pieces of this between sectors and
    % was responsible for the bulk of the per-sector host residue
    % (~ 0.6 GB / sector) we measured with mem_diag.
    %
    % gpuArray.randn writes directly to a single-precision VRAM
    % buffer; nothing ever touches host memory. Reproducibility is
    % preserved via gpurng(seed) instead of rng(seed).
    % SEED CONVENTION (real path): gpurng(seed) as before, but the real path
    % draws ONE gpuArray.randn per Krylov block (V0_re only) instead of two
    % (V0_re + V0_im) -- statistically equivalent but DIFFERENT samples than a
    % complex run with the same seed (a new baseline, like the spin-flip).
    % Bit-reproducibility of a real run requires the same seed AND the real
    % path; force_complex reproduces the old complex draws exactly.
    gpurng(seed);
    % lz_diag accumulators (print-only, opt-in): the standard Lanczos residual
    % bound for Ritz pair i of one column is |beta_last| * |Q(end,i)|. Nothing
    % here feeds back into E_sec/w_sec -- results are byte-identical with the
    % flag on or off; it exists so M_lz / R can be tuned from data instead of
    % guesses (kernel time is linear in M_lz and dominates production runs).
    dg_nconv = 0; dg_ntot = 0; dg_bfloor = inf; dg_e0res = 0; dg_wunconv = 0;
    for r_start = 1 : B_used : R_eff
        r_end = min(r_start + B_used - 1, R_eff);
        B_cur = r_end - r_start + 1;

        if n_basis * B_cur <= 2^31 - 1
            V0_re_gpu = gpuArray.randn(n_basis, B_cur, 'single');
            if is_real_clt
                V0_im_gpu = gpuArray(single([]));    % real kernel path: no imag draw
            else
                V0_im_gpu = gpuArray.randn(n_basis, B_cur, 'single');
            end
        elseif is_real_clt
            % SEGMENTED-V0 (B-unlock): n_basis * B_cur exceeds the 2^31-1
            % elements-per-gpuArray cap, but each COLUMN is drawn separately
            % in chunks and scattered into the kernel's interleaved 64-bit
            % buffer ('set_v0' with column arg); block_lanczos then runs
            % with an EMPTY V0. This regime could never run before (it was
            % clamped to B = 1), so it DEFINES its RNG baseline: fixed SPLIT
            % chunking, columns in ascending order, sequential gpurng draws
            % -- deterministic per (seed, n_basis, B). Gates: test_split_v0
            % case C (scatter bit-identity) + Sum Rule.
            SPLIT = 2^31 - 2^20;
            for bcol = 0 : B_cur - 1
                off = 0;
                while off < n_basis
                    nc = min(SPLIT, n_basis - off);
                    ch = gpuArray.randn(nc, 1, 'single');
                    cuda_lanczos_clut_block_pg_Ih('set_v0', ch, off, 0, bcol);
                    clear ch;
                    off = off + nc;
                end
            end
            V0_re_gpu = gpuArray(single([]));    % empty -> kernel uses preloaded V0
            V0_im_gpu = gpuArray(single([]));
        else
            % SPLIT-V0 (complex path): n_basis exceeds the cap; B stays 1
            % (B_elem_cap) and re/im keep the historical chunk-draw order.
            % Gate: test_split_v0 (preloaded == passed-in V0, bit-identical).
            assert(B_cur == 1, 'run_ftlm: complex split-V0 requires B = 1');
            SPLIT = 2^31 - 2^20;
            off = 0;
            while off < n_basis
                nc = min(SPLIT, n_basis - off);
                ch = gpuArray.randn(nc, 1, 'single');
                cuda_lanczos_clut_block_pg_Ih('set_v0', ch, off, 0);
                ch = gpuArray.randn(nc, 1, 'single');
                cuda_lanczos_clut_block_pg_Ih('set_v0', ch, off, 1);
                clear ch;
                off = off + nc;
            end
            V0_re_gpu = gpuArray(single([]));    % empty -> kernel uses preloaded V0
            V0_im_gpu = gpuArray(single([]));
        end

        % 5th arg: actual column count (only read on the preloaded/segmented
        % path -- the last R-chunk can be narrower than B_batch).
        [AL, BE] = cuda_lanczos_clut_block_pg_Ih('block_lanczos', ...
                       V0_re_gpu, V0_im_gpu, M_lz_actual, B_cur);

        for b = 1 : B_cur
            [E_r, q1_r, qe_r] = solve_tridiag(AL(:, b), BE(:, b));
            n_l = numel(E_r);
            E_sec(idx+1 : idx+n_l) = E_r;
            w_sec(idx+1 : idx+n_l) = (D_eff / R_eff) * q1_r;
            idx = idx + n_l;
            if lz_diag
                spread = max(E_r) - min(E_r);  if spread <= 0, spread = 1; end
                res  = BE(n_l, b) * sqrt(qe_r);        % residual bound per Ritz pair
                conv = res < 1e-4 * spread;
                dg_nconv   = dg_nconv + sum(conv);
                dg_ntot    = dg_ntot + n_l;
                dg_bfloor  = min(dg_bfloor, min(BE(1:n_l, b)));
                [~, i0]    = min(E_r);
                dg_e0res   = max(dg_e0res, res(i0) / spread);
                dg_wunconv = max(dg_wunconv, sum(q1_r(~conv)) / max(sum(q1_r), eps));
            end
        end
    end
    if lz_diag
        fprintf(['    [lz-diag] M_lz=%d R=%d: Ritz conv %d/%d (res<1e-4*spread), ', ...
                 'min beta %.2e, E0 res %.1e*spread, max unconv weight %.2e\n'], ...
                M_lz_actual, R_eff, dg_nconv, dg_ntot, dg_bfloor, dg_e0res, dg_wunconv);
    end

    if mem_diag, mem_snapshot('after Lanczos loop', gpu_h); end

    % In-Lanczos VRAM high-water mark (one driver query, no device sync):
    % the kernel's raw buffers are freed by 'cleanup' below, so this is the
    % only spot that sees the true per-sector peak. 4th output -> perf.
    vram_peak_gb = NaN;
    try
        vram_peak_gb = (double(gpu_h.TotalMemory) - double(gpu_h.AvailableMemory)) / 1e9;
    catch
    end

    % Keep-table (wave-2 [13]): when the driver marks the clt (out-of-core
    % streaming, several irreps sharing ONE per-M entry table), preserve the
    % kernel's irrep-independent table state (tile partition + device tiles +
    % resident prefix) across this cleanup. The next init_skel_stream reuses
    % it on a fingerprint match instead of re-reading the whole table from
    % disk (~86 GB per irrep at dodec s=3/2). Mismatch or non-streaming init
    % -> the kernel falls back to a full cleanup + fresh init on its own.
    if isfield(clt, 'keep_table') && clt.keep_table
        cuda_lanczos_clut_block_pg_Ih('cleanup_keep_table');
    else
        cuda_lanczos_clut_block_pg_Ih('cleanup');
    end

    if mem_diag, mem_snapshot('after MEX cleanup', gpu_h); end

    % Explicit release of ALL sector-local gpuArrays AND host workspaces.
    %
    % Safety: 'cleanup' above has just released the kernel's persistent
    % pointers; the gpuArrays we cleared had been used as pointer
    % sources for init_skel_ref/init_ref but the kernel no longer
    % references them, so clearing here is safe (unlike clearing
    % between init and cleanup, which crashed in May 2026).
    %
    % Purpose: stop MATLAB's gpuArray host-side cache from accumulating
    % ~ 0.5-1 GB per sector. Without this, a 10-irrep sweep on the
    % icosidodecahedron leaks 5-10 GB of host RAM that scales badly to
    % larger systems. MATLAB's `clear` of nonexistent variables is a
    % silent no-op, so we list ALL possible locals across the three
    % code paths (init_skel_ref, init_ref, legacy init).
    clear diag_gpu rep_offs_gpu n_per_rep_gpu entries_n_gpu entry_offs_gpu ...
          src_idx_gpu src_idx_gpu_skel g_idx_gpu c_a_gpu ...
          V_re_gpu V_im_gpu rho_re_gpu rho_im_gpu sqrt_eig_gpu ...
          M_re_gpu M_im_gpu M_re_flat M_im_flat ...
          V0_re_gpu V0_im_gpu AL BE;

    if mem_diag, mem_snapshot('after clear of run_ftlm locals', gpu_h); end

    E_sec = E_sec(1:idx);
    w_sec = w_sec(1:idx);
end


% ----------------------------------------------------------------
function [ep, q1, qe] = solve_tridiag(alpha, beta)
    alpha = double(alpha(:));
    beta  = double(beta(:));
    n     = length(alpha);
    T     = diag(alpha(1:n));
    if n > 1
        T = T + diag(beta(1:n-1), 1) + diag(beta(1:n-1), -1);
    end
    [Q, D] = eig(T, 'vector');
    ep = D;
    q1 = abs(Q(1, :)').^2;
    qe = abs(Q(end, :)').^2;   % last-row weights: Lanczos residual bounds (lz_diag)
end
