function skel = build_clt_skeleton_from_entries_Ih(entries, reps, V_per_rep, ...
                                                    eig_per_rep, n_per_rep, ...
                                                    irrep_data, d_irrep, group, ...
                                                    triv_active, eskel, compact_v, ...
                                                    force_complex)
%   *** PRODUCTION ***  per-irrep skeleton half (consumes the shared eskel from
%   build_entry_skeleton_Ih). THIS is the live GPU-FTLM skeleton path. The
%   build_clt_pg_Ih / build_clt_from_entries_Ih{,_gpu,_streamed} files are
%   NON-PRODUCTION (regression/bench references) -- do not edit those for fixes.
%   Optional FORCE_COMPLEX (default false): keep the complex GPU layout (zero
%   V_im/rho_im/Qbar_im arrays) even when IRREP_DATA is real -- the pre-real-
%   kernel baseline, for A/B measurements and reproducing old runs.
%
%   REAL FP32 PATH (June 2026): when IRREP_DATA is REAL (realified space-group
%   or I_h irreps) and FORCE_COMPLEX is not set, skel.is_real = true and the
%   V_im / rho_im / Qbar_im fields are EMPTY gpuArrays. The CUDA kernel detects
%   the empty V_im and runs its real fork: half the Krylov buffers, half the
%   random V[src] gather traffic, ONE V0 RNG draw. (Previously zero-filled
%   imaginary arrays were uploaded -- pure VRAM/traffic waste.)
%
%   Optional COMPACT_V (default false): D2 compact-V storage. The per-rep V/
%   sqrt_eig tensor is [d x d x n_reps_full] = d^2 * n_reps_full floats, which
%   for a high-d irrep on a large sector is huge (e.g. d=8, n_reps~3.15e7 ->
%   16 GB, exceeding VRAM). Since the ~all trivial-stabiliser reps share ONE
%   (V, eig), COMPACT_V stores V/sqrt ONLY once for that shared block plus once
%   per non-trivial active rep, and emits skel.v_slot (per-rep int32, 0-based)
%   mapping each rep to its slot. The kernel reads V[v_slot[rep]]. Requires
%   TRIV_ACTIVE. Empty/false -> full per-rep V (skel.v_slot empty), I_h path
%   unchanged. Bit-identical either way.
%   Optional TRIV_ACTIVE (logical [n_active x 1] from APPLY_IRREP_TO_ORBITS)
%   marks the trivial-stabiliser reps, which all share one (V, eig, n); when
%   given, the host V-padding is bulk-filled for them in one vectorised
%   scatter (D2) instead of looping over all n_active reps. Omitting it falls
%   back to the per-rep loop (byte-identical, just slower).
%
%   Optional ESKEL (from BUILD_ENTRY_SKELETON_IH) is the irrep-INDEPENDENT
%   entry skeleton (src/g/c, per-rep entry counts/offsets, diag) built ONCE per
%   M sector (A1-ph2); when given, its gpuArray handles are reused instead of
%   rebuilt+re-uploaded per irrep. Omitting it builds one inline (single-irrep
%   / legacy callers). The caller must keep ESKEL alive while the skel is used.
%BUILD_CLT_SKELETON_FROM_ENTRIES_IH  Skeleton CLT with gpuArray output.
%
%   STATUS: PRODUCTION (Stufe 5a, May 2026). Returns all large fields
%   as gpuArrays so the sector never carries a host-side shadow of the
%   skeleton data. Saves ~ 1-2 GB of host RAM per sector at icosido-
%   decahedron scale and is the architectural prerequisite for N >= 32.
%
%   The host-side build pattern is unavoidable for now (V_per_rep
%   arrives as a cell of small per-rep matrices that we need to pad
%   into a [d x d x n_active] tensor; doing this slice-by-slice on GPU
%   would be launch-overhead-bound). What we DO ensure is that every
%   large host array is uploaded + cleared in a single step, so the
%   per-array host peak is bounded by its size and not by the SUM of
%   all skel arrays.
%
%   SKEL = BUILD_CLT_SKELETON_FROM_ENTRIES_IH(ENTRIES, REPS, V_PER_REP,
%       EIG_PER_REP, N_PER_REP, IRREP_DATA, D_IRREP, GROUP)
%
%   Output struct SKEL contains, in addition to the small host scalars
%   (is_skeleton, n_basis, n_reps, d_irrep, c_a_const), the gpuArray
%   fields:
%       diag_vals          [n_active x 1] single
%       rep_offsets        [n_active x 1] int64 (basis offsets can exceed 2^31)
%       n_per_rep          [n_active x 1] int32
%       entries_per_rep    [n_active x 1] int32
%       entry_offsets      [n_active x 1] int64 (entry offsets can exceed 2^31)
%       src_idx            [n_coll x 1] int32  (1-based; kernel subtracts 1)
%       g_idx              [n_coll x 1] uint16 (1-based; kernel subtracts 1)
%       c_a                [n_coll x 1] single, OR empty if c_a is constant
%                          (in which case c_a_const carries the scalar)
%       V_re, V_im         flat [d*d*n_active] single
%       rho_re, rho_im     flat [d*d*120] single
%       sqrt_eig           flat [d*n_active] single
%
%   The const-c_a detection (Stufe 1b) is also done here so that
%   RUN_FTLM_PG_SECTOR_GPU_IH doesn't need to gather() the GPU c_a
%   array just to test for constancy.
%
%   The tgt_idx field (Stufe 1a) is NOT produced: the per-(rep, k')
%   OTF kernel derives the target rep from its thread index and never
%   reads tgt_idx.
%
%   See also COLLECT_CLT_ENTRIES_IH, RUN_FTLM_PG_SECTOR_GPU_IH,
%            CUDA_LANCZOS_CLUT_BLOCK_PG_IH.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    d        = d_irrep;
    n_active = numel(reps);

    %% Real FP32 path: REAL irrep data (realified space-group / I_h irreps)
    %  -> drop the imaginary halves (V_im/rho_im/Qbar_im EMPTY, skel.is_real).
    %  V from APPLY_IRREP_TO_ORBITS inherits realness from the irrep data;
    %  the first active V slice is checked as a cheap consistency guard.
    %  FORCE_COMPLEX keeps the old zero-imag complex layout (A/B baseline).
    if nargin < 12 || isempty(force_complex), force_complex = false; end
    is_real = isreal(irrep_data) && ~force_complex && ...
              (n_active == 0 || isreal(V_per_rep{1}));

    if n_active == 0
        skel = empty_skel(d_irrep);
        skel.is_real = is_real;
        return;
    end

    %% A1-phase-2: the irrep-INDEPENDENT entry skeleton (src/g/c with const-c
    %  detection, per-target-rep entry counts/offsets, diag) is built ONCE per
    %  M sector by BUILD_ENTRY_SKELETON_IH and passed in via ESKEL; we reuse its
    %  gpuArray handles below. If the caller did not pass one (single-irrep /
    %  legacy callers, tests) we build it inline -- correct, just not shared.
    %  (A1 full-rep indexing lives inside BUILD_ENTRY_SKELETON_IH now.)
    if nargin < 10 || isempty(eskel)
        eskel = build_entry_skeleton_Ih(entries);
    end
    n_reps_full = eskel.n_reps_full;

    %% PER-IRREP scatter: place n_per_rep / V / sqrt_eig at the FULL-rep slot
    %  of each active rep. active_pos(j) = position of super_reps(j) in `reps`
    %  (0 if inactive); full_pos(i) = super-rep slot of active rep i. Because
    %  apply_irrep returns `reps` in ascending super-rep order, the active reps
    %  occupy their super-rep slots in the same order as before -> the basis
    %  packing (and thus the FTLM weights) is byte-identical.
    [in_active, active_pos] = ismember(int64(entries.super_reps), int64(reps));
    n_per_rep_full = zeros(n_reps_full, 1, 'int32');
    n_per_rep_full(in_active) = int32(n_per_rep(active_pos(in_active)));
    % int64 (2026-07 64-bit basis-offset ABI): offsets and n_basis = d*n_reps
    % exceed 2^31 for icosahedron-s=5 / square-6x7 class blocks.
    rep_offsets_h  = int64([0; cumsum(double(n_per_rep_full(1:end-1)))]);
    n_basis        = int64(sum(double(n_per_rep_full)));
    full_pos = zeros(n_active, 1);
    full_pos(active_pos(in_active)) = find(in_active);

    have_triv  = (nargin >= 9) && ~isempty(triv_active) && any(triv_active);
    do_compact = (nargin >= 11) && ~isempty(compact_v) && compact_v && have_triv;

    if do_compact
        %% D2 COMPACT-V: store V/sqrt once for the shared trivial block (slot 0)
        %  plus once per non-trivial ACTIVE rep; v_slot_full maps each full rep
        %  to its 0-based slot. Shrinks V from d^2*n_reps_full to d^2*(1+n_nt).
        nt_active = find(~triv_active(:)).';
        nt_active = nt_active(arrayfun(@(i) double(n_per_rep(i)) > 0, nt_active));
        n_slots   = 1 + numel(nt_active);
        V_re_all_h     = zeros(d, d, n_slots, 'single');
        if is_real, V_im_all_h = zeros(0, 1, 'single');    % real path: no imag half
        else,       V_im_all_h = zeros(d, d, n_slots, 'single'); end
        sqrt_eig_all_h = ones(d, n_slots, 'single');
        v_slot_full    = zeros(n_reps_full, 1, 'int32');   % 0-based; default 0 = trivial slot

        i0  = find(triv_active, 1);
        n_t = double(n_per_rep(i0));
        if n_t > 0
            V_re_all_h(:, 1:n_t, 1)  = single(real(V_per_rep{i0}));
            if ~is_real
                V_im_all_h(:, 1:n_t, 1) = single(imag(V_per_rep{i0}));
            end
            sqrt_eig_all_h(1:n_t, 1) = single(sqrt(eig_per_rep{i0}(:)));
        end
        v_slot_full(full_pos(triv_active)) = 0;            % all trivial reps -> slot 0
        s0 = 1;                                            % next 0-based slot
        for i = nt_active
            n_i = double(n_per_rep(i));
            V_re_all_h(:, 1:n_i, s0+1)  = single(real(V_per_rep{i}));
            if ~is_real
                V_im_all_h(:, 1:n_i, s0+1) = single(imag(V_per_rep{i}));
            end
            sqrt_eig_all_h(1:n_i, s0+1) = single(sqrt(eig_per_rep{i}));
            v_slot_full(full_pos(i)) = s0;
            s0 = s0 + 1;
        end
    else
        % FULL-V mode: d^2 * n_reps_full elements in ONE gpuArray upload.
        % Reached when compact-V is off OR silently as the have_triv=false
        % DEGRADATION of compact-V -- exactly the path of the original
        % 2^31 incident. Fail EARLY and loudly (cap-closure inventory
        % 2026-07-03) instead of dying in the upload at line ~271.
        assert(double(d)^2 * double(n_reps_full) < 2^31, ...
            ['build_clt_skeleton_from_entries_Ih: FULL per-rep V tensor ', ...
             '(d^2 x n_reps = %g elements) exceeds the 2^31 gpuArray cap. ', ...
             'This sector has no trivial-stabiliser reps (or compact_v is ', ...
             'off), so the compact-V slot table is unavailable -- such a ', ...
             'block needs a per-rep V compaction extension.'], ...
            double(d)^2 * double(n_reps_full));
        v_slot_full    = zeros(0, 1, 'int32');             % empty -> kernel full mode
        V_re_all_h     = zeros(d, d, n_reps_full, 'single');
        if is_real, V_im_all_h = zeros(0, 1, 'single');    % real path: no imag half
        else,       V_im_all_h = zeros(d, d, n_reps_full, 'single'); end
        sqrt_eig_all_h = ones(d, n_reps_full, 'single');

        %% D2: the trivial-stabiliser reps (the ~99.8% majority) all share ONE
        %  (V, eig, n) from APPLY_IRREP_TO_ORBITS's repmat fast path, so when the
        %  caller passes the triv_active mask we fill them in a single vectorised
        %  scatter instead of n_active loop iterations. Only the (few) non-trivial
        %  reps still loop. Without the mask we loop over all reps.
        if have_triv
            i0  = find(triv_active, 1);
            n_t = double(n_per_rep(i0));
            jt  = full_pos(triv_active);
            if n_t > 0
                Vre_t = single(real(V_per_rep{i0}));        % shared [d x n_t]
                se_t  = single(sqrt(eig_per_rep{i0}(:)));   % shared [n_t x 1]
                if ~is_real, Vim_t = single(imag(V_per_rep{i0})); end
                % Chunked scatter: one repmat over ALL trivial reps would
                % materialise a second slab-sized temporary (~7.4 GB at dodec
                % s=3/2 d=5) BEFORE the assignment; 4e6-rep slices bound the
                % transient to ~0.4 GB. Stored values identical.
                FILL_CHUNK = 4e6;
                for c0 = 1 : FILL_CHUNK : numel(jt)
                    jc = jt(c0 : min(c0 + FILL_CHUNK - 1, numel(jt)));
                    V_re_all_h(:, 1:n_t, jc)  = repmat(Vre_t, [1, 1, numel(jc)]);
                    if ~is_real
                        V_im_all_h(:, 1:n_t, jc) = repmat(Vim_t, [1, 1, numel(jc)]);
                    end
                    sqrt_eig_all_h(1:n_t, jc) = repmat(se_t, [1, numel(jc)]);
                end
            end
            loop_idx = find(~triv_active).';
        else
            loop_idx = 1 : n_active;
        end
        for i = loop_idx
            n_i = double(n_per_rep(i));
            if n_i > 0
                j = full_pos(i);
                V_re_all_h(:, 1:n_i, j)  = single(real(V_per_rep{i}));
                if ~is_real
                    V_im_all_h(:, 1:n_i, j) = single(imag(V_per_rep{i}));
                end
                sqrt_eig_all_h(1:n_i, j) = single(sqrt(eig_per_rep{i}));
            end
        end
    end

    %% Precompute conj(rho_Gamma(g)) tensor (small, 120 * d^2 * 4 B).
    rho_re_all_h = zeros(d, d, group.order, 'single');
    if is_real, rho_im_all_h = zeros(0, 1, 'single');      % real path: no imag half
    else,       rho_im_all_h = zeros(d, d, group.order, 'single'); end
    for g = 1 : group.order
        rho = conj(irrep_matrix(irrep_data, g, d));
        rho_re_all_h(:, :, g) = single(real(rho));
        if ~is_real, rho_im_all_h(:, :, g) = single(imag(rho)); end
    end

    %% #1 -- per-g reduced d x d block for TRIVIAL-TRIVIAL entries. When the
    %  source AND target reps both have a trivial stabiliser they share the
    %  SAME projector V = Ve and sqrt_eig = sqrt(eig_e) (APPLY_IRREP's repmat),
    %  so the kernel's on-the-fly inner contraction
    %     M_e[kp,k] = c_a * sqrt_t[kp]/sqrt_r[k] * (Ve' rho_star(g) Ve)[kp,k]
    %  depends only on g (the coefficient c_a stays outside). Precomputing the
    %  120 d x d matrices
    %     Qbar_g[kp,k] = sqrt(eig_e[kp])/sqrt(eig_e[k]) * (Ve' rho_star(g) Ve)[kp,k]
    %  lets the SpMV look them up instead of reading rho + V_s and doing the
    %  m,n contraction per entry per matvec (~99.6% of entries at M=0). triv_full
    %  marks the trivial reps. Built only when the caller passes triv_active;
    %  empty otherwise -> kernel uses the full contraction everywhere (additive).
    Qbar_re_all_h = zeros(d, d, group.order, 'single');
    if is_real, Qbar_im_all_h = zeros(0, 1, 'single');     % Qbar purely real for real irreps
    else,       Qbar_im_all_h = zeros(d, d, group.order, 'single'); end
    triv_full     = zeros(n_reps_full, 1, 'uint8');
    if nargin >= 9 && ~isempty(triv_active) && any(triv_active)
        triv_full(full_pos(triv_active)) = uint8(1);
        i0t  = find(triv_active, 1);
        n_t0 = double(n_per_rep(i0t));
        Ve   = V_per_rep{i0t};                       % [d x n_t0] shared trivial V
        sqe  = sqrt(eig_per_rep{i0t}(:));            % [n_t0 x 1]
        Dsq  = diag(sqe);
        Dinv = diag(1 ./ sqe);
        for g = 1 : group.order
            rho_star = conj(irrep_matrix(irrep_data, g, d));   % same rho the kernel uses
            Q = Dsq * (Ve' * rho_star * Ve) * Dinv;            % [n_t0 x n_t0]
            Qbar_re_all_h(1:n_t0, 1:n_t0, g) = single(real(Q));
            if ~is_real, Qbar_im_all_h(1:n_t0, 1:n_t0, g) = single(imag(Q)); end
        end
    end

    %% Upload the PER-IRREP arrays only (V/rho/sqrt + per-irrep offsets). The
    %  irrep-INDEPENDENT entry arrays (src/g/c/diag/entries_per_rep/
    %  entry_offsets) and the const-c decision are reused from ESKEL, which was
    %  uploaded ONCE for this M sector (A1-ph2).
    skel_V_re   = gpuArray(V_re_all_h(:));           clear V_re_all_h;
    skel_V_im   = gpuArray(V_im_all_h(:));           clear V_im_all_h;
    skel_sqrt   = gpuArray(sqrt_eig_all_h(:));       clear sqrt_eig_all_h;
    skel_rho_re = gpuArray(rho_re_all_h(:));         clear rho_re_all_h;
    skel_rho_im = gpuArray(rho_im_all_h(:));         clear rho_im_all_h;
    skel_Qbar_re = gpuArray(Qbar_re_all_h(:));       clear Qbar_re_all_h;   % #1
    skel_Qbar_im = gpuArray(Qbar_im_all_h(:));       clear Qbar_im_all_h;
    skel_triv    = gpuArray(triv_full);              clear triv_full;
    skel_v_slot  = gpuArray(v_slot_full);            clear v_slot_full;   % D2 (empty -> full mode)
    skel_rep_offs  = gpuArray(rep_offsets_h);        clear rep_offsets_h;
    skel_n_per_rep = gpuArray(n_per_rep_full);       clear n_per_rep_full;

    %% Pack: per-irrep fields fresh; irrep-independent fields reused from ESKEL.
    skel.is_skeleton    = true;
    skel.is_real        = is_real;     % real FP32 kernel path (V_im/rho_im/Qbar_im empty)
    skel.n_basis        = double(n_basis);
    skel.n_reps         = n_reps_full;     % FULL rep count (A1): the kernel
                                           % launches n_reps_full threads;
                                           % inactive reps have n_per_rep = 0
                                           % and return early.
    skel.d_irrep        = d_irrep;
    skel.rep_offsets    = skel_rep_offs;          % per-irrep
    skel.n_per_rep      = skel_n_per_rep;         % per-irrep
    skel.V_re           = skel_V_re;              % per-irrep
    skel.V_im           = skel_V_im;
    skel.rho_re         = skel_rho_re;
    skel.rho_im         = skel_rho_im;
    skel.sqrt_eig       = skel_sqrt;
    skel.Qbar_re        = skel_Qbar_re;           % per-irrep (#1, empty if no triv_active)
    skel.Qbar_im        = skel_Qbar_im;
    skel.triv           = skel_triv;              % per-rep uint8 trivial-stabiliser flag
    skel.v_slot         = skel_v_slot;            % per-rep int32 0-based V/sqrt slot (D2; empty=full)
    skel.diag_vals       = eskel.diag_vals;        % shared (A1-ph2)
    skel.entries_per_rep = eskel.entries_per_rep;  % shared
    skel.entry_offsets   = eskel.entry_offsets;    % shared
    skel.src_idx         = eskel.src_idx;          % shared
    skel.g_idx           = eskel.g_idx;            % shared
    skel.c_a             = eskel.c_a;              % shared (gpuArray, maybe empty)
    skel.c_a_const       = eskel.c_a_const;        % shared host scalar
    skel.c_idx           = eskel.c_idx;            % shared
    skel.c_table         = eskel.c_table;          % shared
    skel.srcg            = eskel.srcg;             % shared (G2 packed src|g, empty if not)
    skel.is_packed       = eskel.is_packed;        % shared
    skel.g_is_u8         = isfield(eskel, 'g_is_u8') && eskel.g_is_u8;  % shared (G1b uint8 g)
    skel.is_b2           = isfield(eskel,'is_b2') && eskel.is_b2;  % per-entry arrays on HOST (B2)
    % Streaming-B2: rep-tile partition (built in ESKEL when is_b2). Present ->
    % RUN_FTLM may stream the host entry table in tiles per SpMV instead of
    % keeping it resident (the only way the large-d N=36 blocks fit VRAM).
    if isfield(eskel, 'tile_rep_ptr')
        skel.tile_rep_ptr = eskel.tile_rep_ptr;   % int32, n_tiles+1 (0-based rep bounds)
        skel.tile_e_start = eskel.tile_e_start;   % int64, n_tiles
        skel.tile_e_count = eskel.tile_e_count;   % int64, n_tiles
    else
        skel.tile_rep_ptr = int32([]);
        skel.tile_e_start = int64([]);
        skel.tile_e_count = int64([]);
    end
    % Out-of-core: if the eskel's entry table was spilled to a memory-mapped file
    % (spill_entries_mmap), propagate the raw uint64 mapped pointers so RUN_FTLM
    % streams the tiles from NVMe instead of a resident host array.
    if isfield(eskel, 'mmap_srcg'), skel.mmap_srcg = eskel.mmap_srcg; end
    if isfield(eskel, 'mmap_src'),  skel.mmap_src  = eskel.mmap_src;  end
    if isfield(eskel, 'mmap_g'),    skel.mmap_g    = eskel.mmap_g;    end
    if isfield(eskel, 'mmap_cidx'), skel.mmap_cidx = eskel.mmap_cidx; end   % uint8 c-index block (s>=1)
end


% ----------------------------------------------------------------
function skel = empty_skel(d_irrep)
    skel.is_skeleton    = true;
    skel.is_real        = false;       % caller overrides; empty sectors never init the kernel
    skel.n_basis        = 0;
    skel.n_reps         = 0;
    skel.d_irrep        = d_irrep;
    skel.rep_offsets    = gpuArray(zeros(0, 1, 'int64'));
    skel.n_per_rep      = gpuArray(zeros(0, 1, 'int32'));
    skel.diag_vals      = gpuArray(zeros(0, 1, 'single'));
    skel.entries_per_rep = gpuArray(zeros(0, 1, 'int32'));
    skel.entry_offsets  = gpuArray(zeros(0, 1, 'int64'));   % int64 (B2: offsets reach n_entries > 2^31)
    skel.src_idx        = gpuArray(zeros(0, 1, 'int32'));
    skel.g_idx          = gpuArray(zeros(0, 1, 'uint16'));
    skel.c_a            = gpuArray(zeros(0, 1, 'single'));
    skel.c_a_const      = single(0);
    skel.c_idx          = gpuArray(zeros(0, 1, 'uint8'));
    skel.c_table        = gpuArray(zeros(0, 1, 'single'));
    skel.srcg           = gpuArray(zeros(0, 1, 'uint32'));
    skel.is_packed      = false;
    skel.g_is_u8        = false;
    skel.tile_rep_ptr   = int32([]);
    skel.tile_e_start   = int64([]);
    skel.tile_e_count   = int64([]);
    skel.Qbar_re        = gpuArray(zeros(0, 1, 'single'));
    skel.Qbar_im        = gpuArray(zeros(0, 1, 'single'));
    skel.triv           = gpuArray(zeros(0, 1, 'uint8'));
    skel.V_re           = gpuArray(zeros(d_irrep*d_irrep*0, 1, 'single'));
    skel.V_im           = gpuArray(zeros(d_irrep*d_irrep*0, 1, 'single'));
    skel.rho_re         = gpuArray(zeros(0, 1, 'single'));   % unused: n_basis=0, rho never indexed
    skel.rho_im         = gpuArray(zeros(0, 1, 'single'));
    skel.sqrt_eig       = gpuArray(ones(d_irrep*0, 1, 'single'));
end


function M = irrep_matrix(irrep_data, g, d)
    if d == 1
        M = complex(irrep_data(g));
    else
        M = complex(irrep_data(:, :, g));
    end
end
