function info = ftlm_observables_pg_gpu_Ih(input_file)
%FTLM_OBSERVABLES_PG_GPU_IH  I_h-FTLM with mixed GPU FP32 / CPU FP64.
%
%   INFO = FTLM_OBSERVABLES_PG_GPU_IH(...) additionally returns a small info
%   struct (irrep_list/irrep_d/M_run/geo_tag/s_str/mat_name/...) consumed by
%   FTLM_ORCHESTRATE_SECTORS to split the block list across GPU workers.
%   Orchestration inputs (both default-off, normal runs byte-identical):
%       precompute_dir   ('')      directory for the precompute caches
%                                  (default: output_dir). Lets several
%                                  sector-parallel workers SHARE one cache.
%       precompute_only  (false)   run only the per-M enumerate+collect and
%                                  persist the caches, then return (no FTLM,
%                                  no result .mat). Requires precompute_cache.
%
%   FTLM_OBSERVABLES_PG_GPU_IH(INPUT_FILE) is the GPU-accelerated sister
%   of FTLM_OBSERVABLES_PG_IH. For each (M, Gamma) block of the
%   icosahedron under I_h symmetry, the driver:
%
%     1. enumerates super-reps and column-basis matrices V_r,
%        eigenvalues lambda_r via ENUMERATE_SECTOR_WITH_IH_GAMMA2;
%     2. builds the compressed lookup table CLT via BUILD_CLT_PG_IH;
%     3. dispatches the block to one of three paths:
%          - dense ED (n_basis <= ed_thresh, deterministic),
%          - GPU FP32 block-Lanczos (CUDA_LANCZOS_CLUT_BLOCK_PG_IH),
%          - CPU FP64 fallback (if the MEX kernel is unavailable);
%     4. aggregates Ritz values + FTLM weights with the
%        mult_M * d_Gamma multiplicity into a single observables run.
%
%   The GPU path uses one CUDA thread per output rep and the gather-
%   style CLT SpMV with no runtime min-image search. Aggregation
%   across (M, Gamma) sectors and computation of C(T), chi(T),
%   Z_eff(T) are byte-identical to the CPU driver.
%
%   Input file syntax matches FTLM_OBSERVABLES_PG_IH plus the optional
%   GPU-specific knobs:
%       B_gpu          (0)         GPU block-Lanczos block size (0 = adaptive)
%
%   Output: a single .mat file named ftlm_pg_gpu_Ih_icos_s<s_str>.mat
%   containing T_range, C_T, chi_T, Z_eff, configuration, per-sector
%   diagnostics including which dispatch path was used.
%
%   See also FTLM_OBSERVABLES_PG_IH, RUN_FTLM_PG_SECTOR_GPU_IH,
%            ENUMERATE_M_ORBITS_IH_GPU, COLLECT_CLT_ENTRIES_IH,
%            BUILD_CLT_FROM_ENTRIES_IH, APPLY_IRREP_TO_ORBITS,
%            BUILD_HEISENBERG_SPARSE_IH_GAMMA2.
%
%   ----------------------------------------------------------------
%   PRODUCTION PATH (post-consolidation, May 2026)
%   ----------------------------------------------------------------
%   Per M sector (called once, irrep-INDEPENDENT):
%     ENUMERATE_M_ORBITS_IH_GPU
%         M-filter chunked on gpuArray + stabiliser flags via one BLAS
%         matmul. Internal CPU fallback if no CUDA device.
%     COLLECT_CLT_ENTRIES_IH
%         Lifts Phase 1 of the CLT (60 vectorised MIN_IMAGE_IH calls)
%         out of the irrep loop. Produces the (src, tgt, g, c_a) entry
%         table shared by all 10 irreps.
%
%   Per (M, Gamma) sector (irrep-DEPENDENT):
%     APPLY_IRREP_TO_ORBITS
%         Builds V_r / lambda_r from the cached stabiliser list.
%     BUILD_CLT_FROM_ENTRIES_IH
%         Phase 2: batched pagemtimes M_e construction on the host.
%         Outperforms GPU pagemtimes (Option A) on d <= 5 by a clear
%         margin on the tested RTX 4000 SFF Ada.
%     RUN_FTLM_PG_SECTOR_GPU_IH
%         CUDA FP32 block-Lanczos with precomputed M tensor
%         ('init' / 'spmv' / 'block_lanczos' modes).
%
%   ARCHIVED ALTERNATIVES (kept as references, NOT called from here):
%     BUILD_CLT_PG_IH                       - monolithic CLT (regression ref)
%     BUILD_CLT_SKELETON_PG_IH              - superseded skeleton
%     BUILD_CLT_FROM_ENTRIES_IH_GPU         - Option A (slower)
%     BUILD_CLT_SKELETON_FROM_ENTRIES_IH    - Option B (kernel spills d>=4)
%   See BENCH_S2_M0_BREAKDOWN(mode) to re-benchmark any of them.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    if nargin < 1 || isempty(input_file)
        error('ftlm_observables_pg_gpu_Ih:NoInput', ...
              'Usage: ftlm_observables_pg_gpu_Ih(''input.m'')');
    end
    if exist(input_file, 'file') ~= 2
        error('ftlm_observables_pg_gpu_Ih:InputNotFound', ...
              'Input file not found: %s', input_file);
    end

    fprintf('=== ftlm_observables_pg_gpu_Ih: symmetry-adapted FTLM (mixed GPU FP32 / CPU FP64) ===\n');
    fprintf('Input file: %s\n\n', input_file);

    run(input_file);   %#ok<*NODEF>

    %% Required + optional inputs
    required = {'s_val', 'J', 'R', 'M_lz', 'T_range'};
    for k = 1 : numel(required)
        if ~exist(required{k}, 'var')
            error('ftlm_observables_pg_gpu_Ih:Missing', ...
                  'Required variable missing in %s: %s', input_file, required{k});
        end
    end

    if ~exist('only_M0',    'var'), only_M0    = false;          end
    if ~exist('ed_thresh',  'var'), ed_thresh  = 0;              end
    if ~exist('output_dir', 'var'), output_dir = '.';            end
    if ~exist('B_gpu',      'var'), B_gpu      = 0;              end
    if ~exist('geometry',   'var'), geometry   = 'icosahedron';  end
    if ~exist('mem_diag',   'var'), mem_diag   = false;          end
    % lz_diag: per-sector Lanczos convergence report (print-only) -- Ritz
    % residual counts, beta floor, E0 residual, unconverged weight. Use it to
    % tune M_lz / R from data (kernel time is linear in M_lz). Results are
    % byte-identical with the flag on or off.
    if ~exist('lz_diag',    'var'), lz_diag    = false;          end
    if ~exist('lookup_method','var'), lookup_method = 'bitmap';  end
    if ~exist('entries_storage','var'), entries_storage = 'host'; end
    if ~exist('checkpoint',   'var'), checkpoint = false;         end
    if ~exist('entries_on_disk', 'var'), entries_on_disk = false;  end
    % ondisk_scratch_dir: optional NODE-LOCAL scratch for the on-disk entry
    % files (e.g. the 3.4-TB /tmp NVMe of the B200 node). Default '' keeps
    % them under output_dir. Motivation: the finalize holds ~2x the table
    % transiently, so concurrent big per-M jobs blew the 600-GB home quota
    % (s=3/2 campaign, 2026-07-03) -> put SCRATCH on the node, keep results
    % and checkpoints on the shared filesystem. Ignored when the precompute
    % cache owns the od files (they must persist/be shared then).
    if ~exist('ondisk_scratch_dir', 'var'), ondisk_scratch_dir = ''; end
    % prefix_budget_gb: EXPERIMENT/PROFILING knob -- explicit
    % cap (in GB) for the VRAM-resident leading prefix of an out-of-core entry
    % table. Default [] keeps the production AUTO sizing (-1, kernel fills the
    % free VRAM). A small value (e.g. 2) forces the SpMV to stream nearly the
    % whole table per iteration: emulates the M=1-5 streamed-tail regime on the
    % (cheaper, cached) M=0 table. Bit-identical to AUTO per test_stream_prefix.
    if ~exist('prefix_budget_gb', 'var'), prefix_budget_gb = []; end
    % esort_src: R2-lite-EXPERIMENT -- Sekundaersortierung der
    % Entry-Liste jeder Ziel-Rep nach SOURCE-Rep. Reine Datenordnung (der
    % Kernel laeuft jede tgt-Liste linear ab; aufsteigende src machen die
    % V[src]-Gathers monoton -> L2/TLB-Lokalitaet). NICHT bit-identisch
    % (per-tgt-FP-Summationsreihenfolge aendert sich); Sum Rule bleibt
    % exakt. Nur Host-/B2-/Stream-Pfade (eine od-Datei braeuchte einen
    % ebs-Re-Sort). Default aus.
    if ~exist('esort_src', 'var'), esort_src = false; end
    % tiled_spmv: R2-v1-EXPERIMENT -- getilter SpMV: Entries pro
    % Irrep (src_tile, tgt)-sortiert, ein Kernel-Launch pro src-Tile, dessen
    % V[src]-Gathers im L2-Fenster bleiben. Nur resident-ref-Pfad (real);
    % Klasse C (Summationsreihenfolge). tiled_window_gb = Fensterbudget.
    if ~exist('tiled_spmv', 'var'), tiled_spmv = false; end
    % Default-Fenster geraetebasiert: 0.8x der L2-Groesse der Karte (via
    % 'l2_size'-Query des Kernels, analog zum Gather-Cap in run_ftlm) statt
    % der versteckten AD104-Eichung 0.04 -- die verschenkte 2/3 des B200-L2
    % (126 MB) bzw. ueberlief kleine L2s (24-32 MB). AD104-Fallback 48e6
    % (stale MEX / kein Kernel) -> 0.0384, praktisch der alte Dev-Box-Wert.
    % Ein explizit im Deck gesetztes tiled_window_gb bleibt unangetastet.
    if ~exist('tiled_window_gb', 'var')
        l2b = 48e6;                       % AD104-Fallback (stale MEX)
        try, l2b = cuda_lanczos_clut_block_pg_Ih('l2_size'); catch, end
        tiled_window_gb = 0.8 * double(l2b) / 1e9;
    end
    if ~exist('precompute_cache','var'), precompute_cache = false; end
    if ~exist('use_spin_flip',  'var'), use_spin_flip   = false;  end
    % force_complex: keep the pre-real-kernel behavior -- I_h named irreps stay
    % complex (NOT realified) and real (realified) irreps still take the COMPLEX
    % GPU layout (zero-imag arrays). Reproduces old baselines / enables an A/B
    % of the real-vs-complex kernel on the SAME matrix. Default off: is_real
    % follows from the irreps (realified space groups + realified I_h -> real).
    if ~exist('force_complex',  'var'), force_complex   = false;  end
    % Orchestration (sector-parallel multi-GPU, see FTLM_ORCHESTRATE_SECTORS):
    % a SHARED precompute-cache directory + a precompute-only pass.
    if ~exist('precompute_dir', 'var') || isempty(precompute_dir)
        precompute_dir = '';                  % '' -> output_dir (set below)
    end
    if ~exist('precompute_only', 'var'), precompute_only = false; end
    if precompute_only
        assert(precompute_cache, ...
            'precompute_only requires precompute_cache=true (nothing would persist).');
        % (entries_on_disk + precompute_cache is the multi-GPU table-sharing
        % combination: phase 0 then also writes the shared on-disk entry
        % files, which persist with the cache.)
    end
    have_irrep_list = exist('irrep_list', 'var') && ~isempty(irrep_list);

    % Create output_dir up front: the precompute cache, the per-irrep
    % checkpoint and the final .mat all save into it, and a missing
    % directory used to kill the run only AFTER the (minutes-long)
    % enumerate+collect had already finished.
    if ~exist(output_dir, 'dir'), mkdir(output_dir); end
    if isempty(precompute_dir), precompute_dir = output_dir; end
    if ~exist(precompute_dir, 'dir'), mkdir(precompute_dir); end
    info = struct();

    %% Validation
    assert(s_val > 0 && abs(2*s_val - round(2*s_val)) < 1e-12, ...
        's_val must be a positive half-integer.');
    assert(isfinite(J) && isnumeric(J) && isscalar(J), 'J must be a finite scalar.');
    assert(R >= 1 && R == round(R), 'R must be a positive integer.');
    assert(M_lz >= 1 && M_lz == round(M_lz), 'M_lz must be a positive integer.');
    assert(all(T_range > 0) && all(isfinite(T_range)), ...
        'T_range must contain positive finite values.');
    T_range = T_range(:).';

    %% GPU init
    assert(gpuDeviceCount > 0, 'No CUDA-capable GPU found.');
    gpu_h = gpuDevice;
    % The reset below destroys the CUDA context. If the kernel MEX still
    % holds state from a PREVIOUS driver run in this session (e.g. a KEPT
    % streaming table: keep_table leaves it pending by design), its raw
    % device pointers would dangle across the reset and the next kernel
    % cleanup would free invalid pointers (harmless but noisy; see the
    % absorb-note in cleanup_core). Drop the kernel state FIRST, while its
    % context is still alive.
    if exist('cuda_lanczos_clut_block_pg_Ih', 'file') == 3
        try, cuda_lanczos_clut_block_pg_Ih('cleanup'); catch, end
    end
    reset(gpu_h);
    gpu_h = gpuDevice;
    fprintf('GPU: %s (%.1f GB VRAM)\n', gpu_h.Name, gpu_h.TotalMemory/1e9);
    have_kernel = (exist('cuda_lanczos_clut_block_pg_Ih', 'file') == 3);
    if ~have_kernel
        fprintf(['Note: cuda_lanczos_clut_block_pg_Ih MEX not found.\n', ...
                 '      Every block will fall back to the CPU FP64 path.\n', ...
                 '      Run build_pg_kernels to enable the I_h GPU kernel.\n\n']);
    end

    %% System setup. The geometry switch picks (group, bonds, N_sites)
    %  from the appropriate provider. Add new geometries here.
    switch lower(geometry)
        case 'icosahedron'
            group  = icosahedron_Ih_full();
            bonds  = adjacency_icosahedron_Ih();
            sys_label = '12-site icosahedron';
            sym_label = 'I_h (|G|=120)';
            geo_tag   = 'icos';
        case 'icosidodecahedron'
            group  = icosidodecahedron_Ih_full();
            bonds  = adjacency_icosidodecahedron_Ih();
            sys_label = '30-site icosidodecahedron';
            sym_label = 'I_h (|G|=120)';
            geo_tag   = 'icosido';
        case 'dodecahedron'
            group  = dodecahedron_Ih_full();
            bonds  = adjacency_dodecahedron_Ih();
            sys_label = '20-site dodecahedron';
            sym_label = 'I_h (|G|=120)';
            geo_tag   = 'dodec';
        case 'cuboctahedron'
            % Quasiregular O_h Archimedean solid (12 vertices, 24 edges,
            % 8 corner-sharing triangles -> frustrated). Irreps generic:
            % 10 irreps, max d=3, all FS=+1 -> realified (real kernel path).
            [group, bonds] = cuboctahedron_Oh();
            group.irreps = irreps_from_group(group);
            group.irreps = realify_irreps(group.irreps, group);   % FS=+1: real orthogonal irreps
            sys_label = '12-site cuboctahedron';
            sym_label = sprintf('O_h (|G|=%d)', group.order);
            geo_tag   = 'cubocta';
        case 'square_lattice'
            assert(exist('Lx', 'var') == 1 && exist('Ly', 'var') == 1, ...
                'square_lattice geometry requires Lx and Ly in the input file.');
            group  = square_lattice_translation_group(Lx, Ly);
            bonds  = adjacency_square_lattice(Lx, Ly);
            sys_label = sprintf('%dx%d periodic square lattice (N=%d)', Lx, Ly, Lx*Ly);
            sym_label = sprintf('C_%d x C_%d (|G|=%d)', Lx, Ly, Lx*Ly);
            geo_tag   = sprintf('sq%dx%d', Lx, Ly);
        case 'square_lattice_c4v'
            assert(exist('Lx', 'var') == 1 && exist('Ly', 'var') == 1, ...
                'square_lattice_c4v geometry requires Lx and Ly in the input file.');
            group  = square_lattice_spacegroup(Lx, Ly);
            group.irreps = irreps_from_group(group);
            % The point group contains C_2 (-k is always in the star of k), so
            % every irrep is self-conjugate with Frobenius-Schur indicator +1:
            % realify them to real orthogonal form. This makes the CPU-fallback
            % / dense-ED H blocks real-symmetric AND routes the GPU blocks to
            % the REAL FP32 kernel path (June 2026: half the Krylov VRAM and
            % gather traffic; force_complex restores the complex layout). The
            % abelian translation path (complex k-irreps) is NOT touched; the
            % I_h polyhedra realify at build_full_irrep_table below.
            group.irreps = realify_irreps(group.irreps, group);
            bonds  = adjacency_square_lattice(Lx, Ly);
            sys_label = sprintf('%dx%d square lattice + %s point group (N=%d)', ...
                Lx, Ly, group.point_group, Lx*Ly);
            sym_label = sprintf('%s space group (|G|=%d)', group.point_group, group.order);
            geo_tag   = sprintf('sq%dx%dsg', Lx, Ly);
        case 'kagome'
            assert(exist('kag_a','var')==1 && exist('kag_b','var')==1, ...
                'kagome geometry requires kag_a and kag_b in the input file.');
            [group, bonds] = kagome_spacegroup(kag_a, kag_b);
            group.irreps = irreps_from_group(group);
            group.irreps = realify_irreps(group.irreps, group);   % FS=+1: real orthogonal irreps
            sys_label = sprintf('kagome torus (a,b)=(%d,%d), N=%d', kag_a, kag_b, group.N);
            sym_label = sprintf('%s space group (|G|=%d)', group.point_group, group.order);
            geo_tag   = sprintf('kag%d_%d', kag_a, kag_b);
        case 'triangular'
            assert(exist('tri_a','var')==1 && exist('tri_b','var')==1, ...
                'triangular geometry requires tri_a and tri_b in the input file.');
            [group, bonds] = triangular_spacegroup(tri_a, tri_b);
            group.irreps = irreps_from_group(group);
            group.irreps = realify_irreps(group.irreps, group);   % FS=+1: real orthogonal irreps
            sys_label = sprintf('triangular torus (a,b)=(%d,%d), N=%d', tri_a, tri_b, group.N);
            sym_label = sprintf('%s space group (|G|=%d)', group.point_group, group.order);
            geo_tag   = sprintf('tri%d_%d', tri_a, tri_b);
        case 'generators'
            % User-supplied symmetry: the input file gives only the permutation
            % GENERATORS (cell/matrix of 1xN site perms); group_from_generators
            % closes them into the full group in the standard provider layout, so
            % systems NOT hard-coded above can be run. The nearest-neighbour bond
            % list is geometry, supplied SEPARATELY in the input file ('bonds'),
            % and MUST be invariant under every group element (correctness cond.).
            assert(exist('gens', 'var') == 1, ...
                ['generators geometry requires permutation generators ''gens'' ', ...
                 '(cell/matrix of 1xN perms) in the input file.']);
            assert(exist('bonds', 'var') == 1 && ~isempty(bonds), ...
                ['generators geometry requires a bond list ''bonds'' [n_bonds x 2] ', ...
                 'in the input file (geometry is separate from symmetry).']);
            if exist('point_group', 'var') == 1, pg_lab = point_group; else, pg_lab = 'custom'; end
            group = group_from_generators(gens, [], pg_lab);
            % HARD GUARD before the |G|^2-cost irrep build: a non-invariant
            % bond list would give silently WRONG physics (the sum rule is a
            % trace identity and does NOT catch it).
            assert_bonds_group_invariant(bonds, group, 'generators geometry');
            group.irreps = irreps_from_group(group);
            group.irreps = realify_irreps(group.irreps, group);   % FS=+1: real orthogonal irreps
            if exist('sys_name', 'var') == 1, geo_tag = sys_name; else, geo_tag = 'custom'; end
            sys_label = sprintf('user-generated group ''%s'' (N=%d, |G|=%d)', ...
                geo_tag, group.N, group.order);
            sym_label = sprintf('user permutation group (|G|=%d)', group.order);
        otherwise
            error('ftlm_observables_pg_gpu_Ih:Geometry', ...
                  ['Unknown geometry "%s" (icosahedron / dodecahedron / ', ...
                   'icosidodecahedron / cuboctahedron / square_lattice / ', ...
                   'square_lattice_c4v / kagome / triangular / generators).'], geometry);
    end

    % Pre-flight feasibility estimate (n_reps / n_entries / host RAM / VRAM mode +
    % the kernel hard limits) computed from the provider BEFORE the multi-minute
    % enumerate, so an infeasible launch fails fast instead of thrashing. The
    % report is informative; only the HOST gate errors here (skip_feasibility=true
    % overrides it). The d>12 (MAX_D) and |G|>65535 (uint16 g) hard caps are
    % asserted unconditionally below / in collect, regardless of skip_feasibility.
    if ~(exist('skip_feasibility', 'var') == 1 && skip_feasibility)
        est_pf = estimate_feasibility(group, bonds, s_val, R, M_lz, true, ...
                                      use_spin_flip, force_complex);
        if ~est_pf.host_ok
            if entries_on_disk && est_pf.host_od_ok
                % The host_GB estimate models the IN-RAM collect peak (~24 B
                % per entry). With entries_on_disk the entry TABLE streams to
                % disk buckets; what remains in RAM is the od floor (digit
                % matrix + per-rep arrays + EBS finalize bucket), which fits.
                fprintf(['Pre-flight: in-RAM host estimate ~%.0f GB exceeds the budget, ', ...
                         'but entries_on_disk=true bounds the entry table; the remaining ', ...
                         'collect floor ~%.0f GB fits in ~%.0f GB -- continuing.\n'], ...
                        est_pf.host_GB, est_pf.host_od_GB, est_pf.host_budget_GB);
            elseif entries_on_disk
                % The blanket "od bounds the collect" bypass OOMed the dodec
                % s=3/2 run at the SLURM cgroup cap (2026-07): the od floor
                % itself must fit the JOB's memory budget.
                error('ftlm_observables_pg_gpu_Ih:Infeasible', ...
                    ['Pre-flight: even the entries_on_disk collect floor (~%.0f GB: digit ', ...
                     'matrix + per-rep arrays + EBS finalize bucket) exceeds the host ', ...
                     'budget (~%.0f GB -- node MemAvailable capped by the cgroup/SLURM ', ...
                     'allocation where present). Request more memory (sbatch --mem) or ', ...
                     'set skip_feasibility=true to override.'], ...
                    est_pf.host_od_GB, est_pf.host_budget_GB);
            else
                error('ftlm_observables_pg_gpu_Ih:Infeasible', ...
                    'Pre-flight feasibility: %s. Set skip_feasibility=true to override.', ...
                    strjoin(est_pf.notes, '; '));
            end
        end
    end

    % D2 compact-V: needed for high-d irreps on large sectors (the per-rep V
    % tensor is d^2*n_reps; e.g. d=6 kagome / d=8 square-c4v would store
    % 16-18 GB). Auto-enable whenever the provider supplies multi-dim irreps
    % (group.irreps with max d >= 2); the I_h / translation paths are untouched.
    % Bit-identical either way.
    % Kernel hard cap MAX_D = 12 (cuda_lanczos_clut_block_pg_Ih.cu #define MAX_D):
    % the OTF SpMV caches V_t[:,kp] (d floats) plus B accumulators per thread, so
    % an irrep with d > 12 would overrun those registers -> refuse early with an
    % actionable message (a larger d needs a kernel MAX_D bump + recompile,
    % watching register pressure). The CPU driver does dense ED so has no such cap.
    if isfield(group, 'irreps')
        max_irrep_d = max(arrayfun(@(s) s.d, group.irreps));
        assert(max_irrep_d <= 12, ...
            ['ftlm_observables_pg_gpu_Ih: irrep dimension d=%d exceeds the kernel ' ...
             'MAX_D=12. Raise #define MAX_D in cuda_lanczos_clut_block_pg_Ih.cu and ' ...
             'recompile, or use a smaller-|G| symmetry.'], max_irrep_d);
        compact_v_flag = (max_irrep_d >= 2);
    else
        % Named-irrep providers (I_h): max d = 5 (Hg/Hu), safely under MAX_D.
        % Compact-V was historically never enabled on this branch, so every
        % irrep rebuilt, filled and UPLOADED the full per-rep V tensor
        % (d^2 * n_reps_full singles: ~7.4 GB host + VRAM for a dodec-s=3/2
        % d=5 block) although >99% of reps share the ONE trivial-stabiliser
        % (V, eig). apply_irrep_to_orbits returns triv_active on the I_h path
        % too, the skeleton builder is bit-identical either way
        % (test_compact_v) and degrades to full-V when no trivial rep exists,
        % so mirror the space-group rule. For the icosahedron s=5 (>2^31)
        % targets compact-V is an ENABLER: the full d=5 V tensor (~60 GB)
        % would not fit any card.
        compact_v_flag = true;
    end
    N_sites = double(group.N);
    d_loc   = round(2*s_val + 1);
    n_total = double(d_loc)^N_sites;
    M_max   = round(N_sites * s_val);

    %% M-sector selection. Input M_sectors (vector of non-negative |M| in 0..M_max)
    %  restricts the run to those sectors; else only_M0 -> {0}; else the full sweep
    %  0..M_max (the physical C(T)+chi(T)). The +/-M mirror is folded via mult_M, so
    %  list only non-negative |M|. M_sectors overrides only_M0.
    if exist('M_sectors', 'var') && ~isempty(M_sectors)
        M_run = unique(round(abs(M_sectors(:)')));
        assert(all(M_run >= 0 & M_run <= M_max), ...
            'M_sectors must lie in 0..M_max=%d (got [%s]).', M_max, num2str(M_run));
    elseif only_M0
        M_run = 0;
    else
        M_run = 0 : M_max;
    end
    is_full_sweep = isequal(M_run, 0:M_max);

    % The dense-ED branch (build_heisenberg_sparse_Ih_gamma2) uses the
    % 32-state bitmap lookup internally -> it requires n_total <= 2^32. A
    % FULL sweep guarantees tiny high-M sectors that fall below ed_thresh,
    % so with ed_thresh > 0 it would crash there after hours of good
    % sectors (bench incident, icosahedron s=3 at M=29). Fail at STARTUP
    % instead; for restricted M lists it is only a latent risk -> warn.
    if ed_thresh > 0 && n_total > 2^32
        if is_full_sweep
            error('ftlm_observables_pg_gpu_Ih:edThresh', ...
                ['ed_thresh = %g > 0 with n_total = %.3g > 2^32: the dense-ED ', ...
                 'builder (bitmap lookup) cannot handle this system, and a full ', ...
                 'M sweep always reaches blocks below ed_thresh. Set ed_thresh = 0 ', ...
                 '(all blocks via the GPU kernel).'], ed_thresh, n_total);
        else
            warning('ftlm_observables_pg_gpu_Ih:edThresh', ...
                ['ed_thresh = %g > 0 with n_total = %.3g > 2^32: any block with ', ...
                 'n_basis <= ed_thresh would crash in the dense-ED builder ', ...
                 '(needs n_total <= 2^32). Consider ed_thresh = 0.'], ed_thresh, n_total);
        end
    end

    two_s = round(2 * s_val);
    if mod(two_s, 2) == 0, s_str = sprintf('%d', two_s/2);
    else,                  s_str = sprintf('%do2', two_s); end

    %% Irreps: generic list from the provider (group.irreps) or the I_h
    %  named fields. Default to ALL irreps when the input gives no list.
    irreps_all = build_full_irrep_table(group);
    % I_h polyhedra: the named irreps T1g..Hu are stored partly COMPLEX although
    % all ten have FS = +1 -> realify them to real orthogonal form (real V/H
    % blocks + the REAL FP32 GPU kernel path). force_complex skips this, which
    % reproduces the historical complex-irrep runs byte-identically.
    if ~isfield(group, 'irreps') && ~force_complex
        irreps_all = realify_irrep_table(irreps_all, group);
    end
    if have_irrep_list
        irreps = filter_irreps(irreps_all, irrep_list);
    else
        irreps     = irreps_all;
        irrep_list = cellfun(@(c) c.name, irreps, 'uni', 0);
    end

    %% Spin-flip Z2 (opt-in `use_spin_flip`): extend G -> G x Z2 by the
    %  phase-free spin inversion P (Sz -> -Sz) and split each Gamma into
    %  Gamma+/Gamma- with rho'((g,f)) = (+-1)^f rho(g). Applied ONLY to the
    %  M = 0 sector (P maps M to -M), where it roughly HALVES n_reps,
    %  n_entries and every block's n_basis. The flip irrep list is doubled
    %  from the (already filtered, possibly realified) driver list, so it
    %  works for both the generic space-group and the named-I_h providers.
    %  See ADD_SPIN_FLIP_Z2.
    if use_spin_flip
        group_sf  = add_spin_flip_z2(group);
        irreps_sf = cell(1, 0);
        for kk = 1 : numel(irreps)
            c0 = irreps{kk};
            for sgn = [1, -1]
                c2 = c0;
                if c0.d == 1
                    mm      = c0.data(:).';
                    c2.data = [mm, sgn * mm];
                else
                    c2.data = cat(3, c0.data, sgn * c0.data);
                end
                if sgn > 0, c2.name = [c0.name '+']; else, c2.name = [c0.name '-']; end
                irreps_sf{end + 1} = c2; %#ok<AGROW>
            end
        end
        fprintf('Spin-flip Z2 ON for M=0: |G| %d -> %d, %d -> %d irreps (Gamma+-).\n', ...
                group.order, group_sf.order, numel(irreps), numel(irreps_sf));
    end

    %% Info struct (for the sector-parallel orchestrator): the BASE (pre-flip)
    %  block list actually run + everything the merge needs. Filled before the
    %  M loop so a precompute_only pass returns it too.
    mat_name = sprintf('ftlm_pg_gpu_Ih_%s_s%s.mat', geo_tag, s_str);
    info.irrep_list    = irrep_list;                       % base names (flip doubles each into +-)
    info.irrep_d       = cellfun(@(c) c.d, irreps);
    info.use_spin_flip = use_spin_flip;
    info.M_run         = M_run;
    info.M_max         = M_max;
    info.N_sites       = N_sites;
    info.s_val         = s_val;
    info.n_total       = n_total;
    info.geo_tag       = geo_tag;
    info.s_str         = s_str;
    info.mat_name      = mat_name;
    info.is_full_sweep = is_full_sweep;

    fprintf('System:    %s, s=%s (d_loc=%d, n_total=%g, M_max=%d)\n', ...
        sys_label, s_str, d_loc, n_total, M_max);
    fprintf('Symmetry:  %s on %d / %d irreps\n', ...
        sym_label, numel(irreps), numel(irreps_all));
    fprintf('FTLM:      R=%d, M_lz=%d, T-grid: %d points in [%.3g, %.3g]\n', ...
        R, M_lz, numel(T_range), min(T_range), max(T_range));
    if is_full_sweep
        fprintf('Full M sweep (M = 0..%d) -> physical C(T) and chi(T)\n', M_max);
    else
        fprintf('Restricted to M sectors {%s} of 0..%d (partial / sector observables)\n', ...
            num2str(M_run), M_max);
    end
    if ed_thresh == inf
        fprintf('Exact ED for every block (ed_thresh = inf)\n');
    elseif ed_thresh > 0
        fprintf('Dense ED for blocks with n_basis <= %d, FTLM otherwise\n', ed_thresh);
    else
        fprintf('FTLM throughout (ed_thresh = 0)\n');
    end
    fprintf('\n');

    %% Main loop over (M, Gamma)
    all_E       = zeros(0, 1);
    all_w       = zeros(0, 1);
    all_M       = zeros(0, 1);
    sector_M    = zeros(0, 1);
    sector_G    = zeros(0, 1);
    sector_dims = zeros(0, 1);
    sector_path = {};
    M_start     = 0;
    ig_resume   = 1;     % first irrep index to (re)compute within M_start on resume

    %% Per-IRREP checkpoint / resume (opt-in via `checkpoint = true`). After
    %  every (M, Gamma) sector the accumulated (E, w, M) + diagnostics are
    %  written ATOMICALLY (temp file + rename) together with the position
    %  (M_cur, ig_done), so a crash/reboot costs at most ONE irrep (plus
    %  redoing the current M's enumerate/collect precompute). Essential for
    %  long M=0-ONLY runs, where a per-M checkpoint would never fire mid-run.
    %  The accumulators are small (~MB); config-stamped (mismatch -> fresh);
    %  deleted on successful completion.
    %  K7c env fingerprint: FTLM_FP16 CHANGES the Lanczos numerics (the kernel
    %  stores v/vp/w as __half) -- a requeue with drifted env would silently
    %  mix fp16/fp32 sectors in ONE result .mat. Requeue+resume is the DESIGN
    %  mode of the heavy per-M campaign jobs, so fp16 is stamped into the
    %  checkpoint and a mismatch is a HARD error (below), not a silent fresh
    %  start that throws away days of sector work. esort/tiled are stamped as
    %  cheap class-C guards against deck drift (they change summation order,
    %  not physics). FTLM_R3/FTLM_LEVER_A are bit-identical pipelines and
    %  deliberately NOT stamped. NB: adding fields invalidates PRE-K7c
    %  checkpoints (isequal mismatch -> warning + fresh start) -- deploy
    %  between campaigns, not mid-run.
    ckpt_cfg  = struct('s_val', s_val, 'J', J, 'geometry', lower(geometry), ...
        'N_sites', N_sites, 'R', R, 'M_lz', M_lz, 'M_max', M_max, ...
        'only_M0', only_M0, 'M_run', M_run, 'ed_thresh', ed_thresh, ...
        'n_irreps', numel(irreps), 'nT', numel(T_range), ...
        'use_spin_flip', use_spin_flip, 'force_complex', force_complex, ...
        'fp16', strcmp(getenv('FTLM_FP16'), '1'), ...
        'fp16_scale', strcmp(getenv('FTLM_FP16'), '1') ...
            && ~strcmp(getenv('FTLM_FP16_SCALE'), '0'), ...   % the 2^k storage grid changes the fp16 rounding
        'esort', logical(esort_src), 'tiled', logical(tiled_spmv));
    ckpt_path = fullfile(output_dir, sprintf('ckpt_ftlm_gpu_%s_s%s.mat', geo_tag, s_str));
    if checkpoint && exist(ckpt_path, 'file')
        cp = load(ckpt_path);
        if isfield(cp, 'cfg') && isfield(cp.cfg, 'fp16') ...
                && cp.cfg.fp16 ~= ckpt_cfg.fp16
            % FP16 drift is the one mismatch that must NOT silently restart:
            % the checkpoint holds expensive finished sectors and the fix is
            % a one-line env change on resubmit.
            error('ftlm_observables_pg_gpu_Ih:ckptFp16', ...
                ['Checkpoint %s was written with FTLM_FP16=%d but this run ', ...
                 'has FTLM_FP16=%d. Mixed fp16/fp32 sectors in one result ', ...
                 'would be silent poison. Either relaunch with the matching ', ...
                 'env (FTLM_FP16=%d) to resume, or delete the checkpoint ', ...
                 'file to start fresh under the new env.'], ...
                ckpt_path, cp.cfg.fp16, ckpt_cfg.fp16, cp.cfg.fp16);
        end
        if isfield(cp, 'cfg') && isfield(cp.cfg, 'fp16_scale') ...
                && cp.cfg.fp16_scale ~= ckpt_cfg.fp16_scale
            % Same logic as the fp16 drift above: the 2^k storage grid
            % changes the fp16 rounding -- mixed sectors would be silent
            % poison.
            error('ftlm_observables_pg_gpu_Ih:ckptFp16Scale', ...
                ['Checkpoint %s was written with fp16_scale=%d but this run ', ...
                 'has fp16_scale=%d (env FTLM_FP16_SCALE). Relaunch with the ', ...
                 'matching env to resume, or delete the checkpoint to start ', ...
                 'fresh.'], ckpt_path, cp.cfg.fp16_scale, ckpt_cfg.fp16_scale);
        end
        if isfield(cp, 'cfg') && isequal(cp.cfg, ckpt_cfg)
            all_E = cp.all_E;  all_w = cp.all_w;  all_M = cp.all_M;
            sector_M = cp.sector_M;  sector_G = cp.sector_G;
            sector_dims = cp.sector_dims;  sector_path = cp.sector_path;
            M_start = cp.M_cur;  ig_resume = cp.ig_done + 1;
            n_ig_Mcur = numel(irreps);           % irrep count of M_cur's list
            if use_spin_flip && M_start == 0, n_ig_Mcur = numel(irreps_sf); end
            if ig_resume > n_ig_Mcur             % M_cur was fully finished
                M_start = M_start + 1;  ig_resume = 1;
            end
            fprintf('Resuming from checkpoint %s: %d sectors done; continue at M=%d, irrep %d/%d.\n', ...
                ckpt_path, numel(sector_M), M_start, ig_resume, numel(irreps));
        else
            warning('ftlm_observables_pg_gpu_Ih:ckptMismatch', ...
                'Checkpoint config mismatch -- ignoring %s and starting fresh.', ckpt_path);
        end
    end

    %% Lightweight ALWAYS-ON performance accounting (for the paper's runtime
    %  tables): phase timers + a VRAM-usage high-water mark sampled at phase
    %  boundaries (AvailableMemory queries are driver calls, no device sync,
    %  ~us each) + the exact host peak RSS (Linux VmHWM) at the end. Saved
    %  as `perf` in the results .mat and printed as a summary block.
    perf = struct('t_enumerate', 0, 't_collect', 0, 't_skeleton', 0, ...
                  't_gpu_sectors', 0, 't_ed_sectors', 0, 't_total', NaN, ...
                  'sector_t', zeros(0, 1), 'vram_used_peak_gb', 0, ...
                  'vram_total_gb', NaN, 'host_peak_gb', NaN, ...
                  'gpu_name', '', 'R', R, 'M_lz', M_lz);
    if have_kernel
        try
            perf.vram_total_gb = double(gpu_h.TotalMemory) / 1e9;
            perf.gpu_name      = gpu_h.Name;
        catch
        end
    end

    t_start = tic;
    if mem_diag
        fprintf('\n=== Memory diagnostics ENABLED ===\n');
        mem_snapshot('baseline before any sector', gpu_h);
    end

    for M = M_start : M_max
        if ~ismember(M, M_run), continue; end
        mult_M = 1 + (M > 0);

        % Spin-flip Z2: P preserves only the M = 0 sector, so the extended
        % group + doubled (Gamma+-) irrep list apply there alone; every
        % M > 0 sector runs with the base group exactly as before.
        if use_spin_flip && M == 0
            group_M = group_sf;  irreps_M = irreps_sf;
        else
            group_M = group;     irreps_M = irreps;
        end

        % Irrep-INDEPENDENT precompute: M-sector enumeration + min_image
        % + stabiliser lists. Runs once per M and is reused for all 10
        % irreps below.
        %
        % GPU-accelerated path: ENUMERATE_M_ORBITS_IH_GPU pushes the
        % M-filter (Step 1) into gpuArray chunks and replaces the 120
        % per-g APPLY_PERM_TO_STATE calls in the stabiliser pass with
        % one BLAS matmul. Falls back to the CPU path internally if no
        % CUDA device is present, so the contract is unchanged.
        % To revert, swap the call below for the CPU original:
        %   cache_M = enumerate_M_orbits_Ih(s_val, M, group);
        if mem_diag, fprintf('\n--- M = %d ---\n', M); mem_snapshot('M-loop start', gpu_h); end

        % ===== Precompute cache (opt-in `precompute_cache`) =====
        % cache_M (enumerate) + entries_M (collect) depend ONLY on
        % (geometry, s, M) -- NOT on R / M_lz / irrep. Persist them after the
        % first compute so a resume / rerun of the SAME system skips the
        % (minutes-long) enumerate + collect, loading in ~seconds instead. The
        % stamp carries bond + perm checksums, so any geometry or provider-code
        % change invalidates the cache (recompute). KEPT on disk for reuse (NOT
        % deleted on completion, unlike the checkpoint). Disabled only under
        % gpu_native collect (gpuArray entries); WITH entries_on_disk the cache
        % stores the lightweight on-disk struct + keeps the sorted entry file
        % as a cache artifact (multi-GPU table sharing, see below).
        od_dir = '';
        pc_loaded = false;  pc_path = '';
        % Spin-flip runs cache under a distinct name (the flip M=0 precompute
        % differs from the base one; the config stamp below also differs via
        % order/perms_chk, but a separate file keeps both reusable side by
        % side). sf_tag also names the on-disk entry directory below.
        if use_spin_flip && M == 0, sf_tag = '_sf'; else, sf_tag = ''; end
        % precompute_cache COMPOSES with entries_on_disk (June 2026 -- the
        % multi-GPU table-SHARING enabler): the cache then stores cache_M plus
        % the LIGHTWEIGHT on-disk entries struct, whose sorted_path points into
        % precompute_dir. Several orchestrated workers load the same cache and
        % memory-map the SAME sorted file (the OS page cache shares the
        % physical pages) instead of re-collecting / holding the table 4x.
        % NB: a concurrent cache-MISS with a shared dir must be serialized
        % (the orchestrator's phase-0 precompute does exactly that) -- the od
        % bucket scratch files have fixed names and would clobber.
        pc_on = precompute_cache && ~strcmpi(entries_storage, 'gpu_native');
        if pc_on
            pc_path = fullfile(precompute_dir, ...
                sprintf('precompute_%s_s%s_M%d%s.mat', geo_tag, s_str, M, sf_tag));
            pc_cfg = struct('geometry', lower(geometry), 'N_sites', N_sites, ...
                'M', M, 's_val', s_val, 'J', J, 'order', group_M.order, ...
                'n_bonds', size(bonds, 1), ...
                'bonds_chk', mod(sum(double(bonds(:)) .* (1:numel(bonds))'), 2^53), ...
                'perms_chk', mod(sum(double(group_M.perms(:)) .* (1:numel(group_M.perms))'), 2^53));
            if exist(pc_path, 'file')
                pc = load(pc_path);
                if isfield(pc, 'pc_cfg') && isequal(pc.pc_cfg, pc_cfg)
                    % The cached entries layout must MATCH this run's
                    % entries_on_disk mode, and a cached on-disk struct must
                    % still find its sorted file (right size) on disk.
                    pc_is_od = isfield(pc.entries_M, 'on_disk') && pc.entries_M.on_disk;
                    od_ok = (pc_is_od == logical(entries_on_disk));
                    if od_ok && pc_is_od
                        bpe_od = 6 + double(isfield(pc.entries_M, 'c_is_indexed') ...
                                            && pc.entries_M.c_is_indexed);
                        dd = dir(pc.entries_M.sorted_path);
                        od_ok = (double(pc.entries_M.n_entries) == 0) || ...
                                (~isempty(dd) && dd.bytes == double(pc.entries_M.n_entries) * bpe_od);
                    end
                    if od_ok
                        cache_M = pc.cache_M;  entries_M = pc.entries_M;  pc_loaded = true;
                        fprintf('  [precompute-cache] loaded %s -> skip enumerate+collect.\n', pc_path);
                    else
                        warning('ftlm_observables_pg_gpu_Ih:pcOdMismatch', ...
                            ['Precompute-cache entries layout/file mismatch ', ...
                             '(entries_on_disk vs cache) -- ignoring %s, recomputing.'], pc_path);
                    end
                else
                    warning('ftlm_observables_pg_gpu_Ih:pcMismatch', ...
                        'Precompute-cache config mismatch -- ignoring %s, recomputing.', pc_path);
                end
                clear pc;
            end
        end

        if ~pc_loaded
            % Irrep-INDEPENDENT precompute: M-sector enumeration + min_image +
            % stabiliser lists (enumerate), then the CLT entry table src/tgt/g/c
            % (collect). Both built once per M and reused for all irreps below.
            t_ph = tic;
            cache_M = enumerate_M_orbits_Ih_gpu(s_val, M, group_M);
            perf.t_enumerate = perf.t_enumerate + toc(t_ph);
            perf = perf_vram_sample(perf, gpu_h, have_kernel);
            if mem_diag, mem_snapshot('after enumerate_M_orbits', gpu_h); end

            % Out-of-core (opt-in): collect streams entries straight to an NVMe
            % file (bucket external sort), so neither the collect peak nor the
            % resident table live in host RAM; build_entry_skeleton then mmaps
            % the sorted file. s=1/2 writes [src][g] (6 B/entry); s>=1 appends
            % the uint8 c-index ([src][g][c_idx], 7 B/entry) and streams it --
            % the dodecahedron-s=3/2 companion-A path (table 73 GB > local host).
            if entries_on_disk
                assert(exist('mmap_file', 'file') == 3, ...
                    'entries_on_disk needs the mmap_file MEX (run build_all).');
                % With the precompute cache the od files are CACHE ARTIFACTS:
                % they live in precompute_dir (shareable across orchestrated
                % workers) and persist with the cache. Without it they are
                % per-run scratch under output_dir, deleted at end of M.
                if pc_on
                    od_base = precompute_dir;          % cache artifact: persists
                elseif ~isempty(ondisk_scratch_dir)
                    od_base = ondisk_scratch_dir;      % node-local scratch (see above)
                else
                    od_base = output_dir;
                end
                od_dir = fullfile(od_base, ...
                    sprintf('ondisk_%s_s%s_M%d%s', geo_tag, s_str, M, sf_tag));
            end
            t_ph = tic;
            switch lower(entries_storage)
                case 'gpu_native'
                    assert(~entries_on_disk, 'entries_on_disk not supported with gpu_native collect.');
                    % (Flip-safe: its only group action is MIN_IMAGE_IH_GPU,
                    % which handles the G x Z2 extension internally.)
                    entries_M = collect_clt_entries_Ih_gpu(cache_M.super_reps, ...
                                                            bonds, s_val, J, group_M);
                otherwise
                    entries_M = collect_clt_entries_Ih(cache_M.super_reps, bonds, ...
                                                        s_val, J, group_M, lookup_method, ...
                                                        entries_storage, od_dir);
            end
            perf.t_collect = perf.t_collect + toc(t_ph);
            perf = perf_vram_sample(perf, gpu_h, have_kernel);
            if mem_diag, mem_snapshot('after collect_clt_entries', gpu_h); end

            % Persist the precompute for future resumes/reruns (atomic temp+rename).
            if pc_on
                cache_M   = pc_gather_struct(cache_M);    % ensure host (no gpuArray)
                entries_M = pc_gather_struct(entries_M);
                % PID-unique temp name: with a SHARED precompute_dir several
                % orchestrated workers can miss the cache simultaneously; a
                % fixed '.tmp' would make them clobber each other's half-
                % written file. (The orchestrator's phase-0 precompute avoids
                % the duplicated work; this guards the write itself.)
                pc_tmp = sprintf('%s.%d.tmp', pc_path, feature('getpid'));
                save(pc_tmp, 'cache_M', 'entries_M', 'pc_cfg', '-v7.3', '-nocompression');
                movefile(pc_tmp, pc_path, 'f');
                fprintf('  [precompute-cache] wrote %s.\n', pc_path);
            end
        end

        % precompute_only (orchestrator phase 0): the cache for this M is
        % written -- skip the irrep loop entirely and move to the next M.
        if precompute_only
            cache_M = [];  entries_M = [];   %#ok<NASGU>
            fprintf('  [precompute-only] M=%d cache done; skipping the irrep loop.\n', M);
            continue;
        end

        % A1-ph2 + D2: build the irrep-INDEPENDENT entry skeleton ONCE per M
        % (src/g/c/diag/entries_per_rep, ~8 GB at s=7/2, uploaded a single
        % time). The per-irrep build_clt_skeleton calls below reuse its
        % gpuArray handles instead of rebuilding + re-uploading them for each
        % of the 10 irreps -> the 10-irrep skeleton phase drops from ~1260 s
        % to ~24 s at s=7/2 M=0.
        mmap_handle = [];
        if have_kernel
            % On-disk entries (entries_M.on_disk) take build_entry_skeleton's
            % out-of-core branch: it mmaps the sorted file (eskel_M.mmap_handle)
            % and builds the tile partition from the per-rep histogram -- no
            % in-RAM entry arrays. Otherwise the normal in-RAM skeleton.
            % R2-lite (siehe Options-Block): (tgt,src)-Re-Sort der Entries vor
            % dem Skeleton-Build. uint64-Schluessel tgt*2^32+src ist exakt und
            % ordnungserhaltend (beide Indizes < 2^31).
            if esort_src && ~entries_on_disk && isfield(entries_M, 'src_sorted') ...
                    && ~isa(entries_M.src_sorted, 'gpuArray')
                t_es = tic;
                es_key = uint64(entries_M.tgt_sorted) * uint64(2^32) ...
                       + uint64(entries_M.src_sorted);
                [~, p_es] = sort(es_key);  clear es_key;
                entries_M.src_sorted = entries_M.src_sorted(p_es);
                entries_M.tgt_sorted = entries_M.tgt_sorted(p_es);
                entries_M.g_sorted   = entries_M.g_sorted(p_es);
                if isfield(entries_M, 'c_idx') && numel(entries_M.c_idx) == numel(p_es)
                    entries_M.c_idx = entries_M.c_idx(p_es);
                end
                if isfield(entries_M, 'c_sorted') && numel(entries_M.c_sorted) == numel(p_es)
                    entries_M.c_sorted = entries_M.c_sorted(p_es);
                end
                clear p_es;
                fprintf('    [esort] Entries (tgt,src)-sortiert in %.1f s (R2-lite).\n', toc(t_es));
            end
            t_ph = tic;
            eskel_M = build_entry_skeleton_Ih(entries_M);
            perf.t_skeleton = perf.t_skeleton + toc(t_ph);
            perf = perf_vram_sample(perf, gpu_h, have_kernel);
            if entries_on_disk
                mmap_handle = eskel_M.mmap_handle;     % unmap at end of M
                % Abort guard (error/Ctrl-C mid-M): drop the kernel's borrowed
                % host pointer and close the mapping even on abnormal exit.
                % Without it the mapped file stays locked (Windows: the
                % ondisk_* dir cannot be deleted until MATLAB restarts) and
                % the kernel keeps its raw device buffers until the next
                % init. The normal path releases via `clear mmap_guard` at
                % the end of M, so the release runs exactly once; both steps
                % tolerate already-released state.
                mmap_guard = onCleanup(@() release_mmap_quiet(mmap_handle)); %#ok<NASGU>
                if mem_diag, fprintf('    [out-of-core] entries collected+sorted on disk + mmapped (%s)\n', entries_M.sorted_path); end
            else
                % The per-irrep build_clt_skeleton reads ONLY entries.super_reps
                % once an eskel is passed (verified: single entries.* consumer),
                % and the precompute cache -- if any -- was saved ABOVE. So the
                % big per-entry host arrays are dead weight from here on:
                % build_entry_skeleton uploaded src/g/c_idx to gpuArrays (or
                % holds copy-on-write references in eskel on the B2/packed
                % variants, which keep the data alive through eskel). Freeing
                % them here returns ~7 B/entry of host RAM for the WHOLE irrep
                % loop (~12 GB at s=7/2 icosahedron M=0, ~7 GB at N=36).
                entries_M.tgt_sorted = [];
                entries_M.src_sorted = [];
                if isfield(entries_M, 'g_sorted'),  entries_M.g_sorted  = []; end
                if isfield(entries_M, 'c_idx'),     entries_M.c_idx     = []; end
                if isfield(entries_M, 'c_sorted'),  entries_M.c_sorted  = []; end
                if isfield(entries_M, 'diag_vals'), entries_M.diag_vals = []; end
            end
        else
            eskel_M = [];
        end

        if M == M_start, ig_lo = ig_resume; else, ig_lo = 1; end
        % Irrep ORDERING (K7d analysis, 2026-07): irreps run in TABLE order,
        % i.e. ASCENDING d (A_g, A_u, T1g, ..., H_g, H_u). A speculative
        % largest-first reorder was chased and REVERTED on 2026-06-06
        %; any stale comment claiming
        % largest-first is wrong. Consequence under keep_table + streamed
        % tables: the kernel sizes its resident prefix ONLY on the FIRST
        % (fresh, i.e. smallest-d) init of an M sector; a later, larger irrep
        % whose Lanczos buffers no longer fit triggers the kernel self-heal
        % which DROPS the prefix -- without a kernel-side re-grow that drop
        % was permanent for all remaining irreps of the M sector (prod_m0
        % log, 2026-07). Mitigations live elsewhere: run_ftlm clamps B so a
        % fully-fitting table stays resident (K1c) and the kernel re-sizes
        % the prefix after a drop (K1b/K7d). A driver-side descending-d
        % reorder would also fix it structurally but PERMUTES the all_E/all_w
        % concatenation (sums in compute_observables can differ in the last
        % ulp) and changes checkpoint positions -- if ever done, keep the
        % seed bound to the CANONICAL ig (8e6+M*1e4+ig*100), store the
        % permuted position in the checkpoint, and stamp ckpt_cfg with
        % 'irrep_order'; gate separately from the bit-identical kernel fix.
        for ig = ig_lo : numel(irreps_M)
            ir = irreps_M{ig};

            [reps, V_per_rep, eig_per_rep, n_per_rep, ~, triv_active] = ...
                apply_irrep_to_orbits(cache_M, ir.data, ir.d, group_M);
            % Sum in DOUBLE: n_per_rep is int32 and MATLAB's native integer
            % sum SATURATES at 2^31-1 -- exactly the n_basis > 2^31 blocks
            % the 64-bit ABI targets (poisons sector_dims/logs, not physics).
            n_basis = sum(double(n_per_rep));
            if n_basis == 0
                continue;
            end

            sector_M(end+1, 1)    = M;             %#ok<AGROW>
            sector_G(end+1, 1)    = ig;            %#ok<AGROW>
            sector_dims(end+1, 1) = n_basis;       %#ok<AGROW>

            t_sec = tic;
            if n_basis <= ed_thresh
                %% --- Dense ED branch ---
                H_block = build_heisenberg_sparse_Ih_gamma2(reps, V_per_rep, ...
                    eig_per_rep, n_per_rep, bonds, s_val, J, ir.data, ir.d, group_M);
                Hd = full(H_block);
                Hd = 0.5 * (Hd + Hd');
                E_sec = sort(real(eig(Hd)));
                w_sec = ones(numel(E_sec), 1);
                path = 'ED';
            elseif have_kernel
                %% --- GPU FP32 block-Lanczos with precomputed M tensor ---
                %  Uses BUILD_CLT_PG_IH (pagemtimes-batched M_e on host).
                %  The wrapper auto-dispatches to the legacy 'init' mode
                %  with the precomputed M tensor, which on the RTX 4000
                %  Ada empirically beats the on-the-fly SpMV path (the
                %  d^4 inner inflates register pressure / Lanczos time
                %  on d >= 4 irreps).
                %  To switch back to the on-the-fly path, replace this
                %  call with BUILD_CLT_SKELETON_PG_IH (which leaves
                %  clt.is_skeleton = true and triggers init_skel).
                %
                %  PRODUCTION SCALABILITY PATH (May 2026 rewrite):
                %  BUILD_CLT_SKELETON_FROM_ENTRIES_IH packages only the
                %  small per-rep / per-group data (V, sqrt_eig, rho)
                %  and the per-entry quadruple (src, tgt, g, c_a). The
                %  d x d x n_entries M tensor is NEVER materialised --
                %  it is recomputed on-the-fly inside the CUDA SpMV
                %  kernel from V_t, V_r, rho_Gamma(g), and the spin
                %  coefficient.
                %
                %  GPU peak for H_g at s = 1/2 icosidodecahedron drops
                %  from ~ 16 GB (precomputed-M) to ~ 2.4 GB (skeleton).
                %  Host peak during per-irrep build: a few hundred MB.
                %  This is the only path that scales beyond N = 30.
                %
                %  The companion CUDA OTF kernel was rewritten with
                %  per-(rep, k') thread parallelisation to eliminate
                %  the register-spilling that crippled the earlier
                %  one-thread-per-rep design for d >= 4.
                %
                %  Precomputed-M fallback (small systems only,
                %  do NOT use beyond N ~ 30):
                %    clt = build_clt_from_entries_Ih_streamed(entries_M, reps, ...
                %        V_per_rep, eig_per_rep, n_per_rep, ir.data, ir.d, ...
                %        group, gpu_h);
                if mem_diag
                    fprintf('  > %s sector M=%d %s (d=%d)\n', ir.name, M, ir.name, ir.d);
                    mem_snapshot('before build_clt_skel', gpu_h);
                end
                clt = build_clt_skeleton_from_entries_Ih(entries_M, reps, ...
                    V_per_rep, eig_per_rep, n_per_rep, ir.data, ir.d, group_M, ...
                    triv_active, eskel_M, compact_v_flag, force_complex);
                if entries_on_disk
                    clt.force_stream = true;    % stream from the mmap'd file
                    % AUTO resident-prefix, set EXPLICITLY (explicit is more
                    % robust than a default). Historical note: force_stream
                    % ALONE used to map to prefix OFF in run_ftlm, and without
                    % this line every production out-of-core run re-streamed
                    % the FULL entry table (~86 GB at dodec s=3/2) from disk /
                    % page cache on EVERY SpMV. Since K2e run_ftlm defaults a
                    % missing prefix_budget to AUTO (-1) even under
                    % force_stream; streaming-coverage tests opt out with an
                    % explicit prefix_budget = 0. AUTO (-1) lets the kernel pin
                    % as many leading tiles as the VRAM left after its Lanczos
                    % buffers allows and stream only the tail; on a card that
                    % fits the whole table it becomes fully resident. Bit-
                    % identical either way (test_stream_prefix gates it).
                    clt.prefix_budget = -1;
                    if ~isempty(prefix_budget_gb)
                        % Experiment knob (see option block at the top):
                        % explicit prefix cap in bytes instead of AUTO.
                        clt.prefix_budget = round(prefix_budget_gb * 1e9);
                        fprintf(['    [prefix-cap] resident prefix capped at ' ...
                                 '%.1f GB (streamed-tail emulation)\n'], ...
                                prefix_budget_gb);
                    end
                    % Keep-table: all irreps of this M stream the SAME sorted
                    % on-disk table -- have run_ftlm's end-of-sector cleanup
                    % preserve the kernel's table state (incl. the resident
                    % prefix) so the next irrep skips the ~full-table disk
                    % re-read. The end-of-M release_mmap_quiet still performs
                    % the FULL kernel cleanup before unmapping.
                    clt.keep_table = true;
                end
                % R2-v1 (tiled_spmv): CLT-Entries in (src_tile, tgt)-Ordnung
                % permutieren + Run-CSR anhaengen. Nur resident-ref (real,
                % kein od/B2) -- sonst unveraendert weiterreichen.
                if tiled_spmv && isfield(clt, 'is_real') && clt.is_real ...
                        && ~entries_on_disk ...
                        && ~(isfield(clt, 'is_b2') && clt.is_b2)
                    t_tl = tic;
                    ro_t  = [int64(0); cumsum(int64(gather(clt.n_per_rep)))];
                    epr_t = double(gather(clt.entries_per_rep));
                    ct_t  = struct();
                    ct_t.tgt = repelem((1:numel(epr_t))', epr_t);
                    if isfield(clt, 'srcg') && ~isempty(clt.srcg)
                        ct_t.src = double(bitand(gather(clt.srcg), uint32(2^25 - 1)));
                    else
                        ct_t.src = double(gather(clt.src_idx));
                    end
                    if isempty(ct_t.src)
                        % Diagonal-only-Sektor: nichts zu tilen -> Standardpfad
                        % (clt bekommt kein run_ptr-Feld).
                        clear ct_t ro_t epr_t;
                    else
                        Bw  = max(1, B_gpu);
                        tl_t = build_tiled_entries(ct_t, ro_t, Bw, tiled_window_gb * 1e9);
                        clear ct_t;
                        if isfield(clt, 'srcg') && ~isempty(clt.srcg)
                            clt.srcg = clt.srcg(tl_t.perm);
                        end
                        if ~isempty(clt.src_idx), clt.src_idx = clt.src_idx(tl_t.perm); end
                        if isfield(clt, 'g_idx') && ~isempty(clt.g_idx)
                            clt.g_idx = clt.g_idx(tl_t.perm);
                        end
                        if isfield(clt, 'c_a') && numel(clt.c_a) == numel(tl_t.perm)
                            clt.c_a = clt.c_a(tl_t.perm);
                        end
                        if isfield(clt, 'c_idx') && numel(clt.c_idx) == numel(tl_t.perm)
                            clt.c_idx = clt.c_idx(tl_t.perm);
                        end
                        clt.run_ptr      = gpuArray(int64(tl_t.run_ptr));
                        clt.run_tgt      = gpuArray(int32(tl_t.run_tgt));
                        clt.tile_run_ptr = tl_t.tile_run_ptr;
                        fprintf('    [R2] tiled: %d Tiles, %d Runs (Fenster <= %.1f MB, %.1f s)\n', ...
                                tl_t.n_tiles, tl_t.n_runs, tiled_window_gb * 1e3, toc(t_tl));
                        clear tl_t;
                    end
                end
                if mem_diag, mem_snapshot('after build_clt_skel (host)', gpu_h); end
                seed = 8e6 + M * 1e4 + ig * 100;
                [E_sec, w_sec, ~, v_pk] = run_ftlm_pg_sector_gpu_Ih( ...
                    clt, R, M_lz, B_gpu, seed, gpu_h, mem_diag, lz_diag);
                if isfinite(v_pk)
                    perf.vram_used_peak_gb = max(perf.vram_used_peak_gb, v_pk);
                end
                path = 'GPU(FP32)';

                % CRITICAL: force-release the CLT gpuArrays NOW.
                % Without this, MATLAB's lazy GC keeps the per-sector M
                % buffers resident across irreps, causing VRAM to climb
                % stair-step until OOM. Setting fields to [] drops the
                % references; wait(gpu_h) flushes any pending free.
                clt = [];                                 %#ok<NASGU>
                wait(gpu_h);
                if mem_diag, mem_snapshot('after clt=[]+wait', gpu_h); end

                % Also release the per-irrep host-side arrays from
                % apply_irrep_to_orbits. These cell arrays of single-
                % precision V / lambda matrices are small individually
                % but accumulate via MATLAB's reference behavior;
                % explicit clear here is a safe (post-kernel) no-op
                % if there was no leak, and a real saving if there is.
                clear reps V_per_rep eig_per_rep n_per_rep;
                if mem_diag, mem_snapshot('after sector locals clear', gpu_h); end
            else
                %% --- CPU FP64 fallback ---
                H_block = build_heisenberg_sparse_Ih_gamma2(reps, V_per_rep, ...
                    eig_per_rep, n_per_rep, bonds, s_val, J, ir.data, ir.d, group_M);
                H_block = 0.5 * (H_block + H_block');
                is_real_block = isreal(H_block);
                if is_real_block
                    H_block = real(H_block);
                end
                seed = 7e6 + M * 1e4 + ig * 100;
                [E_sec, w_sec] = run_ftlm_pg_sector(H_block, n_basis, R, ...
                                                    M_lz, is_real_block, seed);
                path = 'CPU(FP64)';
            end

            % Apply mult_M (M <-> -M) and d_Gamma (partner-row degeneracy)
            w_sec = w_sec * (mult_M * ir.d);

            all_E = [all_E; E_sec];                       %#ok<AGROW>
            all_w = [all_w; w_sec];                       %#ok<AGROW>
            all_M = [all_M; M * ones(numel(E_sec), 1)];   %#ok<AGROW>
            sector_path{end+1, 1} = path;                  %#ok<AGROW>

            t_sector = toc(t_sec);
            perf.sector_t(end+1, 1) = t_sector;           %#ok<AGROW>
            if strncmp(path, 'GPU', 3)
                perf.t_gpu_sectors = perf.t_gpu_sectors + t_sector;
            else
                perf.t_ed_sectors  = perf.t_ed_sectors  + t_sector;
            end
            fprintf('  M=%2d %-4s (d=%d) n_basis=%4d full_block=%4d %-12s t=%.2fs\n', ...
                M, ir.name, ir.d, n_basis, n_basis * ir.d, path, t_sector);

            % Per-IRREP checkpoint: atomic write (temp + rename) after each
            % completed (M, ig) sector so a crash/reboot costs at most one irrep.
            if checkpoint
                cfg = ckpt_cfg;  M_cur = M;  ig_done = ig;   %#ok<NASGU>
                ckpt_tmp = strrep(ckpt_path, '.mat', '.tmp.mat');
                save(ckpt_tmp, 'cfg', 'M_cur', 'ig_done', 'all_E', 'all_w', 'all_M', ...
                    'sector_M', 'sector_G', 'sector_dims', 'sector_path', ...
                    '-v7.3', '-nocompression');
                movefile(ckpt_tmp, ckpt_path, 'f');
            end
        end

        % Release the per-M shared entry skeleton before the next M sector.
        eskel_M = []; %#ok<NASGU>
        % Out-of-core: unmap this M sector's mapping. The on-disk entry files
        % are deleted ONLY when they are per-run scratch (no precompute cache);
        % under pc_on they persist with the cache for reuse / worker sharing.
        if entries_on_disk && ~isempty(mmap_handle)
            clear mmap_guard;   % fires release_mmap_quiet: kernel cleanup + unmap
            if ~pc_on && ~isempty(od_dir) && exist(od_dir, 'dir'), rmdir(od_dir, 's'); end
            mmap_handle = [];   %#ok<NASGU>
        end
        if have_kernel, wait(gpu_h); end
    end
    t_wall = toc(t_start);
    fprintf('\nTotal wall time: %.2f s\n', t_wall);
    info.t_wall = t_wall;

    %% Performance summary (always on; for the paper's runtime tables).
    perf.t_total = t_wall;
    perf.host_peak_gb = host_peak_rss_gb();
    t_other = max(0, t_wall - perf.t_enumerate - perf.t_collect ...
                       - perf.t_skeleton - perf.t_gpu_sectors - perf.t_ed_sectors);
    fprintf('=== performance summary ===\n');
    fprintf(['  enumerate %.1f s | collect %.1f s | skeleton %.1f s | ', ...
             'FTLM(GPU) %.1f s | ED %.1f s | other %.1f s\n'], ...
            perf.t_enumerate, perf.t_collect, perf.t_skeleton, ...
            perf.t_gpu_sectors, perf.t_ed_sectors, t_other);
    if isfinite(perf.vram_total_gb)
        fprintf('  peak VRAM used ~%.1f GB (of %.1f GB, %s)', ...
                perf.vram_used_peak_gb, perf.vram_total_gb, perf.gpu_name);
    else
        fprintf('  peak VRAM n/a');
    end
    if isfinite(perf.host_peak_gb)
        fprintf(' | host peak RSS %.1f GB\n', perf.host_peak_gb);
    else
        fprintf(' | host peak RSS n/a (exact only on Linux/VmHWM)\n');
    end
    info.perf = perf;

    %% precompute_only: the caches are on disk; nothing to aggregate or save.
    if precompute_only
        fprintf('precompute_only: caches written to %s; no FTLM run.\n', precompute_dir);
        return;
    end

    %% Observables
    [C_T, chi_T, Z_eff] = compute_observables_pg(all_E, all_w, all_M, T_range);

    %% Sum-rule check. The FTLM weights (with the mult_M * d_Gamma factor
    %  already folded into all_w) sum to the dimension of the part of the
    %  Hilbert space actually covered: dim(M=0) for an only_M0 run, the full
    %  n_total for a complete M sweep. Compare against that analytic value so
    %  the check is meaningful in BOTH cases. (It used to always compare
    %  against n_total, which gave a spurious rel.err ~ 0.86 for only_M0
    %  runs even though sum_i w_i == dim(M=0) exactly.)
    if is_full_sweep
        Z_expected = n_total;                                  % full M sweep
        chk_label  = 'full dim';
    else
        A_M0 = round(N_sites * s_val);                         % digit sum at M=0
        [D_chk, ~] = build_D_table(N_sites, round(2*s_val), A_M0 + M_max);
        Z_expected = 0;
        for Mc = M_run
            Z_expected = Z_expected + (1 + (Mc > 0)) * ...
                double(D_chk(N_sites + 1, A_M0 + Mc + 1));     % mult_M * dim(M)
        end
        chk_label  = sprintf('dim(M in {%s})', num2str(M_run));
    end
    Z_inf = sum(all_w);
    fprintf('Sum-rule check: sum_i w_i = %.6g, %s = %.6g, rel.err = %.2e\n', ...
        Z_inf, chk_label, Z_expected, abs(Z_inf - Z_expected) / max(Z_expected, 1));

    %% Save results
    % Geometry tag (set in the geometry switch above) in the filename keeps
    % different systems from overwriting each other in the same output_dir.
    % (mat_name built once, before the M loop -- also part of the info struct.)
    mat_path = fullfile(output_dir, mat_name);
    info.mat_path = mat_path;
    n_total_save = n_total;     %#ok<NASGU>
    % K7c provenance: record whether the run's Lanczos numerics were fp16
    % (env-driven, so otherwise invisible in the result .mat).
    fp16_used = strcmp(getenv('FTLM_FP16'), '1');     %#ok<NASGU>
    save(mat_path, ...
        'T_range', 'C_T', 'chi_T', 'Z_eff', ...
        's_val', 'J', 'R', 'M_lz', 'M_max', 'n_total_save', ...
        'only_M0', 'irrep_list', 'ed_thresh', 'B_gpu', 'fp16_used', ...
        'sector_M', 'sector_G', 'sector_dims', 'sector_path', ...
        'all_E', 'all_w', 'all_M', ...
        't_wall', 'perf', '-v7.3');
    fprintf('Results saved to: %s\n', mat_path);

    % Run completed successfully -> drop the resume checkpoint.
    if checkpoint && exist(ckpt_path, 'file')
        delete(ckpt_path);
        fprintf('Checkpoint %s removed (run complete).\n', ckpt_path);
    end
end


% ----------------------------------------------------------------
function full = build_full_irrep_table(group)
%BUILD_FULL_IRREP_TABLE  All irreps as {name, d, data}, group-generically.
%   data is the form IRREP_MATRIX expects: an [order x 1] character vector
%   for d == 1, else a [d x d x order] matrix stack.
    if isfield(group, 'irreps')
        n    = numel(group.irreps);
        full = cell(1, n);
        for p = 1 : n
            irp = group.irreps(p);
            if irp.d == 1
                data = reshape(irp.mats, [], 1);     % [order x 1]
            else
                data = irp.mats;                     % [d x d x order]
            end
            full{p} = struct('name', irp.name, 'd', irp.d, 'data', data);
        end
    else
        full = {};
        full{end+1} = struct('name', 'A_g', 'd', 1, 'data', group.Ag);
        full{end+1} = struct('name', 'A_u', 'd', 1, 'data', group.Au);
        full{end+1} = struct('name', 'T1g', 'd', 3, 'data', group.T1g);
        full{end+1} = struct('name', 'T1u', 'd', 3, 'data', group.T1u);
        full{end+1} = struct('name', 'T2g', 'd', 3, 'data', group.T2g);
        full{end+1} = struct('name', 'T2u', 'd', 3, 'data', group.T2u);
        full{end+1} = struct('name', 'F_g', 'd', 4, 'data', group.Fg);
        full{end+1} = struct('name', 'F_u', 'd', 4, 'data', group.Fu);
        full{end+1} = struct('name', 'H_g', 'd', 5, 'data', group.Hg);
        full{end+1} = struct('name', 'H_u', 'd', 5, 'data', group.Hu);
    end
end

function irreps = filter_irreps(full, names_keep)
    known = cellfun(@(c) c.name, full, 'uni', 0);
    % A misspelled entry used to be dropped SILENTLY (the run continued with
    % the matching subset) -- error per unknown name instead.
    unknown = names_keep(~ismember(names_keep, known));
    if ~isempty(unknown)
        error('ftlm_observables_pg_gpu_Ih:no_irreps', ...
              'Unknown irrep name(s) in irrep_list: %s. Known: %s.', ...
              strjoin(cellstr(string(unknown)), ', '), strjoin(known, ', '));
    end
    irreps = full(ismember(known, names_keep));
end

function release_mmap_quiet(h)
%RELEASE_MMAP_QUIET  Abort-safe unmap (onCleanup target for the od path):
%   drop the kernel's borrowed host pointer first, then close the mapping.
%   Both steps tolerate already-released state (kernel 'cleanup' NULLs all
%   statics; a second close on the same handle is caught), so it is safe as
%   BOTH the normal end-of-M release and the error/Ctrl-C destructor.
    try, cuda_lanczos_clut_block_pg_Ih('cleanup'); catch, end
    try, if ~isempty(h), mmap_file('close', h); end, catch, end
end

function perf = perf_vram_sample(perf, gpu_h, have_kernel)
%PERF_VRAM_SAMPLE  Update the VRAM high-water mark (driver query, no sync).
    if ~have_kernel || ~isfinite(perf.vram_total_gb), return; end
    try
        used = perf.vram_total_gb - double(gpu_h.AvailableMemory) / 1e9;
        perf.vram_used_peak_gb = max(perf.vram_used_peak_gb, used);
    catch
    end
end

function gb = host_peak_rss_gb()
%HOST_PEAK_RSS_GB  Peak resident set of THIS process. EXACT on Linux (VmHWM,
%   kernel-tracked high-water mark, read once at the end -- zero runtime
%   overhead); NaN on other platforms (Windows has no built-in peak query).
    gb = NaN;
    try
        txt = fileread('/proc/self/status');
        tok = regexp(txt, 'VmHWM:\s*(\d+)\s*kB', 'tokens', 'once');
        if ~isempty(tok), gb = str2double(tok{1}) * 1024 / 1e9; end
    catch
    end
end
