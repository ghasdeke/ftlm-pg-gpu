function est = estimate_feasibility(group, bonds, s_val, R, M_lz, verbose, use_spin_flip, force_complex)
%ESTIMATE_FEASIBILITY  Pre-flight resource estimate for an M=0 GPU-FTLM run,
%   computed from the provider (group + bonds) WITHOUT running the ~minutes-long
%   enumerate/collect. Predicts n_reps, n_entries, host RAM, VRAM mode
%   (resident / B2-resident / streaming-B2) and peak VRAM, and flags the hard
%   limits (|G| <= 65535 for the uint16 g-index; max irrep d <= 12 for the kernel
%   MAX_D). Lets the driver warn / fail-fast before a multi-hour thrash.
%
%   EST = ESTIMATE_FEASIBILITY(GROUP, BONDS, S_VAL, R, M_LZ) returns a struct:
%       .N .order .max_d .dim_M0 .n_reps .n_entries
%       .host_GB .vram_GB .vram_mode .table_GB
%       .g_ok .d_ok .host_ok .feasible   (logical)
%       .notes  (cellstr of warnings)
%
%   The n_reps/n_entries/host/VRAM numbers are ORDER-OF-MAGNITUDE estimates
%   calibrated against the measured icosido N=30, kagome N=36 and triangular
%   N=36 M=0 runs (free-orbit n_reps ~ dim(M=0)/|G|; ~0.5 flippable bonds/rep;
%   ~24 B/entry host peak during collect; resident entry table src(int32)+
%   g(uint16) = 6-7 B/entry, s >= 1 adds the uint8 c-index; Lanczos ~5 fp32
%   buffers of n_basis at B=1 on the REAL kernel path -- today's default
%   with realified irreps -- or ~9 with FORCE_COMPLEX=true, mirroring
%   BUF_FACTOR in RUN_FTLM_PG_SECTOR_GPU_IH).
%   They are meant for a go/no-go gate, not a precise forecast.
%
%   See also FTLM_OBSERVABLES_PG_GPU_IH, ESTIMATE_SQUARE_FEASIBILITY.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    if nargin < 6, verbose = true; end
    if nargin < 7, use_spin_flip = false; end
    if nargin < 8, force_complex = false; end
    N      = double(group.N);
    order  = double(group.order);
    % Spin-flip Z2 (use_spin_flip in the driver): at M=0 the group doubles
    % to G x Z2, so n_reps / n_entries / n_basis all roughly HALVE (the
    % free-orbit estimates below pick this up through `order`). Irrep
    % dimensions are unchanged (each Gamma splits into Gamma+-).
    if use_spin_flip, order = 2 * order; end
    d_loc  = round(2*s_val + 1);
    n_bonds = size(bonds, 1);

    %% max irrep dimension (named I_h fields OR the generic .irreps list).
    if isfield(group, 'irreps')
        max_d = max(arrayfun(@(s) s.d, group.irreps));
    elseif isfield(group, 'Hg')
        max_d = 5;                                   % I_h: {1,1,3,3,3,3,4,4,5,5}
    else
        max_d = 1;
    end

    %% dim(M=0): #length-N strings over digits 0..d_loc-1 with digit-sum = N*s
    %  (m_i = digit_i - s, sum m_i = 0). DP convolution of N uniform digits.
    target = N * (d_loc - 1) / 2;                    % = N*s
    if mod(N * (d_loc - 1), 2) ~= 0
        dim_M0 = NaN;                                % no exact M=0 (odd N*(2s))
    else
        dp = 1;
        for k = 1 : N, dp = conv(dp, ones(1, d_loc)); end
        dim_M0 = dp(target + 1);
    end

    %% Calibrated estimates.
    n_reps    = dim_M0 / order;                      % free-orbit approximation
    n_entries = n_reps * n_bonds * 0.5;              % ~half the bonds flippable at M=0
    n_basis   = n_reps * max_d;                      % largest (max-d) irrep block

    HOST_B_PER_ENTRY = 24;                           % in-RAM collect-peak calibration
    host_GB   = n_entries * HOST_B_PER_ENTRY / 1e9;

    % entries_on_disk collect FLOOR: the entry table streams to disk, but the
    % int8 digit matrix (N B/rep), the per-rep arrays + per-(bond,sign)
    % transients (~64 B/rep) and the EBS finalize bucket (~23 B/entry per
    % bucket) stay in RAM. This is what remains after the 2026-07 mi8 fix;
    % a blanket "od bounds the collect" bypass OOMed the dodec s=3/2 run at
    % the SLURM cgroup cap. The bucket count MIRRORS collect_clt_entries_Ih
    % (projected-entry / 2.5-GB target + 192-bucket cap): when the 192 cap
    % binds -- wave-B scale -- the bucket grows back past 2.5 GB, which is
    % exactly what this floor must see. The old reps-only rule
    % (ceil(n_reps/1e7)) modelled ~8-GB buckets there and over-estimated the
    % od floor by ~5-13 GB (spurious driver abort at tight sbatch --mem).
    % EBS_BUCKET_COUNT is the shared single source with the collect.
    proj_e_fb    = 2 * n_reps * n_bonds;
    n_bkt_fb     = ebs_bucket_count(n_reps, proj_e_fb);
    ebs_bucket_e = n_entries / n_bkt_fb;
    host_od_GB   = (n_reps * (N + 64) + ebs_bucket_e * 23) / 1e9 + 2;

    % Resident entry table: src(int32)+g(uint16) = 6 B/entry; s >= 1 systems
    % (d_loc > 2) carry the uint8 c-index on top (7 B/entry -- the flat 6
    % under-estimated dodec s=3/2 by ~17%, 74 vs 86 GB, and could flip the
    % resident-B2 vs streaming classification on border cards).
    bpe_est   = 6 + double(d_loc > 2);
    table_B   = n_entries * bpe_est;
    table_GB  = table_B / 1e9;
    % Lanczos buffer count mirrors the driver: the REAL FP32 path (realified
    % irreps, today's default) holds 4 re Krylov buffers + the V0 RNG draw = 5
    % fp32 n_basis buffers; force_complex (legacy baselines) needs 8 + 1 = 9.
    % Using 9 unconditionally over-reported the dodec d=5 Lanczos term ~1.8x
    % and could mispredict resident vs streaming-B2.
    lz_bufs = 5; if force_complex, lz_bufs = 9; end
    lanczos_B = n_basis * lz_bufs * 4;               % fp32 buffers at B=1

    %% Hardware envelope: query the ACTUAL device and host instead of assuming
    %  the dev box. AvailableMemory (not TotalMemory) so the CUDA context /
    %  driver reserve is already subtracted. NB: gpuDevice SELECTS device 1
    %  when none is selected yet -- harmless here (the GPU driver has already
    %  selected its device before the pre-flight; standalone/bench callers get
    %  device 1, which is what they would use anyway). The host budget comes
    %  from the shared detector HOST_AVAIL_GB (memory()/MemAvailable capped by
    %  this process's cgroup limit and the SLURM allocation -- inside a SLURM
    %  step the cgroup is the BINDING limit, not the node total: the 2026-07
    %  dodecahedron run passed host_ok=1 against the node and was then
    %  OOM-killed by slurmstepd at the much smaller cgroup cap). The old
    %  constants remain as conservative fallbacks (RTX 4000 SFF Ada / 63 GB
    %  dev box).
    VRAM_TOTAL_GB = 20.5; HOST_TOTAL_GB = 63;
    vb = gpu_free_bytes();                 % honours FTLM_FAKE_FREE_VRAM_GB
    if isfinite(vb), VRAM_TOTAL_GB = vb / 1e9; end
    [hb_gb, node_gb] = host_avail_gb();    % behaviour-identical extraction (2026-07 audit)
    if isfinite(node_gb), HOST_TOTAL_GB = node_gb; end
    if isfinite(hb_gb) && hb_gb < HOST_TOTAL_GB
        if verbose
            fprintf(['  ! host budget capped by the job allocation: %.0f GB ', ...
                     '(node MemAvailable %.0f GB) -- SLURM: request --mem >= ', ...
                     'the estimated host peak.\n'], hb_gb, HOST_TOTAL_GB);
        end
        HOST_TOTAL_GB = hb_gb;
    end

    %% VRAM mode -- mirrors the ACTUAL gates: build_entry_skeleton_Ih's is_b2
    %  rule (n_entries > 2e9 OR projected table > 0.5 x free VRAM) and
    %  run_ftlm's resident-B2 vs streaming decision. per_rep_B covers the
    %  resident per-rep gpuArrays (diag/offsets/counts/triv/v_slot ~33 B/rep)
    %  the old model omitted (~12 GB at dodec s=3/2).
    per_rep_B = n_reps * 33;
    is_b2_est = (n_entries > 2e9) || (table_B > 0.5 * VRAM_TOTAL_GB * 1e9);
    if ~is_b2_est
        vram_mode = 'resident';          vram_GB = (table_B + lanczos_B + per_rep_B) / 1e9;
    elseif (table_B + lanczos_B + per_rep_B) / 1e9 <= VRAM_TOTAL_GB - 1
        vram_mode = 'resident-B2';       vram_GB = (table_B + lanczos_B + per_rep_B) / 1e9;
    else
        vram_mode = 'streaming-B2';      vram_GB = (1e9 + lanczos_B + per_rep_B) / 1e9;
    end

    %% Hard limits + feasibility flags.
    g_ok    = (order <= 65535);                      % uint16 g-index
    d_ok    = (max_d <= 12);                         % kernel MAX_D (OTF SpMV path)
    host_ok = (host_GB <= HOST_TOTAL_GB);
    host_od_ok = (host_od_GB <= HOST_TOTAL_GB);      % entries_on_disk floor
    % Hard INDEX caps (device-independent; portability audit 2026-07-03):
    % rep indices are int32 in the kernel, and the V0 random draw is ONE
    % gpuArray of n_basis*B elements (2^31-1 elements max even at B=1).
    nreps_ok  = (n_reps  <= 2^31 - 257);
    nbasis_ok = (n_basis <= 2^31 - 1);
    notes = {};
    if ~g_ok,    notes{end+1} = sprintf('|G|=%d exceeds uint16 (65535) g-index cap', order); end
    if ~d_ok,    notes{end+1} = sprintf('max irrep d=%d exceeds kernel MAX_D=12', max_d); end
    if ~nreps_ok,  notes{end+1} = sprintf('n_reps~%.3g exceeds the int32 rep-index cap 2^31-257', n_reps); end
    if ~nbasis_ok, notes{end+1} = sprintf('n_basis~%.3g > 2^31 (max-d block): B=1 + split-V0 path', n_basis); end
    if ~host_ok, notes{end+1} = sprintf('host ~%.0f GB > ~%.0f GB available (collect would thrash)', host_GB, HOST_TOTAL_GB); end
    if strcmp(vram_mode, 'streaming-B2'), notes{end+1} = 'large-d blocks need STREAMING-B2 (slower)'; end
    if host_GB > 0.8 * HOST_TOTAL_GB && host_ok, notes{end+1} = sprintf('host ~%.0f GB is tight (>80%% of %.0f GB)', host_GB, HOST_TOTAL_GB); end
    % nbasis_ok is INFORMATIONAL since the split-V0 path (2026-07): blocks
    % beyond the single-gpuArray cap run at B=1 with chunked V0 uploads.
    feasible = g_ok && d_ok && host_ok && nreps_ok;

    est = struct('N', N, 'order', order, 'max_d', max_d, 'dim_M0', dim_M0, ...
        'n_reps', n_reps, 'n_entries', n_entries, 'n_basis', n_basis, ...
        'host_GB', host_GB, 'host_od_GB', host_od_GB, ...
        'host_budget_GB', HOST_TOTAL_GB, ...
        'vram_GB', vram_GB, 'vram_mode', vram_mode, ...
        'table_GB', table_GB, 'g_ok', g_ok, 'd_ok', d_ok, 'host_ok', host_ok, ...
        'host_od_ok', host_od_ok, 'nreps_ok', nreps_ok, 'nbasis_ok', nbasis_ok, ...
        'feasible', feasible, 'notes', {notes});

    if verbose
        if use_spin_flip, sf_lab = ', spin-flip Z2 ON'; else, sf_lab = ''; end
        fprintf('=== feasibility estimate (M=0, rough%s) ===\n', sf_lab);
        fprintf('  N=%d  |G|=%d  max_d=%d  dim(M=0)=%.3g\n', N, order, max_d, dim_M0);
        fprintf('  n_reps~%.3g  n_entries~%.3g  n_basis(max-d)~%.3g\n', n_reps, n_entries, n_basis);
        fprintf('  host ~%.0f GB in-RAM / ~%.0f GB entries_on_disk floor (budget ~%.0f GB)\n', ...
                host_GB, host_od_GB, HOST_TOTAL_GB);
        fprintf('  VRAM ~%.1f GB [%s] (resident table ~%.1f GB)\n', ...
                vram_GB, vram_mode, table_GB);
        fprintf('  g_ok=%d d_ok=%d host_ok=%d nreps_ok=%d nbasis_ok=%d -> FEASIBLE=%d\n', ...
                g_ok, d_ok, host_ok, nreps_ok, nbasis_ok, feasible);
        for i = 1:numel(notes), fprintf('  ! %s\n', notes{i}); end
    end
end
% cgroup_limit_gb / cg_walkup_min / slurm_mem_gb moved verbatim to the shared
% detector HOST_AVAIL_GB.M (2026-07 audit: EBS_FINALIZE's parallel-worker cap
% needs the same cgroup/SLURM-aware budget).
