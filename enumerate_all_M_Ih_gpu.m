function caches = enumerate_all_M_Ih_gpu(s_val, group, use_gpu)
%ENUMERATE_ALL_M_IH_GPU  Single-pass enumeration of ALL M>=0 sectors.
%
%   CACHES = ENUMERATE_ALL_M_IH_GPU(S_VAL, GROUP)
%   CACHES = ENUMERATE_ALL_M_IH_GPU(S_VAL, GROUP, USE_GPU)
%
%   Returns CACHES{M+1} for M = 0 .. round(N*s), each a struct with the
%   same fields as ENUMERATE_M_ORBITS_IH_GPU (M_target, s_val, super_reps,
%   orbit_lens, stab_flat, stab_ptr), so a full-sweep driver can replace
%   the per-M call
%       cache_M = enumerate_M_orbits_Ih_gpu(s_val, M, group)
%   with one up-front
%       caches  = enumerate_all_M_Ih_gpu(s_val, group);  ...; cache_M = caches{M+1};
%
%   *** HISTORICAL (pre-R3) -- do NOT wire into the driver. The premise
%   below ("the per-M path walks the ENTIRE integer space once per M") was
%   TRUE when this was written but is OBSOLETE since the R3 unranking:
%   stream_super_reps_general in ENUMERATE_M_ORBITS_IH_GPU now generates
%   ONLY the M-sector states via unrank_composition, so a full per-M sweep
%   already visits exactly the M>=0 half once -- the same min_image volume
%   as this routine's single scan. Wiring it today would add all-M cache
%   residency (~105 GB of super_reps at icosahedron s=5) and break the
%   spin-flip M=0 special-casing (different group) for no measurable gain.
%   Kept as a reference implementation only.
%
%   ORIGINAL RATIONALE (obsolete): for s >= 1 the per-M enumeration walked
%   the entire integer space [0, d_loc^N) once per M-sector just to
%   M-filter it; this routine scans once, computes each state's M, keeps
%   the M>=0 half, runs MIN_IMAGE_IH on it, and bins the orbit-minima by M.
%   For s = 1/2 the per-M path always unranked directly (no full scan).
%
%   super_reps stay sorted ascending within each M (chunks ascending, the
%   per-chunk orbit-minima are an ascending subset, binning preserves
%   order) -- as BUILD_LOOKUP_SCHNACK and the bitmap CLT require.
%
%   See also ENUMERATE_M_ORBITS_IH_GPU, MIN_IMAGE_IH, APPLY_IRREP_TO_ORBITS.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    d_loc   = round(2*s_val + 1);
    N_sites = double(group.N);
    n_total = double(d_loc)^N_sites;
    M_max   = round(N_sites * s_val);
    if nargin < 3 || isempty(use_gpu)
        use_gpu = (gpuDeviceCount > 0);
    end

    caches = cell(M_max + 1, 1);

    %% s = 1/2: per-M unranking already avoids the full scan -- just loop.
    if s_val == 0.5
        for M = 0 : M_max
            caches{M+1} = enumerate_M_orbits_Ih_gpu(s_val, M, group);
        end
        return;
    end

    %% s >= 1: ONE scan over [0, n_total), bin orbit-minima by M.
    d_loc_i64 = int64(d_loc);
    n_total_i = int64(n_total);
    N_s       = N_sites * s_val;
    chunk     = int64(16e6);
    if ~use_gpu, chunk = int64(4e6); end

    sr_cells = {};
    Mv_cells = {};
    pos = int64(0);
    while pos < n_total_i
        hi = min(pos + chunk, n_total_i);
        idx_local = (pos : hi - 1).';

        % digit sum A -> M = A - N*s, per state. GPU: float-exact extraction
        % for n_total <= 2^52 (any d_loc -- see the MIN_IMAGE_IH_GPU note),
        % the int64 loop where the device supports gpuArray int64 arithmetic
        % (GPU_INT64_ARITH_OK), else fall through to the host branch below
        % (CUDA forward compatibility, e.g. B200 on R2024b).
        if use_gpu && (n_total <= 2^52 || gpu_int64_arith_ok())
            ig  = gpuArray(idx_local);
            A   = gpuArray.zeros(numel(idx_local), 1);
            if n_total <= 2^52
                sd = double(ig);
                dl = double(d_loc);
                for k = 1 : N_sites
                    dgp = rem(sd, dl);      % exact: sd is an integer < 2^52
                    A   = A + dgp;
                    sd  = (sd - dgp) / dl;  % exact (an integer multiple of dl)
                end
                clear sd;
            else
                tmp = ig;
                for k = 1 : N_sites
                    dg  = double(mod(tmp, d_loc_i64));
                    A   = A + dg;
                    tmp = (tmp - int64(dg)) / d_loc_i64;
                end
                clear tmp;
            end
            Mv   = round(2 * (A - N_s)) / 2;          % half-integer grid
            m_ok = (Mv >= 0);
            cand   = gather(ig(m_ok));
            Mv_can = gather(Mv(m_ok));
            clear ig A Mv m_ok;
        else
            A   = zeros(numel(idx_local), 1);
            tmp = idx_local;
            for k = 1 : N_sites
                dg  = double(mod(tmp, d_loc_i64));
                A   = A + dg;
                tmp = (tmp - int64(dg)) / d_loc_i64;
            end
            Mv   = round(2 * (A - N_s)) / 2;
            m_ok = (Mv >= 0);
            cand   = idx_local(m_ok);
            Mv_can = Mv(m_ok);
            clear A tmp Mv m_ok;
        end
        clear idx_local;

        if ~isempty(cand)
            % min_image is M-conserving -> a super-rep keeps its state's M.
            reps_c = min_image_Ih(cand, group, s_val);     % single output
            keep   = (reps_c == cand);
            if any(keep)
                sr_cells{end+1} = cand(keep);     %#ok<AGROW>
                Mv_cells{end+1} = Mv_can(keep);   %#ok<AGROW>
            end
            clear reps_c keep;
        end
        clear cand Mv_can;
        pos = hi;
    end

    if isempty(sr_cells)
        all_sr = int64([]); all_Mv = [];
    else
        all_sr = vertcat(sr_cells{:});   % globally ascending
        all_Mv = vertcat(Mv_cells{:});
    end
    clear sr_cells Mv_cells;

    %% Per-M cache: bin the super-reps, then build the stabiliser CSR.
    for M = 0 : M_max
        sr_M = all_sr(all_Mv == M);      % preserves ascending order
        caches{M+1} = build_cache_with_stab(sr_M, M, s_val, group);
    end
end


% ----------------------------------------------------------------
function cache = build_cache_with_stab(super_reps, M_target, s_val, group)
%BUILD_CACHE_WITH_STAB  Stabiliser CSR for a given super-rep list (mirrors
%   Step 3+4 of ENUMERATE_M_ORBITS_IH_GPU; kept self-contained here).
    d_loc   = round(2*s_val + 1);
    N_sites = double(group.N);
    super_reps = int64(super_reps(:));
    n_reps = numel(super_reps);

    cache.M_target = M_target;
    cache.s_val    = s_val;
    if n_reps == 0
        cache.super_reps = int64([]);
        cache.orbit_lens = int32([]);
        cache.stab_flat  = zeros(0, 1, 'uint16');
        cache.stab_ptr   = int64(1);
        return;
    end

    perm_powers = d_loc .^ (double(group.perms) - 1);   % [order x N_sites]
    d_loc_i64   = int64(d_loc);

    orbit_lens = zeros(n_reps, 1, 'int32');
    n_chunks   = ceil(n_reps / 131072);
    flat_cells = cell(n_chunks, 1);
    REP_CHUNK  = 131072;
    cc = 0;
    for cs = 1 : REP_CHUNK : n_reps
        cc  = cc + 1;
        ce  = min(cs + REP_CHUNK - 1, n_reps);
        idx = cs : ce;
        sr_chunk = super_reps(idx);

        digits_chunk = zeros(N_sites, numel(idx));
        tmp = sr_chunk;
        for k = 1 : N_sites
            digits_chunk(k, :) = double(mod(tmp, d_loc_i64)).';
            tmp = idivide(tmp, d_loc_i64);
        end
        states_g_chunk  = perm_powers * digits_chunk;
        stab_flag_chunk = (states_g_chunk == double(sr_chunk).');

        [g_rows, ~]      = find(stab_flag_chunk);
        flat_cells{cc}   = uint16(g_rows);     % g in 1..order (uint16: |G| up to 65535)
        counts_chunk     = sum(stab_flag_chunk, 1).';
        orbit_lens(idx)  = int32(group.order ./ counts_chunk);
        clear digits_chunk states_g_chunk stab_flag_chunk g_rows counts_chunk;
    end

    cache.super_reps = super_reps;
    cache.orbit_lens = orbit_lens;
    cache.stab_flat  = vertcat(flat_cells{:});
    cache.stab_ptr   = [int64(1); int64(1) + cumsum(int64(group.order) ./ int64(orbit_lens))];
end
