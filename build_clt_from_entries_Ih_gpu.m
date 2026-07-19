function clt = build_clt_from_entries_Ih_gpu(entries, reps, V_per_rep, ...
                                              eig_per_rep, n_per_rep, ...
                                              irrep_data, d_irrep, group)
%   *** NON-PRODUCTION -- bench/regression reference only (no driver callers).
%   Production skeleton path: build_entry_skeleton_Ih +
%   build_clt_skeleton_from_entries_Ih.
%BUILD_CLT_FROM_ENTRIES_IH_GPU  GPU pagemtimes variant of Phase 2.
%
%   STATUS: ARCHIVED EXPERIMENT (Option A). On the tested RTX 4000 SFF
%   Ada this path is slower than the CPU pagemtimes variant
%   BUILD_CLT_FROM_ENTRIES_IH for the s = 2 M = 0 sweep (39.9 s vs
%   30.3 s for the 10 irreps combined). Reason: the per-entry d x d
%   complex matmuls (d <= 5) are too small for cuBLAS-Batched to
%   amortise its kernel-launch overhead; MKL on host stays in L1/L2
%   cache and wins. Retained as a fall-back for hypothetical larger d
%   problems and as a benchmarking option in BENCH_S2_M0_BREAKDOWN
%   (mode = 'gpu'). NOT in the production driver.
%
%   CLT = BUILD_CLT_FROM_ENTRIES_IH_GPU(ENTRIES, REPS, V_PER_REP,
%       EIG_PER_REP, N_PER_REP, IRREP_DATA, D_IRREP, GROUP)
%
%   Same contract as BUILD_CLT_FROM_ENTRIES_IH (drop-in replacement),
%   but runs the per-chunk PAGEMTIMES on the GPU via gpuArray. The
%   padded V_all, rho_all, sqrt_eig_all tensors are uploaded ONCE per
%   call; each chunk then gathers V_t/V_r/rho slices on the device and
%   issues a cuBLAS Batched GEMM. The d x d x cs M_chunk is gathered
%   back to host memory and stored in clt.M (CPU complex double), so
%   the downstream RUN_FTLM_PG_SECTOR_GPU_IH wrapper does not need to
%   change.
%
%   For the s = 2 H_g sector (n_entries ~ 0.6e6), CPU pagemtimes runs
%   at the system DRAM bandwidth (~ 50 GB/s on a typical workstation
%   CPU). The RTX 4000 SFF Ada has ~ 360 GB/s VRAM bandwidth, so this
%   path should be 5-8x faster on the same Phase 2 workload, all else
%   equal. Net wall-time saving on the 10-irrep s = 2 M = 0 sweep:
%   expected ~ 20-25 s of the current 30 s Phase 2 budget.
%
%   Falls back to BUILD_CLT_FROM_ENTRIES_IH if no GPU is available.
%
%   See also BUILD_CLT_FROM_ENTRIES_IH, COLLECT_CLT_ENTRIES_IH,
%            RUN_FTLM_PG_SECTOR_GPU_IH.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    %% Probe GPU; fall back to CPU variant if unavailable.
    use_gpu = false;
    try
        gd = gpuDevice();
        if gd.DeviceSupported, use_gpu = true; end
    catch
        use_gpu = false;
    end
    if ~use_gpu
        clt = build_clt_from_entries_Ih(entries, reps, V_per_rep, ...
                                         eig_per_rep, n_per_rep, ...
                                         irrep_data, d_irrep, group);
        return;
    end

    d        = d_irrep;
    n_active = numel(reps);

    if n_active == 0
        clt = empty_clt(d_irrep);
        return;
    end

    %% Map cache positions -> active positions (same as CPU variant).
    [in_active, active_pos] = ismember(int64(entries.super_reps), int64(reps));
    src_cache = double(entries.src_sorted);
    tgt_cache = double(entries.tgt_sorted);
    keep = in_active(src_cache) & in_active(tgt_cache);

    src_sorted = int32(active_pos(src_cache(keep)));
    tgt_sorted = int32(active_pos(tgt_cache(keep)));
    g_sorted   = entries.g_sorted(keep);
    % Constant-c (s=1/2): entries carries a scalar c_const, not a per-entry
    % array. isfield guard keeps old entries structs working.
    c_is_const   = isfield(entries, 'c_is_const')   && entries.c_is_const;
    c_is_indexed = isfield(entries, 'c_is_indexed') && entries.c_is_indexed;
    if c_is_const
        c_sorted = [];
        c_const  = entries.c_const;
    elseif c_is_indexed
        c_sorted = double(entries.c_table(double(gather(entries.c_idx(keep)))));  % reconstruct
        c_const  = [];
    else
        c_sorted = entries.c_sorted(keep);
        c_const  = [];
    end
    n_coll     = numel(src_sorted);

    diag_vals_active = entries.diag_vals(in_active);

    %% rep_offsets + n_basis (active reps only)
    % int64 (2026-07 64-bit basis-offset ABI; kernel validates the class).
    rep_offsets = zeros(n_active, 1, 'int64');
    n_basis = int64(0);
    for i = 1 : n_active
        rep_offsets(i) = n_basis;
        n_basis = n_basis + int64(n_per_rep(i));
    end

    %% Build padded V_all, sqrt_eig_all, rho_all on the HOST first.
    V_all_h = complex(zeros(d, d, n_active));
    sqrt_eig_all_h = ones(d, n_active);
    for i = 1 : n_active
        n_i = double(n_per_rep(i));
        if n_i > 0
            V_all_h(:, 1:n_i, i) = V_per_rep{i};
            sqrt_eig_all_h(1:n_i, i) = sqrt(eig_per_rep{i});
        end
    end
    rho_all_h = complex(zeros(d, d, group.order));
    for g = 1 : group.order
        rho_all_h(:, :, g) = conj(irrep_matrix(irrep_data, g, d));
    end

    %% Upload constant tensors ONCE.
    V_all_g    = gpuArray(V_all_h);
    sqrt_eig_g = gpuArray(sqrt_eig_all_h);
    rho_all_g  = gpuArray(rho_all_h);

    %% Phase 2 on GPU: per-chunk gather + pagemtimes via cuBLAS Batched.
    %  Chunk size: keep per-chunk peak ~ a few hundred MB.
    M_tens = complex(zeros(d, d, n_coll));    % output stays on host
    chunk_size = 200000;                      % larger than CPU variant; VRAM is bigger

    src_sorted_g = gpuArray(src_sorted);
    tgt_sorted_g = gpuArray(tgt_sorted);
    g_sorted_g   = gpuArray(g_sorted);
    if c_is_const, c_sorted_g = []; else, c_sorted_g = gpuArray(c_sorted); end

    for chunk_start = 1 : chunk_size : n_coll
        chunk_end = min(chunk_start + chunk_size - 1, n_coll);
        idx_g = gpuArray(int32(chunk_start : chunk_end));
        cs    = numel(chunk_start : chunk_end);

        V_t_chunk = V_all_g(:, :, tgt_sorted_g(idx_g));
        V_r_chunk = V_all_g(:, :, src_sorted_g(idx_g));
        rho_chunk = rho_all_g(:, :, g_sorted_g(idx_g));

        temp        = pagemtimes(pagectranspose(V_t_chunk), rho_chunk);
        inner_chunk = pagemtimes(temp, V_r_chunk);

        sqrt_t_3d     = reshape(sqrt_eig_g(:, tgt_sorted_g(idx_g)), [d, 1, cs]);
        sqrt_r_inv_3d = reshape(1 ./ sqrt_eig_g(:, src_sorted_g(idx_g)), [1, d, cs]);
        if c_is_const
            c_3d = c_const;          % scalar broadcasts across the chunk
        else
            c_3d = reshape(c_sorted_g(idx_g), [1, 1, cs]);
        end

        M_chunk = c_3d .* inner_chunk .* sqrt_t_3d .* sqrt_r_inv_3d;
        M_tens(:, :, chunk_start:chunk_end) = gather(M_chunk);
    end

    %% Per-output-rep counts and offsets
    entries_per_rep = zeros(n_active, 1, 'int32');
    if n_coll > 0
        counts = accumarray(double(tgt_sorted), 1, [n_active, 1]);
        entries_per_rep = int32(counts);
    end
    entry_offsets = zeros(n_active, 1, 'int32');
    n_entries = int32(0);
    for t = 1 : n_active
        entry_offsets(t) = n_entries;
        n_entries = n_entries + entries_per_rep(t);
    end

    %% Pack (identical layout to build_clt_pg_Ih / build_clt_from_entries_Ih)
    clt.n_basis         = double(n_basis);
    clt.n_reps          = n_active;
    clt.d_irrep         = d_irrep;
    clt.rep_offsets     = rep_offsets;
    clt.n_per_rep       = n_per_rep;
    clt.diag_vals       = diag_vals_active;
    clt.entries_per_rep = entries_per_rep;
    clt.entry_offsets   = entry_offsets;
    clt.src_idx         = src_sorted;
    clt.M               = M_tens;
end


% ----------------------------------------------------------------
function clt = empty_clt(d_irrep)
    clt.n_basis         = 0;
    clt.n_reps          = 0;
    clt.d_irrep         = d_irrep;
    clt.rep_offsets     = zeros(0, 1, 'int64');
    clt.n_per_rep       = zeros(0, 1, 'int32');
    clt.diag_vals       = zeros(0, 1);
    clt.entries_per_rep = zeros(0, 1, 'int32');
    clt.entry_offsets   = zeros(0, 1, 'int32');
    clt.src_idx         = zeros(0, 1, 'int32');
    clt.M               = complex(zeros(d_irrep, d_irrep, 0));
end


function M = irrep_matrix(irrep_data, g, d)
    if d == 1
        M = complex(irrep_data(g));
    else
        M = complex(irrep_data(:, :, g));
    end
end
