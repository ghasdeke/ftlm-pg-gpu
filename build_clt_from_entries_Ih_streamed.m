function clt = build_clt_from_entries_Ih_streamed(entries, reps, V_per_rep, ...
                                                    eig_per_rep, n_per_rep, ...
                                                    irrep_data, d_irrep, group, ...
                                                    gpu_h)
%   *** NON-PRODUCTION -- archived precomputed-M-on-GPU path; bench/regression
%   only (no driver callers). Production skeleton path: build_entry_skeleton_Ih +
%   build_clt_skeleton_from_entries_Ih.
%BUILD_CLT_FROM_ENTRIES_IH_STREAMED  Stream M into VRAM, never into host RAM.
%
%   CLT = BUILD_CLT_FROM_ENTRIES_IH_STREAMED(ENTRIES, REPS, V_PER_REP,
%       EIG_PER_REP, N_PER_REP, IRREP_DATA, D_IRREP, GROUP, GPU_H)
%
%   Same mathematical contract as BUILD_CLT_FROM_ENTRIES_IH, but the
%   d_irrep x d_irrep x n_entries M tensor is NEVER materialised on the
%   host. Instead:
%
%     1. The destination is pre-allocated as two flat single-precision
%        gpuArrays of length d^2 * n_entries (one for real, one for
%        imaginary parts), filling the GPU memory budget for M directly.
%
%     2. The host loops over chunks of CHUNK_SIZE (default 500 000)
%        entries. Per chunk it does the small pagemtimes M_e construction
%        on the host (~ 200-400 MB peak host RAM for d = 5), converts the
%        result to single precision, and uploads the flat slice into the
%        right offsets of the GPU arrays via slice assignment.
%
%     3. The host chunk is then discarded before the next iteration.
%
%   On the s = 1/2 icosidodecahedron H_g sector (n_entries ~ 80 M, d = 5)
%   this caps host RAM at ~ 600-800 MB transient instead of the ~ 32 GB
%   that BUILD_CLT_FROM_ENTRIES_IH would have allocated as a complex
%   double M_tens. Peak VRAM for the M tensor is 2 * n_entries * d^2 * 4
%   bytes (single complex split layout): ~ 16 GB for H_g, fits in 20 GB
%   on the RTX 4000 SFF Ada.
%
%   The CLT struct returned is marked with M_ON_GPU = true and contains
%   the flat gpuArrays M_RE_FLAT_GPU and M_IM_FLAT_GPU instead of a host
%   M tensor. RUN_FTLM_PG_SECTOR_GPU_IH detects this flag and skips its
%   own host-to-GPU conversion, passing the gpuArrays directly to the
%   MEX init call.
%
%   This is the production scalability path for systems where the M
%   tensor would otherwise dominate host RAM. Tiny systems (icosahedron
%   s = 1/2) can still use BUILD_CLT_FROM_ENTRIES_IH if desired (it is
%   slightly faster for tiny n_entries because of the per-chunk launch
%   overhead saved here).
%
%   See also BUILD_CLT_FROM_ENTRIES_IH, RUN_FTLM_PG_SECTOR_GPU_IH,
%            COLLECT_CLT_ENTRIES_IH.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    if nargin < 9 || isempty(gpu_h)
        gpu_h = gpuDevice();
    end

    d        = d_irrep;
    n_active = numel(reps);

    if n_active == 0
        clt = empty_clt(d_irrep);
        return;
    end

    %% Filter+remap entries to the active rep numbering (same as host variant).
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
        c_sorted = double(entries.c_table(entries.c_idx(keep)));   % reconstruct per-entry
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

    %% Padded V, sqrt_eig, rho on the host (small).
    V_all = complex(zeros(d, d, n_active));
    sqrt_eig_all = ones(d, n_active);
    for i = 1 : n_active
        n_i = double(n_per_rep(i));
        if n_i > 0
            V_all(:, 1:n_i, i) = V_per_rep{i};
            sqrt_eig_all(1:n_i, i) = sqrt(eig_per_rep{i});
        end
    end
    rho_all = complex(zeros(d, d, group.order));
    for g = 1 : group.order
        rho_all(:, :, g) = conj(irrep_matrix(irrep_data, g, d));
    end

    %% Allocate the M flat arrays on the GPU (single precision).
    %  These are the ONLY large M-related allocations; no host M_tens.
    d2 = d * d;
    n_floats = double(d2) * double(n_coll);
    if n_floats > intmax('int32')
        % Even with the streamed build, the flat index can overflow.
        % Bail out early with a clear diagnostic so the caller can
        % switch to the on-the-fly path or tile by entries.
        error('build_clt_from_entries_Ih_streamed:Overflow', ...
              'Flat M length %.3g exceeds int32 indexing.', n_floats);
    end
    M_re_flat_gpu = gpuArray.zeros(n_floats, 1, 'single');
    M_im_flat_gpu = gpuArray.zeros(n_floats, 1, 'single');
    wait(gpu_h);

    %% Stream M chunks into the GPU buffers.
    %  CHUNK_SIZE in entries. Per chunk peak host RAM ~ CHUNK_SIZE * d^2
    %  * 16 bytes for the intermediate complex double, plus ~ same for the
    %  three gathered tensors. 500 k entries -> ~ 500 MB peak for d = 5,
    %  comfortably below any modern workstation budget.
    CHUNK_SIZE = 500000;

    for cs = 1 : CHUNK_SIZE : n_coll
        ce = min(cs + CHUNK_SIZE - 1, n_coll);
        idx = cs : ce;
        chunk_n = numel(idx);

        % Gather small per-entry tensors from the padded V / rho tables.
        V_t_chunk = V_all(:, :, tgt_sorted(idx));
        V_r_chunk = V_all(:, :, src_sorted(idx));
        rho_chunk = rho_all(:, :, g_sorted(idx));

        % Per-entry kernel V_t^dag * rho * V_r
        temp        = pagemtimes(pagectranspose(V_t_chunk), rho_chunk);
        inner_chunk = pagemtimes(temp, V_r_chunk);

        % Scaling factors c_a * sqrt(lambda_t / lambda_i)
        sqrt_t_3d     = reshape(sqrt_eig_all(:, tgt_sorted(idx)), [d, 1, chunk_n]);
        sqrt_r_inv_3d = reshape(1 ./ sqrt_eig_all(:, src_sorted(idx)), [1, d, chunk_n]);
        if c_is_const
            c_3d = c_const;          % scalar broadcasts across the chunk
        else
            c_3d = reshape(c_sorted(idx), [1, 1, chunk_n]);
        end

        % Final per-chunk M, complex double on host (~ chunk_n * d^2 * 16 B).
        M_chunk = c_3d .* inner_chunk .* sqrt_t_3d .* sqrt_r_inv_3d;

        % Flatten as single re/im and upload into the right GPU slice.
        flat_offset = (cs - 1) * d2;
        flat_idx    = flat_offset + (1 : chunk_n * d2);

        % gpuArray() promotes; slice assignment is GPU-side copy.
        M_re_flat_gpu(flat_idx) = gpuArray(single(real(M_chunk(:))));
        M_im_flat_gpu(flat_idx) = gpuArray(single(imag(M_chunk(:))));

        % Explicitly drop the largest host intermediates to keep peak
        % host RAM bounded across iterations.
        clear V_t_chunk V_r_chunk rho_chunk temp inner_chunk M_chunk;
    end

    %% Per-output-rep counts and offsets (same as host variant).
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

    %% Pack. M_on_gpu marks the new layout; the SpMV wrapper checks for it.
    clt.n_basis         = double(n_basis);
    clt.n_reps          = n_active;
    clt.d_irrep         = d_irrep;
    clt.rep_offsets     = rep_offsets;
    clt.n_per_rep       = n_per_rep;
    clt.diag_vals       = diag_vals_active;
    clt.entries_per_rep = entries_per_rep;
    clt.entry_offsets   = entry_offsets;
    clt.src_idx         = src_sorted;
    clt.M_on_gpu        = true;
    clt.M_re_flat_gpu   = M_re_flat_gpu;
    clt.M_im_flat_gpu   = M_im_flat_gpu;
    % NB: clt.M (host complex double) is intentionally NOT set.
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
    clt.M_on_gpu        = true;
    clt.M_re_flat_gpu   = gpuArray.zeros(0, 1, 'single');
    clt.M_im_flat_gpu   = gpuArray.zeros(0, 1, 'single');
end


function M = irrep_matrix(irrep_data, g, d)
    if d == 1
        M = complex(irrep_data(g));
    else
        M = complex(irrep_data(:, :, g));
    end
end
