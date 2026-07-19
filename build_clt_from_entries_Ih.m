function clt = build_clt_from_entries_Ih(entries, reps, V_per_rep, eig_per_rep, ...
                                          n_per_rep, irrep_data, d_irrep, group)
%   *** NON-PRODUCTION -- bench/regression reference only (no driver callers).
%   Production skeleton path: build_entry_skeleton_Ih +
%   build_clt_skeleton_from_entries_Ih.
%BUILD_CLT_FROM_ENTRIES_IH  Irrep-dependent CLT half (Phase 2 + packing).
%
%   CLT = BUILD_CLT_FROM_ENTRIES_IH(ENTRIES, REPS, V_PER_REP, EIG_PER_REP,
%                                    N_PER_REP, IRREP_DATA, D_IRREP, GROUP)
%
%   Consumes the irrep-INDEPENDENT entry list produced by
%   COLLECT_CLT_ENTRIES_IH (called once per M sector, over the full cache
%   super-rep list) and finishes the CLT for one specific I_h irrep.
%
%   REPS, V_PER_REP, EIG_PER_REP, N_PER_REP come from
%   APPLY_IRREP_TO_ORBITS and may be a SUBSET of the cache super-reps
%   (only reps with n_Gamma(r) > 0). The first thing this function does
%   is filter ENTRIES to drop any entry whose source or target rep was
%   not retained for this irrep, then re-index the surviving entries
%   into the filtered rep numbering (1..numel(REPS)). The resulting
%   CLT struct is then binary-identical to what BUILD_CLT_PG_IH would
%   have built directly.
%
%   See also COLLECT_CLT_ENTRIES_IH, BUILD_CLT_PG_IH, APPLY_IRREP_TO_ORBITS.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    d        = d_irrep;
    n_active = numel(reps);

    %% Empty-irrep early out
    if n_active == 0
        clt = empty_clt(d_irrep);
        return;
    end

    %% Map cache positions -> active (filtered) positions.
    %  entries.super_reps is the full cache list; reps is the active
    %  subset (still sorted ascending, since apply_irrep_to_orbits
    %  preserves order via a keep_mask on the sorted cache).
    [in_active, active_pos] = ismember(int64(entries.super_reps), int64(reps));
    % in_active(c)  = true iff cache rep c survived
    % active_pos(c) = its 1-based position in REPS, 0 otherwise

    %% Filter entries: keep only those where BOTH endpoints survived.
    src_cache = double(entries.src_sorted);     % indices into cache.super_reps
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

    %% Diag values for the surviving reps, in active order.
    diag_vals_active = entries.diag_vals(in_active);

    %% rep_offsets and n_basis (active reps only)
    % int64 (2026-07 64-bit basis-offset ABI): int32 would silently SATURATE
    % at 2^31-1 once n_basis = d*n_reps crosses it; the kernel now expects
    % int64 rep_offsets and validates the class.
    rep_offsets = zeros(n_active, 1, 'int64');
    n_basis = int64(0);
    for i = 1 : n_active
        rep_offsets(i) = n_basis;
        n_basis = n_basis + int64(n_per_rep(i));
    end

    %% Pad V_per_rep / sqrt(eig) into padded tensors over active reps.
    V_all = complex(zeros(d, d, n_active));
    sqrt_eig_all = ones(d, n_active);
    for i = 1 : n_active
        n_i = double(n_per_rep(i));
        if n_i > 0
            V_all(:, 1:n_i, i) = V_per_rep{i};
            sqrt_eig_all(1:n_i, i) = sqrt(eig_per_rep{i});
        end
    end

    %% Precompute conj(rho_Gamma(g)) as a [d x d x group.order] tensor.
    rho_all = complex(zeros(d, d, group.order));
    for g = 1 : group.order
        rho_all(:, :, g) = conj(irrep_matrix(irrep_data, g, d));
    end

    %% Phase 2: batched M_e build via PAGEMTIMES (chunked).
    M_tens = complex(zeros(d, d, n_coll));
    chunk_size = 100000;
    for chunk_start = 1 : chunk_size : n_coll
        chunk_end = min(chunk_start + chunk_size - 1, n_coll);
        idx = chunk_start : chunk_end;
        cs  = numel(idx);

        V_t_chunk = V_all(:, :, tgt_sorted(idx));
        V_r_chunk = V_all(:, :, src_sorted(idx));
        rho_chunk = rho_all(:, :, g_sorted(idx));

        temp = pagemtimes(pagectranspose(V_t_chunk), rho_chunk);
        inner_chunk = pagemtimes(temp, V_r_chunk);

        sqrt_t_3d = reshape(sqrt_eig_all(:, tgt_sorted(idx)), [d, 1, cs]);
        sqrt_r_inv_3d = reshape(1 ./ sqrt_eig_all(:, src_sorted(idx)), [1, d, cs]);
        if c_is_const
            c_3d = c_const;          % scalar broadcasts across the chunk
        else
            c_3d = reshape(c_sorted(idx), [1, 1, cs]);
        end

        M_tens(:, :, idx) = c_3d .* inner_chunk .* sqrt_t_3d .* sqrt_r_inv_3d;
    end

    %% Realified (FS=+1) irreps -> real M tensor (see BUILD_CLT_PG_IH); collapse
    %  to real storage so the gather SpMV stays on the real arithmetic path.
    if isreal(irrep_data) && ~isempty(M_tens) && ...
            max(abs(imag(M_tens(:)))) <= 1e-12 * max(1, max(abs(real(M_tens(:)))))
        M_tens = real(M_tens);
    end

    %% Per-output-rep counts and offsets (over active reps)
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

    %% Pack (identical field layout to build_clt_pg_Ih)
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
