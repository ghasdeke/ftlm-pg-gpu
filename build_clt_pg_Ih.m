function clt = build_clt_pg_Ih(super_reps, V_per_rep, eig_per_rep, ...
                                n_per_rep, bonds, s_val, J, ...
                                irrep_data, d_irrep, group)
%   *** NON-PRODUCTION -- regression reference (hardcoded N=12; used by
%   test_clt_refactor / test_milestone_G_*). NOT group-generic; do NOT use for
%   lattices. Production skeleton path: build_entry_skeleton_Ih +
%   build_clt_skeleton_from_entries_Ih.
%BUILD_CLT_PG_IH  Compressed lookup table for the I_h gather SpMV (pagemtimes-batched).
%
%   STATUS: REGRESSION REFERENCE (no longer in the production driver).
%   The production driver FTLM_OBSERVABLES_PG_GPU_IH now uses the
%   two-phase split COLLECT_CLT_ENTRIES_IH (once per M) +
%   BUILD_CLT_FROM_ENTRIES_IH (per irrep), which saves ~14 s of the
%   redundant irrep-independent Phase 1 work on the s = 2 M = 0 sweep.
%   This monolithic builder is retained as the bit-for-bit reference
%   for TEST_CLT_REFACTOR and a few older smoke tests (TEST_MILESTONE_G_GPU,
%   TEST_MILESTONE_G_SPMV_CLT_IH).
%
%   Two-phase vectorised implementation:
%
%   PHASE 1 (entry collection): iterates over the 60 (bond, sign) pairs
%       and, for each, calls MIN_IMAGE_IH once on the entire vector of
%       generated spin-flip states. Collects (src_idx, tgt_idx, g_min,
%       c_a) into preallocated flat arrays. NO scalar MIN_IMAGE calls.
%
%   PHASE 2 (batched M_e tensor build): converts V_per_rep, sqrt(eig_per_rep)
%       and the precomputed conj(rho_Gamma(g)) matrices into padded 3D
%       tensors, then computes
%           M^(e)_{k', k} = c_a^(e) * sqrt(lambda_t / lambda_i) *
%                           [V_t^dag * rho_Gamma(g)^T * V_i]_{k', k}
%       for ALL entries at once via PAGEMTIMES, in chunks of ~ 100k
%       entries to bound peak memory. This replaces the previous
%       per-entry interpreter loop (~ tens of microseconds per entry,
%       dominant cost) with a batched BLAS call that scales near the
%       memory-bandwidth limit. On d_irrep = 5 and n_entries ~ 3e6 the
%       step drops from ~ minute to a few seconds.
%
%   Output convention: clt.M is now a [d_irrep x d_irrep x n_entries]
%   PADDED complex double tensor (NOT a cell array). For entry e, the
%   active submatrix is clt.M(1:n_per_rep(tgt), 1:n_per_rep(src), e);
%   the remaining rows/cols are exactly zero (by construction of the
%   padded V_all and the inner product). The GPU wrapper can therefore
%   single-convert and upload clt.M in one step, eliminating the prior
%   per-entry repack loop.
%
%   Backwards-incompatible storage change: clt.M is no longer a cell
%   array. SPMV_PG_IH_CLT_MATLAB has been updated to slice the tensor
%   per entry instead of dereferencing a cell.
%
%   See also SPMV_PG_IH_CLT_MATLAB, RUN_FTLM_PG_SECTOR_GPU_IH.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    d_loc   = round(2*s_val + 1);
    N_sites = 12;
    n_total = double(d_loc)^N_sites;
    n_reps  = numel(super_reps);
    n_b     = size(bonds, 1);
    powers  = int64(d_loc) .^ int64((0 : N_sites - 1)');
    d       = d_irrep;

    %% rep_offsets and n_basis
    % int64 (2026-07 64-bit basis-offset ABI; kernel validates the class).
    rep_offsets = zeros(n_reps, 1, 'int64');
    n_basis = int64(0);
    for i = 1 : n_reps
        rep_offsets(i) = n_basis;
        n_basis = n_basis + int64(n_per_rep(i));
    end

    %% State -> rep-index lookup (1-based; 0 = not in basis)
    assert(n_total <= 2^32, ...
        'build_clt_pg_Ih: n_total too large for dense lookup.');
    lookup = zeros(n_total, 1, 'int32');
    lookup(double(super_reps) + 1) = int32(1 : n_reps);

    %% Digit decomposition + diagonal Sz Sz per rep
    mi = zeros(n_reps, N_sites);
    tmp = super_reps;
    for site = 1 : N_sites
        dg = double(mod(tmp, int64(d_loc)));
        mi(:, site) = dg - s_val;
        tmp = (tmp - int64(dg)) / int64(d_loc);
    end

    diag_vals = zeros(n_reps, 1);
    for b = 1 : n_b
        diag_vals = diag_vals + J * mi(:, bonds(b, 1)) .* mi(:, bonds(b, 2));
    end

    %% Pad V_per_rep into a [d x d x n_reps] complex tensor.
    %  Columns beyond n_per_rep(i) are zero. This ensures inner-product
    %  rows/cols beyond n_per_rep are exactly zero, so M_tens carries
    %  zero padding without any masking.
    V_all = complex(zeros(d, d, n_reps));
    for i = 1 : n_reps
        n_i = double(n_per_rep(i));
        if n_i > 0
            V_all(:, 1:n_i, i) = V_per_rep{i};
        end
    end

    %% Pad sqrt(eig) into a [d x n_reps] matrix with 1 in the unused
    %  slots. The 1's protect 1./sqrt_eig from producing Inf/NaN; they
    %  never reach M_tens because the corresponding inner-product
    %  entries are zero.
    sqrt_eig_all = ones(d, n_reps);
    for i = 1 : n_reps
        n_i = double(n_per_rep(i));
        if n_i > 0
            sqrt_eig_all(1:n_i, i) = sqrt(eig_per_rep{i});
        end
    end

    %% Precompute conj(rho_Gamma(g)) as a [d x d x group.order] tensor.
    rho_all = complex(zeros(d, d, group.order));
    for g = 1 : group.order
        rho_all(:, :, g) = conj(irrep_matrix(irrep_data, g, d));
    end

    %% PHASE 1: collect entry indices via vectorised spin flips
    N_max = n_reps * n_b * 2;
    all_src_flat = zeros(N_max, 1, 'int32');
    all_tgt_flat = zeros(N_max, 1, 'int32');
    all_g_flat   = zeros(N_max, 1, 'int32');
    all_c_flat   = zeros(N_max, 1);
    n_coll = 0;

    for b = 1 : n_b
        si = bonds(b, 1); sj = bonds(b, 2);
        m_si_col = mi(:, si);
        m_sj_col = mi(:, sj);

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

            [rep_a_vec, g_min_vec] = min_image_Ih(state_a_vec, group, s_val);
            t_idx_vec = lookup(double(rep_a_vec) + 1);
            in_basis  = t_idx_vec > 0;
            if ~any(in_basis), continue; end

            src_idx_can = src_idx_can(in_basis);
            t_idx_vec   = t_idx_vec(in_basis);
            c_a_vec     = c_a_vec(in_basis);
            g_min_vec   = g_min_vec(in_basis);

            n_surv = numel(src_idx_can);
            slot   = (n_coll + 1) : (n_coll + n_surv);
            all_src_flat(slot) = src_idx_can;
            all_tgt_flat(slot) = t_idx_vec;
            all_g_flat(slot)   = g_min_vec;
            all_c_flat(slot)   = c_a_vec;
            n_coll = n_coll + n_surv;
        end
    end

    all_src_flat = all_src_flat(1:n_coll);
    all_tgt_flat = all_tgt_flat(1:n_coll);
    all_g_flat   = all_g_flat(1:n_coll);
    all_c_flat   = all_c_flat(1:n_coll);

    %% Sort entries by target rep so the gather SpMV can iterate
    %  contiguous slices per output rep.
    [tgt_sorted, order] = sort(all_tgt_flat);
    src_sorted = all_src_flat(order);
    g_sorted   = all_g_flat(order);
    c_sorted   = all_c_flat(order);

    %% PHASE 2: batched M_e build via PAGEMTIMES (chunked).
    %  Per-chunk peak memory: cs * d^2 * 16 bytes (complex double) for
    %  each of V_t_chunk, V_r_chunk, rho_chunk, temp, inner_chunk.
    %  For d = 5, cs = 100k -> ~ 40 MB per intermediate, ~ 200 MB
    %  combined peak, easily fits on a standard workstation.
    M_tens = complex(zeros(d, d, n_coll));
    chunk_size = 100000;
    for chunk_start = 1 : chunk_size : n_coll
        chunk_end = min(chunk_start + chunk_size - 1, n_coll);
        idx = chunk_start : chunk_end;
        cs  = numel(idx);

        V_t_chunk = V_all(:, :, tgt_sorted(idx));    % d x d x cs
        V_r_chunk = V_all(:, :, src_sorted(idx));    % d x d x cs
        rho_chunk = rho_all(:, :, g_sorted(idx));    % d x d x cs

        % Per-slice: V_t' * rho * V_r  (conjugate transpose on slice 1)
        temp = pagemtimes(pagectranspose(V_t_chunk), rho_chunk);
        inner_chunk = pagemtimes(temp, V_r_chunk);

        % Per-entry scaling: M[k', k] = c_a * inner[k', k]
        %                              * sqrt_eig_t[k'] / sqrt_eig_r[k]
        sqrt_t_3d = reshape(sqrt_eig_all(:, tgt_sorted(idx)), [d, 1, cs]);
        sqrt_r_inv_3d = reshape(1 ./ sqrt_eig_all(:, src_sorted(idx)), [1, d, cs]);
        c_3d = reshape(c_sorted(idx), [1, 1, cs]);

        M_tens(:, :, idx) = c_3d .* inner_chunk .* sqrt_t_3d .* sqrt_r_inv_3d;
    end

    %% Realified (FS=+1) irreps -> real V and real rho -> a real M tensor (the
    %  imaginary part is FP noise from the complex-typed intermediates). Collapse
    %  to real storage so SPMV_PG_IH_CLT_MATLAB keeps the real arithmetic path.
    %  Complex irreps (I_h T1g..Hu, momentum k) keep the complex M unchanged.
    if isreal(irrep_data) && ~isempty(M_tens) && ...
            max(abs(imag(M_tens(:)))) <= 1e-12 * max(1, max(abs(real(M_tens(:)))))
        M_tens = real(M_tens);
    end

    %% Per-output-rep counts and offsets
    entries_per_rep = zeros(n_reps, 1, 'int32');
    if n_coll > 0
        counts = accumarray(double(tgt_sorted), 1, [n_reps, 1]);
        entries_per_rep = int32(counts);
    end
    entry_offsets = zeros(n_reps, 1, 'int32');
    n_entries = int32(0);
    for t = 1 : n_reps
        entry_offsets(t) = n_entries;
        n_entries = n_entries + entries_per_rep(t);
    end

    %% Pack
    clt.n_basis         = double(n_basis);
    clt.n_reps          = n_reps;
    clt.d_irrep         = d_irrep;
    clt.rep_offsets     = rep_offsets;
    clt.n_per_rep       = n_per_rep;
    clt.diag_vals       = diag_vals;
    clt.entries_per_rep = entries_per_rep;
    clt.entry_offsets   = entry_offsets;
    clt.src_idx         = src_sorted;
    clt.M               = M_tens;   % 3D padded tensor [d x d x n_entries]
end


% ----------------------------------------------------------------
function M = irrep_matrix(irrep_data, g, d)
    if d == 1
        M = complex(irrep_data(g));
    else
        M = complex(irrep_data(:, :, g));
    end
end
