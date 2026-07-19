function H = build_heisenberg_sparse_Ih_gamma2(super_reps, V_per_rep, ...
                                                eig_per_rep, n_per_rep, ...
                                                bonds, s_val, J, ...
                                                irrep_data, d_irrep, group)
%BUILD_HEISENBERG_SPARSE_IH_GAMMA2  Sparse H on (M, Gamma) basis
%   (pagemtimes-batched).
%
%   Same external behaviour as before. Internally, the per-entry matrix
%   element computation (V_t^dag * rho_Gamma(g)^T * V_r * scaling) is
%   now batched via PAGEMTIMES on padded 3D tensors, and the triplet
%   write loop is chunk-vectorised. The two changes together cut the
%   build time on a ~ 3e6 entry sector (typical for s = 2 H_g, M = 0)
%   from of order a minute to a few seconds.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    d_loc   = round(2*s_val + 1);
    N_sites = double(group.N);     % was hardcoded 12; now driven by group
    n_total = double(d_loc)^N_sites;
    n_reps  = numel(super_reps);
    n_b     = size(bonds, 1);
    powers  = int64(d_loc) .^ int64((0 : N_sites - 1)');
    d       = d_irrep;

    %% rep_offsets + n_basis
    rep_offsets = zeros(n_reps, 1, 'int32');
    n_basis = int32(0);
    for i = 1 : n_reps
        rep_offsets(i) = n_basis;
        n_basis = n_basis + n_per_rep(i);
    end
    n_basis = double(n_basis);
    if n_basis == 0
        H = sparse([], [], [], 0, 0); return;
    end

    %% Compressed lookup (32-state bitmap), 16x smaller than the dense
    %  int32 lookup. Critical at icosidodecahedron scale (n_total = 2^30
    %  on s=1/2 -> 256 MB instead of 4 GB).
    assert(n_total <= 2^32, ...
        'build_heisenberg_sparse_Ih_gamma2: n_total too large for the 32-state bitmap CLT.');
    lookup_clt = build_clt_lookup(super_reps, n_total);

    %% Digit decomposition + diagonal Sz Sz
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

    %% Padded V, sqrt(eig), rho tensors -- same convention as in
    %  BUILD_CLT_PG_IH (sqrt_eig padded with 1, V padded with 0).
    V_all = complex(zeros(d, d, n_reps));
    for i = 1 : n_reps
        n_i = double(n_per_rep(i));
        if n_i > 0, V_all(:, 1:n_i, i) = V_per_rep{i}; end
    end
    sqrt_eig_all = ones(d, n_reps);
    for i = 1 : n_reps
        n_i = double(n_per_rep(i));
        if n_i > 0, sqrt_eig_all(1:n_i, i) = sqrt(eig_per_rep{i}); end
    end
    rho_all = complex(zeros(d, d, group.order));
    for g = 1 : group.order
        rho_all(:, :, g) = conj(irrep_matrix(irrep_data, g, d));
    end

    %% Phase 1: collect (src, tgt, g_min, c_a) via vectorised flips
    N_max = n_reps * n_b * 2;
    all_src = zeros(N_max, 1, 'int32');
    all_tgt = zeros(N_max, 1, 'int32');
    all_g   = zeros(N_max, 1, 'int32');
    all_c   = zeros(N_max, 1);
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
            j_idx_vec = query_clt_lookup(lookup_clt, rep_a_vec);
            in_basis  = j_idx_vec > 0;
            if ~any(in_basis), continue; end

            src_idx_can = src_idx_can(in_basis);
            j_idx_vec   = j_idx_vec(in_basis);
            c_a_vec     = c_a_vec(in_basis);
            g_min_vec   = g_min_vec(in_basis);

            n_surv = numel(src_idx_can);
            slot   = (n_coll + 1) : (n_coll + n_surv);
            all_src(slot) = src_idx_can;
            all_tgt(slot) = j_idx_vec;
            all_g(slot)   = g_min_vec;
            all_c(slot)   = c_a_vec;
            n_coll = n_coll + n_surv;
        end
    end

    all_src = all_src(1:n_coll);
    all_tgt = all_tgt(1:n_coll);
    all_g   = all_g(1:n_coll);
    all_c   = all_c(1:n_coll);

    %% Pre-compute per-rep counts for triplet sizing
    n_t_vec = double(n_per_rep(all_tgt));
    n_r_vec = double(n_per_rep(all_src));
    n_trip_per_entry = n_t_vec .* n_r_vec;
    n_trip_offdiag   = sum(n_trip_per_entry);

    rows = zeros(n_basis + n_trip_offdiag, 1);
    cols = zeros(n_basis + n_trip_offdiag, 1);
    vals = complex(zeros(n_basis + n_trip_offdiag, 1));

    %% Diagonal entries
    nz = 0;
    for i = 1 : n_reps
        off_i = double(rep_offsets(i));
        n_i = double(n_per_rep(i));
        for k = 1 : n_i
            nz = nz + 1;
            rows(nz) = off_i + k;
            cols(nz) = off_i + k;
            vals(nz) = diag_vals(i);
        end
    end

    %% Phase 2: batched M_block build + chunked triplet emission
    chunk_size = 100000;
    off_t_vec = double(rep_offsets(all_tgt));
    off_s_vec = double(rep_offsets(all_src));

    for chunk_start = 1 : chunk_size : n_coll
        chunk_end = min(chunk_start + chunk_size - 1, n_coll);
        idx = chunk_start : chunk_end;
        cs  = numel(idx);

        V_t_chunk = V_all(:, :, all_tgt(idx));
        V_r_chunk = V_all(:, :, all_src(idx));
        rho_chunk = rho_all(:, :, all_g(idx));

        temp = pagemtimes(pagectranspose(V_t_chunk), rho_chunk);
        inner_chunk = pagemtimes(temp, V_r_chunk);

        sqrt_t_3d     = reshape(sqrt_eig_all(:, all_tgt(idx)), [d, 1, cs]);
        sqrt_r_inv_3d = reshape(1 ./ sqrt_eig_all(:, all_src(idx)), [1, d, cs]);
        c_3d = reshape(all_c(idx), [1, 1, cs]);

        M_chunk = c_3d .* inner_chunk .* sqrt_t_3d .* sqrt_r_inv_3d;

        % Emit triplets for the chunk. For each entry e in the chunk:
        %   for k' in 1..n_t(e), k in 1..n_r(e): triplet (off_t+k', off_s+k, M[k', k, e])
        % Vectorise via the (KP, KK) grid + a per-entry mask.
        [KP_grid, KK_grid] = ndgrid(1:d, 1:d);
        KP_flat = KP_grid(:);   % d^2 x 1
        KK_flat = KK_grid(:);
        n_t_chunk = n_t_vec(idx);    % cs x 1
        n_r_chunk = n_r_vec(idx);
        off_t_chunk = off_t_vec(idx);
        off_s_chunk = off_s_vec(idx);

        % mask (cs x d^2): true iff kp <= n_t and kk <= n_r for that entry
        mask_mat = (KP_flat.' <= n_t_chunk) & (KK_flat.' <= n_r_chunk);

        rows_mat = off_t_chunk + KP_flat.';   % cs x d^2
        cols_mat = off_s_chunk + KK_flat.';
        vals_mat = reshape(M_chunk, d^2, cs).';   % cs x d^2

        keep = mask_mat(:);
        n_keep = sum(keep);
        rows(nz + 1 : nz + n_keep) = rows_mat(keep);
        cols(nz + 1 : nz + n_keep) = cols_mat(keep);
        vals(nz + 1 : nz + n_keep) = vals_mat(keep);
        nz = nz + n_keep;
    end

    %% Realified (FS=+1) irreps -> real V and real rho -> a real H block (the
    %  imaginary part is FP noise from the complex-typed intermediates). Collapse
    %  to real storage so the dense-ED / CPU-fallback paths take the real eig /
    %  real Lanczos branch (isreal(H_block) == true). Complex irreps (I_h
    %  T1g..Hu, momentum k) keep the complex-Hermitian H unchanged.
    if isreal(irrep_data) && nz > 0 && ...
            max(abs(imag(vals(1:nz)))) <= 1e-12 * max(1, max(abs(real(vals(1:nz)))))
        vals = real(vals);
    end

    H = sparse(rows(1:nz), cols(1:nz), vals(1:nz), n_basis, n_basis);
end


% ----------------------------------------------------------------
function M = irrep_matrix(irrep_data, g, d)
    if d == 1
        M = complex(irrep_data(g));
    else
        M = complex(irrep_data(:, :, g));
    end
end
