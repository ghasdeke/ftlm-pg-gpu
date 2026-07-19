function entries = collect_clt_entries_Ih_gpu(super_reps, bonds, s_val, J, group)
%COLLECT_CLT_ENTRIES_IH_GPU  GPU-native irrep-independent CLT collect.
%
%   STATUS: DEPRECATED EXPERIMENT (Phase B.2.1, May 2026). The idea
%   was to eliminate the host-side entry arrays by having every
%   per-bond chunk produced via on-device logical indexing -- no
%   gpuArray(host_array) calls anywhere. This DID succeed in moving
%   the entries onto the GPU (verified, ~ +1.4 GB on the GPU side),
%   but MATLAB's internal gpuArray pool grew the host RAM by ~ 3 GB
%   per sector regardless, because the many small gpuArray operations
%   (mod, indexing, find, schnack_rank, ismember, vertcat, sort)
%   each leave host-side allocation residue that MATLAB does not
%   reliably recycle. Net effect: HOST RAM went UP by ~ 5 GB per
%   sector at the icosidodecahedron scale -- the opposite of the
%   goal. Same failure mode as Stufe 6b.
%
%   Code retained for reference. To actually eliminate the host-side
%   entry burden, the route forward is Phase B.2.2: a custom CUDA MEX
%   kernel that builds entries directly in VRAM without going through
%   any MATLAB gpuArray. Until then the production path is
%   collect_clt_entries_Ih with entries_storage='host'.
%
%   ENTRIES = COLLECT_CLT_ENTRIES_IH_GPU(SUPER_REPS, BONDS, S_VAL, J, GROUP)
%
%   Same contract as COLLECT_CLT_ENTRIES_IH but every large array is
%   computed and lives on the GPU. SUPER_REPS is uploaded once; mi,
%   diag_vals, the Schnack lookup, and the per-bond chunks all stay
%   in VRAM. Output entries struct has src/tgt/g/c on the GPU.
%
%   Crucially, the per-(bond, sign) chunks are produced via on-device
%   logical indexing of existing gpuArrays -- NOT via gpuArray(host).
%   This is the architectural fix for the Stufe-6b debacle where
%   per-chunk uploads inflated MATLAB's pinned-host pool.
%
%   Schnack-CR is the only supported lookup_method here, because the
%   16x-compressed bitmap CLT scales with n_total and is irrelevant at
%   the N>=32 scales for which this routine exists.
%
%   Spin-flip Z2 (ADD_SPIN_FLIP_Z2): supported as-is. The only group
%   action in this routine is MIN_IMAGE_IH_GPU, which handles the G x Z2
%   extension internally (it slices the cached perm_powers to the
%   permutation half and derives the flip branch from the same matmul);
%   g_min indices up to 2|G| fit the uint16 g storage below.
%
%   See also COLLECT_CLT_ENTRIES_IH, BUILD_LOOKUP_SCHNACK_GPU,
%            MIN_IMAGE_IH_GPU.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    d_loc   = round(2*s_val + 1);
    N_sites = double(group.N);
    n_reps  = numel(super_reps);
    n_b     = size(bonds, 1);

    %% Upload super_reps once. Stays on GPU for the whole M sector.
    super_reps_gpu = gpuArray(int64(super_reps(:)));

    %% Powers (small, but used on GPU for spin-flip arithmetic).
    powers_h   = int64(d_loc) .^ int64((0 : N_sites - 1)');
    powers_gpu = gpuArray(powers_h);

    %% mi (digit-decomposition - s) on GPU, used per-bond for spin-condition
    %  masks and c_a coefficients. Size n_reps x N_sites (~ 312 MB at N=30,
    %  ~ 18 GB at N=36 -- watch this for large systems; for N>=36 we may
    %  need to compute mi on-the-fly per bond instead).
    %  Digit extraction: float-exact for n_total <= 2^52 (any d_loc), the
    %  int64 loop where the device supports gpuArray int64 arithmetic, else
    %  host decomposition (CUDA forward compatibility, e.g. B200 -- see
    %  GPU_INT64_ARITH_OK). Identical integers on every path.
    d_loc_i64 = int64(d_loc);
    n_total_c = double(d_loc) ^ N_sites;
    % 2^31 elements-per-gpuArray cap (portability audit 2026-07-03): fail
    % EARLY with the remedy instead of a device error after minutes.
    assert(double(n_reps) * N_sites < 2^31, ...
        ['collect_clt_entries_Ih_gpu: the mi matrix (n_reps x N_sites = ', ...
         '%.3g elements) exceeds the 2^31 gpuArray cap. Use the host ', ...
         'collector (entries_storage=''host'', optionally entries_on_disk).'], ...
        double(n_reps) * N_sites);
    % Cap-closure inventory 2026-07-03: the final vertcat/sort in this
    % (deprecated, reference-only) collector builds ONE gpuArray of n_coll
    % entries -- n_entries is structurally UNBOUNDED, so gate the projected
    % upper bound up front too.
    assert(2 * double(n_reps) * size(bonds, 1) < 2^31, ...
        ['collect_clt_entries_Ih_gpu: projected entry count (<= 2*n_bonds*', ...
         'n_reps = %.3g) exceeds the 2^31 gpuArray cap of the final ', ...
         'vertcat/sort. Use entries_storage=''host'' (with entries_on_disk ', ...
         'for large systems).'], 2 * double(n_reps) * size(bonds, 1));
    mi_gpu = gpuArray.zeros(n_reps, N_sites);
    if n_total_c <= 2^52
        sd = double(super_reps_gpu);
        dl = double(d_loc);
        for site = 1 : N_sites
            dgp = rem(sd, dl);              % exact: sd is an integer < 2^52
            mi_gpu(:, site) = dgp - s_val;
            sd = (sd - dgp) / dl;           % exact (an integer multiple of dl)
        end
    elseif gpu_int64_arith_ok()
        tmp = super_reps_gpu;
        for site = 1 : N_sites
            dg = mod(tmp, d_loc_i64);
            mi_gpu(:, site) = double(dg) - s_val;
            tmp = (tmp - dg) / d_loc_i64;
        end
    else
        tmp_h = gather(super_reps_gpu);
        for site = 1 : N_sites
            dg_h = mod(tmp_h, d_loc_i64);
            mi_gpu(:, site) = gpuArray(double(dg_h) - s_val);
            tmp_h = (tmp_h - dg_h) / d_loc_i64;
        end
    end

    %% Diagonal Sz Sz (small, n_reps x 1, gathered to host at the end for
    %  schema compatibility with BUILD_CLT_SKELETON_FROM_ENTRIES_IH which
    %  indexes diag_vals on host).
    diag_vals_gpu = gpuArray.zeros(n_reps, 1);
    for b = 1 : n_b
        diag_vals_gpu = diag_vals_gpu + J * mi_gpu(:, bonds(b, 1)) .* mi_gpu(:, bonds(b, 2));
    end

    %% Schnack lookup (D_cum + sorted super_reps_rank) on the GPU.
    lookup_gpu = build_lookup_schnack_gpu(super_reps_gpu, s_val, N_sites);

    %% Cache perm_powers for min_image (avoid rebuilding 120 times).
    perms_d         = double(group.perms);
    perm_powers_gpu = gpuArray(d_loc .^ (perms_d - 1));

    %% Phase 1: per-bond entry collection, all on GPU.
    src_cells = cell(0, 1);
    tgt_cells = cell(0, 1);
    g_cells   = cell(0, 1);
    c_cells   = cell(0, 1);

    % Constant-c (s=1/2): store one scalar instead of an n_entries array.
    % Same contract as COLLECT_CLT_ENTRIES_IH (see its comment).
    c_is_const  = (abs(s_val - 0.5) < 1e-12);
    c_const_val = 0.5 * J;

    for b = 1 : n_b
        si = bonds(b, 1); sj = bonds(b, 2);
        m_si_col = mi_gpu(:, si);
        m_sj_col = mi_gpu(:, sj);

        for sign_dir = 1 : 2
            if sign_dir == 1
                can = (m_si_col < s_val - 1e-10) & (m_sj_col > -s_val + 1e-10);
            else
                can = (m_si_col > -s_val + 1e-10) & (m_sj_col < s_val - 1e-10);
            end
            % Gather a single bool to decide short-circuit.
            if ~gather(any(can)), continue; end

            m_si_can = m_si_col(can);
            m_sj_can = m_sj_col(can);
            if sign_dir == 1
                c_a_vec  = 0.5 * J ...
                    .* sqrt(s_val*(s_val+1) - m_si_can.*(m_si_can+1)) ...
                    .* sqrt(s_val*(s_val+1) - m_sj_can.*(m_sj_can-1));
                state_a_vec = super_reps_gpu(can) + powers_gpu(si) - powers_gpu(sj);
            else
                c_a_vec  = 0.5 * J ...
                    .* sqrt(s_val*(s_val+1) - m_si_can.*(m_si_can-1)) ...
                    .* sqrt(s_val*(s_val+1) - m_sj_can.*(m_sj_can+1));
                state_a_vec = super_reps_gpu(can) - powers_gpu(si) + powers_gpu(sj);
            end
            src_idx_can = int32(find(can));

            %% min_image_Ih_gpu + Schnack lookup (both GPU-native)
            [rep_a_vec, g_min_vec] = min_image_Ih_gpu(state_a_vec, group, s_val, perm_powers_gpu);
            t_idx_vec = query_lookup_schnack_gpu(lookup_gpu, rep_a_vec);
            in_basis  = t_idx_vec > 0;
            if ~gather(any(in_basis)), continue; end

            %% Filter: all on-device logical indexing, no host roundtrip.
            src_cells{end+1, 1} = src_idx_can(in_basis);    %#ok<AGROW>
            tgt_cells{end+1, 1} = t_idx_vec(in_basis);      %#ok<AGROW>
            g_cells{end+1,   1} = uint16(g_min_vec(in_basis)); %#ok<AGROW> 1..|G| fits uint16
            if c_is_const
                if gather(any(abs(c_a_vec - c_const_val) > 1e-9 * max(abs(c_const_val), 1)))
                    error('collect_clt_entries_Ih_gpu:cNotConst', ...
                          'Off-diagonal c expected constant (=0.5*J) for s=1/2.');
                end
            else
                c_cells{end+1, 1} = c_a_vec(in_basis);   %#ok<AGROW>
            end
        end
    end

    %% Concatenate on GPU and sort by tgt.
    if isempty(src_cells)
        all_src_flat = gpuArray(zeros(0, 1, 'int32'));
        all_tgt_flat = gpuArray(zeros(0, 1, 'int32'));
        all_g_flat   = gpuArray(zeros(0, 1, 'uint16'));
    else
        all_src_flat = vertcat(src_cells{:});
        all_tgt_flat = vertcat(tgt_cells{:});
        all_g_flat   = vertcat(g_cells{:});
    end
    clear src_cells tgt_cells g_cells;
    n_coll = double(gather(numel(all_src_flat)));

    [tgt_sorted, order] = sort(all_tgt_flat);
    src_sorted = all_src_flat(order);
    g_sorted   = all_g_flat(order);
    clear all_src_flat all_tgt_flat all_g_flat;

    % c_sorted: empty when constant (s=1/2), else sorted per-entry.
    if c_is_const
        c_sorted = zeros(0, 1);
    else
        if isempty(c_cells)
            all_c_flat = gpuArray(zeros(0, 1));
        else
            all_c_flat = vertcat(c_cells{:});
        end
        c_sorted = all_c_flat(order);
        clear all_c_flat;
    end
    clear c_cells order;

    %% Pack entries. Convention compatible with the host pipeline:
    %  super_reps host int64, diag_vals host double, src/tgt/g/c are gpuArrays.
    %  NB: entries.mi and entries.lookup_clt were removed (2026-06-03) to match
    %  the host collect -- both were stored but never read downstream.
    entries.super_reps = super_reps(:);              % host (small)
    entries.n_reps     = n_reps;
    entries.diag_vals  = gather(diag_vals_gpu);      % host (small)
    entries.src_sorted = src_sorted;
    entries.tgt_sorted = tgt_sorted;
    entries.g_sorted   = g_sorted;
    entries.c_sorted   = c_sorted;      % [] when c_is_const
    entries.c_is_const = c_is_const;
    entries.c_const    = c_const_val;
    entries.n_entries  = n_coll;
end
