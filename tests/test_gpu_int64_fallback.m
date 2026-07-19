function test_gpu_int64_fallback()
%TEST_GPU_INT64_FALLBACK  The three GPU digit-decomposition paths agree.
%
%   Blackwell-class devices under CUDA forward compatibility (e.g. a B200 on
%   MATLAB R2024b) lack gpuArray int64 MOD -- see GPU_INT64_ARITH_OK. The
%   patched consumers (MIN_IMAGE_IH_GPU, IS_SUPER_REP_IH_GPU,
%   SCHNACK_RANK_GPU, the COLLECT_CLT_ENTRIES_IH_GPU mi block) select between
%     (1) the float-exact broadcast path  (n_total <= 2^52, ANY d_loc),
%     (2) the native gpuArray int64 loop  (where the device supports it),
%     (3) a host-decompose + upload fallback (forced by
%         FTLM_FORCE_NO_GPU_INT64=1 -- the forward-compat emulation).
%
%   WHAT THIS GATE ACTUALLY EXERCISES (bit-identical vs host references):
%     a. min_image / is_super_rep / schnack_rank unit checks at
%          s=1    (d_loc=3,  3^12 <= 2^52)         -- float path, the exact
%                 regime the pre-fix B200 s=1 suite failures hit;
%          s=9.5  (d_loc=20, 20^12 = 4.10e15, i.e. within 10%% of the 2^52
%                 gate)                             -- ADVERSARIAL float path:
%                 crafted states m*p +- 1 around the largest place values;
%          s=10   (d_loc=21, 21^12 = 7.36e15 > 2^52) -- int64 + host paths.
%        (s=9.5/10 are unphysical spins; only the integer digit machinery is
%        exercised, nothing downstream sees them.)
%     b. the GPU-NATIVE COLLECT (mi block + schnack lookup) at d_loc=21 on a
%        tiny M sector: entries + diag bit-identical to the host collect,
%        native AND forced.
%     c. an end-to-end icosa s=1 M=0 GPU-FTLM run, toggled env: at s=1 all
%        selectors take the FLOAT branch, so this is a float-path/determinism
%        check, NOT fallback coverage (that lives in a/b above).
%   NOT covered here: ENUMERATE_ALL_M_IH_GPU (currently has no callers in the
%   repo; its selector mirrors the ones under test).
%
%   See also GPU_INT64_ARITH_OK, MIN_IMAGE_IH_GPU, IS_SUPER_REP_IH_GPU,
%            SCHNACK_RANK_GPU, COLLECT_CLT_ENTRIES_IH_GPU.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    if gpuDeviceCount < 1
        fprintf('test_gpu_int64_fallback: SKIP (no GPU)\n');
        return;
    end
    % Restore the CALLER's env value on exit (run_all_tests may itself run
    % under a global FTLM_FORCE_NO_GPU_INT64=1 emulation -- keep it intact).
    prev_env = getenv('FTLM_FORCE_NO_GPU_INT64');
    cleanup = onCleanup(@() setenv('FTLM_FORCE_NO_GPU_INT64', prev_env));  %#ok<NASGU>
    group = icosahedron_Ih_full();
    bonds = adjacency_icosahedron_Ih();
    N = 12;

    %% ---- (a) digit-path equivalence: min_image / is_super_rep / schnack ----
    cfgs = [struct('s', 1.0,  'label', 'd=3  (n_total <= 2^52, float path)'), ...
            struct('s', 9.5,  'label', 'd=20 (within 10%% of the 2^52 gate, float path)'), ...
            struct('s', 10.0, 'label', 'd=21 (n_total  > 2^52, int64/host path)')];
    for c = cfgs
        s_val = c.s;
        d_loc = round(2*s_val + 1);
        n_total = double(d_loc)^N;
        rng(20260701);
        st_h = unique([crafted_states(d_loc, N); ...
                       int64(floor(rand(20000, 1) * (n_total - 1)))]);

        % Host reference.
        [rep_h, g_h] = min_image_Ih(st_h, group, s_val);

        % GPU, current-env selection (native machine: float or int64 loop;
        % under a global emulation this already runs the fallback).
        [rep_g, g_g] = min_image_Ih_gpu(gpuArray(st_h), group, s_val);
        ism_g = is_super_rep_Ih_gpu(gpuArray(st_h), group, s_val);

        % GPU, forced forward-compat fallback (the B200 code path).
        setenv('FTLM_FORCE_NO_GPU_INT64', '1');
        [rep_f, g_f] = min_image_Ih_gpu(gpuArray(st_h), group, s_val);
        ism_f = is_super_rep_Ih_gpu(gpuArray(st_h), group, s_val);
        setenv('FTLM_FORCE_NO_GPU_INT64', prev_env);

        assert(isequal(gather(rep_g), rep_h) && isequal(gather(g_g), g_h), ...
            'min_image_Ih_gpu (native) differs from min_image_Ih [%s]', c.label);
        assert(isequal(gather(rep_f), rep_h) && isequal(gather(g_f), g_h), ...
            'min_image_Ih_gpu (forced fallback) differs from min_image_Ih [%s]', c.label);
        assert(isequal(gather(ism_g), rep_h == st_h), ...
            'is_super_rep_Ih_gpu (native) wrong [%s]', c.label);
        assert(isequal(gather(ism_f), rep_h == st_h), ...
            'is_super_rep_Ih_gpu (forced fallback) wrong [%s]', c.label);

        % Schnack ranking on one digit-sum sector of the sample.
        A_all = zeros(numel(st_h), 1);
        tmp = st_h;
        for k = 1:N
            dg = mod(tmp, int64(d_loc));
            A_all = A_all + double(dg);
            tmp = (tmp - dg) / int64(d_loc);
        end
        A_t = mode(A_all);
        st_sec = sort(st_h(A_all == A_t));
        [~, D_cum] = build_D_table(N, d_loc - 1, A_t);
        rk_h = schnack_rank(st_sec, D_cum, N, d_loc - 1, d_loc, A_t);
        rk_g = schnack_rank_gpu(gpuArray(st_sec), gpuArray(D_cum), N, d_loc - 1, d_loc, A_t);
        setenv('FTLM_FORCE_NO_GPU_INT64', '1');
        rk_f = schnack_rank_gpu(gpuArray(st_sec), gpuArray(D_cum), N, d_loc - 1, d_loc, A_t);
        setenv('FTLM_FORCE_NO_GPU_INT64', prev_env);
        assert(isequal(gather(rk_g), rk_h), 'schnack_rank_gpu (native) differs [%s]', c.label);
        assert(isequal(gather(rk_f), rk_h), 'schnack_rank_gpu (forced fallback) differs [%s]', c.label);

        fprintf('  [%s] min_image / is_super_rep / schnack_rank: all paths bit-identical (%d states)\n', ...
            c.label, numel(st_h));
    end

    %% ---- (b) gpu-native collect (mi block + schnack lookup) at d=21 --------
    %  Tiny genuine sector: s=10, M=118 (digit sum 238 -> ~4 orbits). This
    %  exercises the >2^52 branch of the COLLECT_CLT_ENTRIES_IH_GPU mi block
    %  plus SCHNACK_RANK_GPU/MIN_IMAGE_IH_GPU inside it, native and forced,
    %  against the host collect.
    s_val = 10.0;
    cacheX = enumerate_M_orbits_Ih_gpu(s_val, 118, group);
    assert(numel(cacheX.super_reps) >= 2, 'collect unit: unexpected empty s=10 M=118 sector');
    e_h = collect_clt_entries_Ih(cacheX.super_reps, bonds, s_val, 1.0, group, 'schnack', 'host');
    e_g = collect_clt_entries_Ih_gpu(cacheX.super_reps, bonds, s_val, 1.0, group);
    setenv('FTLM_FORCE_NO_GPU_INT64', '1');
    e_f = collect_clt_entries_Ih_gpu(cacheX.super_reps, bonds, s_val, 1.0, group);
    setenv('FTLM_FORCE_NO_GPU_INT64', prev_env);
    % NB: the two collects emit DIFFERENT (both downstream-valid) c schemas --
    % host = c-index layout (c_is_indexed/c_table/c_idx, since the 2026-06
    % compaction), gpu-native = legacy full c_sorted vector; the skeleton
    % builder accepts either (BUILD_ENTRY_SKELETON_IH line ~48/105). Compare
    % the DECODED per-entry c values, which must agree bit-identically.
    pack = @(e) {gather(e.src_sorted), gather(e.tgt_sorted), gather(e.g_sorted), ...
                 c_values(e), gather(e.diag_vals)};
    ref = pack(e_h);
    assert(isequal(pack(e_g), ref), 'gpu-native collect (native) differs from host at d=21');
    assert(isequal(pack(e_f), ref), 'gpu-native collect (forced fallback) differs from host at d=21');
    fprintf('  [collect d=21] gpu-native mi/schnack vs host: native + forced bit-identical (%d reps, %d entries)\n', ...
        e_h.n_reps, e_h.n_entries);

    %% ---- (c) end-to-end s=1 driver run, env toggled ------------------------
    %  At s=1 every selector takes the FLOAT branch (n_total = 3^12 <= 2^52),
    %  so this checks float-path correctness + run-to-run determinism of the
    %  full driver; the fallback branches are covered by (a)/(b) above.
    if exist('cuda_lanczos_clut_block_pg_Ih', 'file') ~= 3
        fprintf('  (kernel MEX not built -- end-to-end part skipped)\n');
        fprintf('PASS: GPU int64 fallback paths bit-identical (digit machinery + collect).\n');
        return;
    end
    od = tempname;  mkdir(od);
    cu2 = onCleanup(@() rmdirq(od));  %#ok<NASGU>
    f = fullfile(od, 'inp.m');
    fid = fopen(f, 'w');
    fprintf(fid, 'geometry=''icosahedron'';\ns_val=1.0;\nJ=1.0;\nR=2;\nM_lz=20;\n');
    fprintf(fid, 'only_M0=true;\ned_thresh=0;\nlookup_method=''bitmap'';\n');
    fprintf(fid, 'T_range=logspace(-1,1,20);\noutput_dir=''%s'';\n', strrep(od, '\', '/'));
    fclose(fid);

    evalc('ftlm_observables_pg_gpu_Ih(f)');
    r1 = load(fullfile(od, 'ftlm_pg_gpu_Ih_icos_s1.mat'));
    setenv('FTLM_FORCE_NO_GPU_INT64', '1');
    evalc('ftlm_observables_pg_gpu_Ih(f)');
    setenv('FTLM_FORCE_NO_GPU_INT64', prev_env);
    r2 = load(fullfile(od, 'ftlm_pg_gpu_Ih_icos_s1.mat'));
    assert(isequal(r1.all_E, r2.all_E) && isequal(r1.all_w, r2.all_w), ...
        'end-to-end s=1 driver run NOT bit-identical across the env toggle');
    fprintf('  [end-to-end] icosa s=1 M=0 GPU FTLM (float path): bit-identical across env toggle (sum_w=%.8g)\n', ...
        sum(r2.all_w));
    fprintf('PASS: GPU int64 fallback paths bit-identical (units + collect + end-to-end float).\n');
end

% ----------------------------------------------------------------
function c = c_values(e)
%   Per-entry c coefficients, decoded across the two entry schemas: the host
%   collect emits the c-INDEX layout (c_is_indexed + uint8 c_idx into
%   c_table; c_sorted empty) since the 2026-06 compaction, while the
%   gpu-native collect emits the legacy full c_sorted vector. Both are
%   accepted downstream; their decoded values must agree bit-identically.
    if isfield(e, 'c_is_indexed') && e.c_is_indexed
        tbl = double(gather(e.c_table(:)));
        c   = tbl(double(gather(e.c_idx(:))));       % 1-based uint8 index
    elseif isfield(e, 'c_sorted') && ~isempty(e.c_sorted)
        c = double(gather(e.c_sorted(:)));
    else                                             % s=1/2: constant c
        c = repmat(double(e.c_const), double(e.n_entries), 1);
    end
end

function st = crafted_states(d_loc, N)
%   Adversarial states for the float-path proof: values adjacent to multiples
%   of the largest place values d_loc^k (where floor(x./p) rounding would
%   first go wrong), plus the range edges.
    n_total = double(d_loc)^N;                 % <= 21^12 < 2^53, exact double
    st = [int64(0); int64(1); int64(n_total) - 1];
    for k = N-1 : -1 : N-3
        p = int64(d_loc)^int64(k);
        for m = [1, 2, d_loc - 1]
            base = int64(m) * p;
            st = [st; base - 1; base; base + 1];     %#ok<AGROW>
        end
    end
    st = unique(st(st >= 0 & st < int64(n_total)));
end

function rmdirq(d)
    try, rmdir(d, 's'); catch, end
end
