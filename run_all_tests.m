function run_all_tests(mode)
%RUN_ALL_TESTS  One-command regression suite for the symmetry-adapted GPU-FTLM.
%
%   >> run_all_tests           % full suite (GPU entries SKIP without GPU/kernel)
%   >> run_all_tests('ci')     % CPU-only CI subset, see below
%
%   Runs a fast, representative set covering: generic irrep extraction, every
%   space-group provider (square C_4v/C_2v, kagome C_6v, triangular C_6v), the
%   end-to-end symmetry-ED-vs-full-ED correctness checks (sum rule + C(T)), and
%   the GPU block-Lanczos kernel regression. CPU-only checks run everywhere; the
%   GPU kernel regression runs only when a GPU + the compiled kernel are present.
%   Each entry passes iff it returns without throwing (the tests assert inside).
%
%   'ci' (hosted CI runners, e.g. GitHub Actions): drops the needs-GPU rows
%   entirely (instead of SKIP) and validate_dodecahedron20, whose independent
%   sparse-Lanczos reference (Sz=0, dim 184756) is too slow for hosted
%   runners. Needs only MATLAB + the mmap_file MEX (mex mmap_file.cpp) -- no
%   CUDA, no kernel build. See .github/workflows/ci.yml.
%
%   See also BUILD_ALL, SETUP_PATHS, TEST_PIPELINE_OPTS.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    if nargin < 1 || isempty(mode), mode = 'all'; end
    assert(ismember(lower(mode), {'all', 'ci'}), ...
        'run_all_tests: unknown mode ''%s'' (expected ''all'' or ''ci'').', mode);
    setup_paths();

    have_gpu = false;
    try, have_gpu = gpuDeviceCount > 0; catch, end
    have_kernel = exist('cuda_lanczos_clut_block_pg_Ih', 'file') == 3;

    %  {name, needs_gpu}
    suite = {
        'test_irreps_from_group',        false   % generic irrep extractor (I_h cross-check + sg)
        'test_realify_irreps',           false   % FS=+1 realification of space-group irreps
        'test_square_lattice_spacegroup',false   % square C_4v/C_2v provider axioms + bonds
        'test_kagome_spacegroup',        false   % kagome C_6v provider
        'test_triangular_spacegroup',    false   % triangular C_6v provider
        'validate_square_4x4_c4v',       false   % space-group ED vs full ED (N=16, CPU)
        'validate_generators_square4x4', false   % generators-only path == native provider + full ED (N=16, CPU)
        'validate_kagome12',             false   % kagome ED vs full ED (N=12, CPU)
        'validate_triangular12',         false   % triangular ED vs full ED (N=12, CPU)
        'validate_dodecahedron20',       false   % dodecahedron I_h provider vs sparse Lanczos (N=20, CPU)
        'validate_cuboctahedron12',      false   % cuboctahedron O_h provider vs full ED (N=12, CPU)
        'validate_spinflip12',           false   % spin-flip Z2 (M=0) == direct M=0 ED (kagome + square s=1)
        'validate_spinflip_driver12',    true    % use_spin_flip driver path: ED full sweep + GPU-kernel FTLM
        'test_external_bucket_sort',     false   % out-of-core external sort == in-RAM entry set (CPU)
        'test_mmap_file',                false   % read-only file mapping round-trips bit-identically (CPU)
        'test_gpu_sizing_invariants',    false   % VRAM-adaptive sizing safe on EVERY card size (CPU, fake-VRAM sweep)
        'test_gpu_alloc_inventory',      false   % every gpuArray creation classified vs the 2^31 cap (CPU, manifest)
        'test_gpu_int64_fallback',       true    % forward-compat digit paths (B200) bit-identical
        'test_perM_merge',               true    % one-job-per-M merged == full sweep (bit-identical)

        'test_pipeline_opts',            true    % GPU kernel + skeleton bit-identical regression
        'test_real_kernel',              true    % real FP32 kernel == complex at V_im=0 (bit gate, B swept to 8)
        'test_fp16_smoke',               true    % FP16 storage: default-env bit gate + u16*W envelope vs FP32
        'test_c4v_spmv',                 true    % OTF SpMV == CPU sparse H (Lanczos-free isolator, B=4)
        'test_tiled_spmv',               true    % R2 tiled SpMV == standard (FP32 reorder tolerance)
        'test_b2',                       true    % B2 entry-tiling (init_skel_b2) == resident (bit-identical)
        'test_split_v0',                 true    % chunked set_v0 upload == direct V0 (bit-identical AL/BE)
        'test_stream',                   true    % streaming-B2 rep-tiles (init_skel_stream) == resident (bit-identical)
        'test_stream_mmap',              true    % out-of-core mmap streaming == in-RAM (bit-identical)
        'test_r3_stream',                true    % R3 pinned-ring mmap streaming == sync (bit-identical, +prefix, +fp16)
        'test_stream_s1',                true    % s>=1 indexed-c streaming == resident (bit-identical)
        'test_stream_prefix',            true    % resident-prefix streaming (+Lever A tail-pin) == resident
        'test_device_sizing_matrix',     true    % emulated card sizes 4..180 GB x FP32/FP16: run, B pow2+monotone, bit-identical
        'test_keep_table',               true    % keep-table reuse (cleanup_keep_table) == fresh inits (bit-identical)
        'test_entries_on_disk',          true    % driver entries_on_disk path == in-RAM (end-to-end)
        'test_precompute_cache',         true    % precompute cache hit==miss bit-identical + stale-stamp reject
    };

    % CI subset: hosted runners have no GPU and are slow -- drop the
    % needs-GPU rows (a wall of SKIPs adds nothing) and the one expensive
    % CPU reference (validate_dodecahedron20). Everything else is identical
    % to the full suite, so a CI PASS covers the whole symmetry/provider/
    % ED-correctness layer.
    if strcmpi(mode, 'ci')
        drop  = cell2mat(suite(:, 2)) | strcmp(suite(:, 1), 'validate_dodecahedron20');
        suite = suite(~drop, :);
    end

    fprintf('\n================ run_all_tests ================\n');
    fprintf('mode: %s | GPU present: %d | kernel compiled: %d\n\n', ...
        lower(mode), have_gpu, have_kernel);

    n = size(suite, 1);
    status = cell(n, 1);
    for i = 1 : n
        name = suite{i, 1};  needs_gpu = suite{i, 2};
        if needs_gpu && ~(have_gpu && have_kernel)
            status{i} = 'SKIP (no GPU/kernel)';
            fprintf('[ .. ] %-32s  SKIP (no GPU/kernel)\n', name);
            continue;
        end
        fprintf('[ run] %-32s ...\n', name);
        % Run quietly but KEEP the captured output: on FAIL the test's own
        % fprintf trail (which case/config was running) is what localises a
        % node-side failure -- a bare evalc discards it with the error.
        ME = [];
        out = evalc('try, feval(name); catch ME, end');
        if isempty(ME)
            status{i} = 'PASS';
            fprintf('[PASS] %-32s\n', name);
        else
            status{i} = ['FAIL: ' ME.message];
            fprintf(2, '[FAIL] %-32s  %s\n', name, ME.message);
            lines = strsplit(out, '\n');
            lines = lines(max(1, numel(lines) - 30) : end);
            fprintf(2, '  ---- last output of %s ----\n', name);
            fprintf(2, '  | %s\n', lines{:});
            fprintf(2, '  ----------------------------\n');
        end
    end

    %% Summary.
    fprintf('\n================ SUMMARY ================\n');
    npass = 0; nfail = 0; nskip = 0;
    for i = 1 : n
        s = status{i};
        if strcmp(s, 'PASS'),            npass = npass + 1;
        elseif startsWith(s, 'SKIP'),    nskip = nskip + 1;
        else,                            nfail = nfail + 1;
        end
        fprintf('  %-32s %s\n', suite{i, 1}, s);
    end
    fprintf('----------------------------------------\n');
    fprintf('  %d passed, %d failed, %d skipped (of %d)\n', npass, nfail, nskip, n);
    if nfail > 0
        error('run_all_tests:failures', '%d test(s) FAILED.', nfail);
    end
    fprintf('  ALL RUN TESTS PASSED.\n');
end
