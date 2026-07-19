function test_entries_on_disk()
%TEST_ENTRIES_ON_DISK  End-to-end driver test of the opt-in out-of-core path:
%   ftlm_observables_pg_gpu_Ih with entries_on_disk=true spills the per-M entry
%   table to an NVMe file + memory-maps it (resident table -> ~0 host RAM) and
%   streams every irrep from the mapping. Must reproduce the in-RAM run: the
%   Sz=0 sum rule stays exact and the spectrum matches (the streaming SpMV is
%   bit-identical, so with the same seed/B the FTLM Ritz values coincide).
%
%   Uses the s=1/2 icosahedron M=0 (tiny, all 10 I_h irreps, ed_thresh=0 -> GPU
%   FTLM exercises the streaming path).
%
%   See also SPILL_ENTRIES_MMAP, MMAP_FILE, TEST_STREAM_MMAP.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    assert(exist('mmap_file', 'file') == 3, 'mmap_file MEX not built (build_all).');
    if gpuDeviceCount < 1 || exist('cuda_lanczos_clut_block_pg_Ih', 'file') ~= 3
        fprintf('test_entries_on_disk: SKIP (no GPU / kernel)\n'); return;
    end
    od = tempname;  mkdir(od);
    cu = onCleanup(@() rmdirq(od));

    % s=1/2: the original 6 B/entry [src][g] layout (constant c).
    [E0, sw0, expw] = run_case(od, false, 0.5);
    [E1, sw1, ~   ] = run_case(od, true,  0.5);
    check_pair('s=1/2', E0, sw0, E1, sw1, expw);

    % s=1: the companion-A 7 B/entry [src][g][c_idx] layout (indexed c streamed
    % from the mapping via clt.mmap_cidx).
    [E2, sw2, expw1] = run_case(od, false, 1.0);
    [E3, sw3, ~    ] = run_case(od, true,  1.0);
    check_pair('s=1  ', E2, sw2, E3, sw3, expw1);

    % Diagonal-only sector (n_entries == 0): the fully polarised M = M_max has
    % no off-diagonal entries -> a ZERO-length sorted file, which can NOT be
    % memory-mapped (POSIX mmap EINVAL / Windows CreateFileMapping failure).
    % The on-disk eskel must fall back to the resident empty-entry skeleton.
    f = fullfile(od, 'inp_mmax.m');
    fid = fopen(f, 'w');
    fprintf(fid, 'geometry=''icosahedron'';\ns_val=0.5;\nJ=1.0;\nR=2;\nM_lz=10;\n');
    fprintf(fid, 'M_sectors=6;\ned_thresh=0;\nlookup_method=''bitmap'';\n');
    fprintf(fid, 'T_range=logspace(-1,1,20);\noutput_dir=''%s'';\nentries_on_disk=true;\n', ...
            strrep(od, '\', '/'));
    fclose(fid);
    evalc('ftlm_observables_pg_gpu_Ih(f)');
    r = load(fullfile(od, 'ftlm_pg_gpu_Ih_icos_s1o2.mat'));
    sw_mm = sum(r.all_w);
    fprintf('  [M=M_max] on-disk diagonal-only sector: sum_w=%.8g (expected 2)\n', sw_mm);
    assert(abs(sw_mm - 2) < 1e-10, 'M=M_max on-disk sum rule off: %.6g (expected 2)', sw_mm);

    % Shared-table mode (multi-GPU enabler): entries_on_disk + precompute_cache
    % -- the sorted entry file becomes a persistent CACHE artifact (it now
    % lives in precompute_dir and survives the run); a rerun loads the cache,
    % skips enumerate+collect entirely, and memory-maps the SAME file. The two
    % runs must be bit-identical (same seeds, same streamed bytes). s=1 also
    % covers the mmapped c_idx block in this mode.
    od2 = fullfile(od, 'shared');  mkdir(od2);
    f2 = fullfile(od, 'inp_shared.m');
    fid = fopen(f2, 'w');
    fprintf(fid, 'geometry=''icosahedron'';\ns_val=1.0;\nJ=1.0;\nR=6;\nM_lz=40;\n');
    fprintf(fid, 'only_M0=true;\ned_thresh=0;\nlookup_method=''bitmap'';\n');
    fprintf(fid, 'T_range=logspace(-1,1,20);\noutput_dir=''%s'';\n', strrep(od2, '\', '/'));
    fprintf(fid, 'entries_on_disk=true;\nprecompute_cache=true;\n');
    fclose(fid);
    log1 = evalc('ftlm_observables_pg_gpu_Ih(f2)');
    r1 = load(fullfile(od2, 'ftlm_pg_gpu_Ih_icos_s1.mat'));
    assert(contains(log1, '[precompute-cache] wrote'), 'shared-od: cache not written');
    assert(~isempty(dir(fullfile(od2, 'ondisk_*', 'entries_sorted.bin'))), ...
        'shared-od: sorted entry file did not persist with the cache');
    log2 = evalc('ftlm_observables_pg_gpu_Ih(f2)');
    r2 = load(fullfile(od2, 'ftlm_pg_gpu_Ih_icos_s1.mat'));
    assert(contains(log2, '[precompute-cache] loaded'), 'shared-od rerun did not load the cache');
    assert(max(abs(r1.all_E - r2.all_E)) == 0 && max(abs(r1.all_w - r2.all_w)) == 0, ...
        'shared-od rerun not bit-identical to the first run');
    fprintf('  [shared-od] cache-backed on-disk entries: rerun bit-identical, sorted file persisted\n');

    % the spilled files must have been cleaned up (per-run scratch mode only;
    % the shared-od files above live in od2 and persist by design)
    assert(isempty(dir(fullfile(od, 'entries_*.bin'))), 'spilled entry file not deleted');
    fprintf('PASS: entries_on_disk driver path == in-RAM (s=1/2 + s=1 + empty M=M_max + shared-od reuse).\n');
end

% ----------------------------------------------------------------
function check_pair(label, E0, sw0, E1, sw1, expw)
    dE = max(abs(sort(E0(:)) - sort(E1(:))));
    fprintf('  [%s] in-RAM  sum_w=%.8g (expected %.0f)\n', label, sw0, expw);
    fprintf('  [%s] on-disk sum_w=%.8g\n', label, sw1);
    fprintf('  [%s] spectra max|dE| (in-RAM vs on-disk) = %.3e\n', label, dE);
    assert(abs(sw0 - expw) < 1e-8 * expw, '%s in-RAM sum-rule off: %.6g vs %.0f', label, sw0, expw);
    assert(abs(sw1 - expw) < 1e-8 * expw, '%s on-disk sum-rule off: %.6g vs %.0f', label, sw1, expw);
    assert(dE < 1e-2, '%s on-disk spectrum differs from in-RAM (max|dE|=%.2e)', label, dE);
end

function [E, sw, expw] = run_case(od, ondisk, s_val)
    f = fullfile(od, 'inp.m');
    fid = fopen(f, 'w');
    fprintf(fid, 'geometry=''icosahedron'';\ns_val=%g;\nJ=1.0;\nR=6;\nM_lz=40;\n', s_val);
    fprintf(fid, 'only_M0=true;\ned_thresh=0;\nlookup_method=''bitmap'';\n');
    fprintf(fid, 'T_range=logspace(-1,1,20);\noutput_dir=''%s'';\n', strrep(od, '\', '/'));
    if ondisk, fprintf(fid, 'entries_on_disk=true;\n'); end
    fclose(fid);
    evalc('ftlm_observables_pg_gpu_Ih(f)');     % run quietly
    if abs(s_val - 0.5) < 1e-12
        r = load(fullfile(od, 'ftlm_pg_gpu_Ih_icos_s1o2.mat'));
        expw = nchoosek(12, 6);                  % dim(M=0) = C(12,6) = 924
    else
        r = load(fullfile(od, sprintf('ftlm_pg_gpu_Ih_icos_s%d.mat', round(s_val))));
        dp = 1;                                  % dim(M=0) = central multinomial coeff
        for k = 1:12, dp = conv(dp, ones(1, round(2*s_val) + 1)); end
        expw = dp(round(12 * s_val) + 1);
    end
    E = r.all_E;  sw = sum(r.all_w);
end

function rmdirq(d)
    try, rmdir(d, 's'); catch, end
end
