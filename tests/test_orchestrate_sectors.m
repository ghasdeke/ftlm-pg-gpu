function test_orchestrate_sectors()
%TEST_ORCHESTRATE_SECTORS  Sector-parallel orchestration == one full run, in
%   bookkeeping: 2 process workers split the icosahedron s=1/2 M=0 block list
%   (shared precompute cache, per-worker output dirs), and the MERGED result
%   must cover all 10 blocks with the EXACT sum rule sum_w = dim(M=0) = 924.
%   E0 agrees with a single full run within FTLM statistics (the per-block
%   seeds differ between split and full runs by design -- the seed is keyed to
%   the list POSITION; see the seed note in FTLM_ORCHESTRATE_SECTORS).
%
%   On a single-GPU machine both workers share device 1 (contention only costs
%   wall time, never correctness).
%
%   NOT in RUN_ALL_TESTS (deliberate): parpool startup under `matlab -batch`
%   crashed once transiently on this machine (0xc0000409, no output) -- a hard
%   crash inside the suite would abort the WHOLE suite run, so this stays a
%   standalone gate. Run it explicitly after touching the orchestrator/driver
%   orchestration options, and once on the cluster node:
%       matlab -batch "setup_paths; test_orchestrate_sectors"
%
%   See also FTLM_ORCHESTRATE_SECTORS, FTLM_OBSERVABLES_PG_GPU_IH.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    if gpuDeviceCount < 1 || exist('cuda_lanczos_clut_block_pg_Ih', 'file') ~= 3
        fprintf('test_orchestrate_sectors: SKIP (no GPU / kernel)\n'); return;
    end
    od = tempname;  mkdir(od);
    cu = onCleanup(@() rmdirq(od));

    f = fullfile(od, 'inp.m');
    fid = fopen(f, 'w');
    fprintf(fid, 'geometry=''icosahedron'';\ns_val=0.5;\nJ=1.0;\nR=4;\nM_lz=30;\n');
    fprintf(fid, 'only_M0=true;\ned_thresh=0;\nlookup_method=''bitmap'';\n');
    fprintf(fid, 'T_range=logspace(-1,1,20);\noutput_dir=''%s'';\n', strrep(od, '\', '/'));
    fclose(fid);

    % Reference: one full single-process run (same input, own subdir).
    fr = fullfile(od, 'inp_ref.m');
    fid = fopen(fr, 'w');
    fprintf(fid, '%s\noutput_dir=''%s'';\n', fileread(f), strrep(fullfile(od, 'ref'), '\', '/'));
    fclose(fid);
    evalc('ftlm_observables_pg_gpu_Ih(fr)');
    rr = load(fullfile(od, 'ref', 'ftlm_pg_gpu_Ih_icos_s1o2.mat'));

    % Orchestrated: 2 process workers.
    merged = ftlm_orchestrate_sectors(f, 2);

    expw = nchoosek(12, 6);                                   % dim(M=0) = 924
    dE0  = abs(min(merged.all_E) - min(rr.all_E)) / max(1, abs(min(rr.all_E)));
    fprintf('  merged sum_w=%.8g (expected %d, rel.err %.2e), blocks=%d, E0 rel diff=%.2e\n', ...
            merged.sum_w, expw, merged.sum_rule_relerr, numel(merged.sector_G), dE0);
    assert(merged.sum_rule_relerr < 1e-10, 'merged sum rule violated: %.3e', merged.sum_rule_relerr);
    assert(numel(merged.sector_G) == 10, 'expected 10 merged blocks, got %d', numel(merged.sector_G));
    assert(abs(sum(rr.all_w) - merged.sum_w) < 1e-8 * expw, 'merged vs full-run sum_w differ');
    assert(dE0 < 1e-3, 'merged E0 != full-run E0 beyond FTLM statistics: %.2e', dE0);

    %% Shared-table variant (the dodec-s=3/2 multi-GPU mode): entries_on_disk +
    %  the orchestrator's shared precompute_dir -> phase 0 writes ONE sorted
    %  entry file; BOTH workers memory-map the SAME file (OS page cache shares
    %  the physical pages). Global sum rule must again be exact.
    od3 = fullfile(od, 'shared');  mkdir(od3);
    f3 = fullfile(od3, 'inp.m');
    fid = fopen(f3, 'w');
    fprintf(fid, '%s\noutput_dir=''%s'';\nentries_on_disk=true;\nprecompute_cache=true;\n', ...
            fileread(f), strrep(od3, '\', '/'));
    fclose(fid);
    merged3 = ftlm_orchestrate_sectors(f3, 2);
    n_sorted = numel(dir(fullfile(od3, 'orch_precompute', 'ondisk_*', 'entries_sorted.bin')));
    fprintf('  [shared-od] merged sum_w=%.8g (rel.err %.2e), shared sorted files=%d\n', ...
            merged3.sum_w, merged3.sum_rule_relerr, n_sorted);
    assert(merged3.sum_rule_relerr < 1e-10, 'shared-od merged sum rule violated');
    assert(numel(merged3.sector_G) == 10, 'shared-od: expected 10 merged blocks');
    assert(n_sorted == 1, 'expected ONE shared sorted entry file, found %d', n_sorted);
    fprintf('PASS: 2-worker orchestration (in-RAM + shared on-disk table); merged sum rules exact; E0 consistent.\n');
end

function rmdirq(d)
    try, rmdir(d, 's'); catch, end
end
