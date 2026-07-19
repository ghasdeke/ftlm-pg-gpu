function test_gpu_alloc_inventory()
%TEST_GPU_ALLOC_INVENTORY  No unclassified gpuArray creation may exist (CPU).
%
%   Compares SCAN_GPU_ALLOC_SITES against the committed manifest
%   tests/gpu_alloc_manifest.txt. Every entry in the manifest corresponds
%   to a site CLASSIFIED against the structural maxima in
%   docs/GPU_ALLOC_INVENTORY_2026-07-03.md (SAFE / GUARDED / CHUNKED).
%
%   If this test fails you added, moved or removed a gpuArray creation:
%     1. classify the new/changed site against the STRUCTURAL maxima
%        (n_reps <= 2^31-257, d <= 12, B <= 8, N <= 64, |G| <= 65535,
%        n_entries unbounded -> must be tiled/streamed/gated) and record
%        it in docs/GPU_ALLOC_INVENTORY_2026-07-03.md;
%     2. regenerate the manifest:  scan_gpu_alloc_sites(true)
%   This closes the 2^31-per-gpuArray failure class structurally: it
%   cannot grow back without a red suite.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    cur = scan_gpu_alloc_sites();
    mf  = fullfile(fileparts(which('scan_gpu_alloc_sites')), 'tests', 'gpu_alloc_manifest.txt');
    assert(exist(mf, 'file') == 2, 'manifest %s missing -- run scan_gpu_alloc_sites(true)', mf);
    ref = strsplit(strtrim(fileread(mf)), newline);
    ref = sort(strtrim(string(ref(:))));
    cur = sort(strtrim(string(cur(:))));

    added   = setdiff(cur, ref);
    removed = setdiff(ref, cur);
    if ~isempty(added)
        fprintf(2, '  NEW unclassified gpuArray creation site(s):\n');
        fprintf(2, '    + %s\n', added);
    end
    if ~isempty(removed)
        fprintf(2, '  Manifest entries with no matching source line (moved/removed):\n');
        fprintf(2, '    - %s\n', removed);
    end
    assert(isempty(added) && isempty(removed), ...
        ['gpuArray allocation inventory out of date (%d new, %d stale). ', ...
         'Classify the sites against the structural maxima (see docstring) ', ...
         'and regenerate: scan_gpu_alloc_sites(true).'], numel(added), numel(removed));
    fprintf('  PASS: all %d gpuArray creation sites classified (cap-closure manifest).\n', numel(cur));
end
