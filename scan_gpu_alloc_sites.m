function sites = scan_gpu_alloc_sites(write_manifest)
%SCAN_GPU_ALLOC_SITES  Every gpuArray-creation source line in production code.
%
%   SITES = SCAN_GPU_ALLOC_SITES() returns a sorted cellstr of
%   "file|normalized-source-line" entries for every line in the production
%   .m sources that creates a gpuArray. SCAN_GPU_ALLOC_SITES(true)
%   additionally (re)writes tests/gpu_alloc_manifest.txt.
%
%   PURPOSE (cap closure, 2026-07-03): the "2^31-1 elements per single
%   gpuArray" failure class struck four times because allocation formulas
%   were reviewed against planned systems instead of structural maxima.
%   Every creation site is now CLASSIFIED (SAFE / GUARDED / CHUNKED) in
%   docs/GPU_ALLOC_INVENTORY_2026-07-03.md, and TEST_GPU_ALLOC_INVENTORY
%   compares this scan against the committed manifest: ANY new/changed
%   creation site fails the suite until it has been classified against the
%   structural maxima (n_reps <= 2^31-257, d <= 12, B <= 8, N <= 64,
%   |G| <= 65535, n_entries unbounded -> never one gpuArray) and the
%   manifest is regenerated. The class cannot silently grow back.
%
%   Scope: ALL root-level .m files (inclusive by default -- new production
%   files are covered automatically); tests/benchmarks/examples/docs and
%   the symmetry_generation toolbox are out of scope.
%
%   See also TEST_GPU_ALLOC_INVENTORY, docs/GPU_ALLOC_INVENTORY_2026-07-03.md.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    if nargin < 1, write_manifest = false; end
    root = fileparts(mfilename('fullpath'));
    d = dir(fullfile(root, '*.m'));
    pats = {'gpuArray(', 'gpuArray.zeros', 'gpuArray.ones', 'gpuArray.randn', ...
            'gpuArray.rand(', 'gpuArray.inf', 'gpuArray.nan', '''gpuArray'''};
    sites = {};
    for k = 1 : numel(d)
        lines = strsplit(fileread(fullfile(root, d(k).name)), newline);
        for j = 1 : numel(lines)
            code = strip_comment(lines{j});
            if isempty(code), continue; end
            hit = false;
            for p = 1 : numel(pats)
                if contains(code, pats{p}), hit = true; break; end
            end
            if hit
                sites{end+1, 1} = sprintf('%s|%s', d(k).name, ...
                    regexprep(code, '\s+', ''));                   %#ok<AGROW>
            end
        end
    end
    sites = sort(sites);
    if write_manifest
        fid = fopen(fullfile(root, 'tests', 'gpu_alloc_manifest.txt'), 'w');
        fprintf(fid, '%s\n', sites{:});
        fclose(fid);
        fprintf('scan_gpu_alloc_sites: %d sites -> tests/gpu_alloc_manifest.txt\n', numel(sites));
    end
end


% ----------------------------------------------------------------
function code = strip_comment(line)
%   Cut the line at the first % that is OUTSIDE single-quoted strings
%   (naive but sufficient: MATLAB transpose-vs-quote ambiguity does not
%   matter for substring detection of gpuArray patterns).
    code = strtrim(line);
    if isempty(code) || code(1) == '%', code = ''; return; end
    in_str = false;
    for i = 1 : numel(code)
        c = code(i);
        if c == ''''
            in_str = ~in_str;
        elseif c == '%' && ~in_str
            code = strtrim(code(1 : i - 1));
            return;
        end
    end
end
