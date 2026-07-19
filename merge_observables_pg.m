function [C_T, chi_T, Z_eff, S_merged] = merge_observables_pg(files, T_range, save_as)
%MERGE_OBSERVABLES_PG  Combine per-M-sector results files into total observables.
%
%   [C_T, CHI_T, Z_EFF] = MERGE_OBSERVABLES_PG(FILES, T_RANGE)
%   [...            ]   = MERGE_OBSERVABLES_PG(FILES, T_RANGE, SAVE_AS)
%   [..., S_MERGED]     additionally returns the merged data struct.
%
%   FILES: cell array of results-.mat paths, OR a char glob pattern, e.g.
%       merge_observables_pg('runs_icosido_R24_perM/M*/ftlm_pg_gpu_Ih_*.mat', T)
%
%   Concatenates the raw FTLM data (all_E / all_w / all_M) of one-M-per-job
%   runs and evaluates the TOTAL C(T), chi(T), Z_eff(T). The negative-M
%   mirror sectors are already included: the drivers fold the mult_M
%   (M <-> -M) factor into the stored weights, so merging is pure
%   concatenation -- no re-weighting.
%
%   Guards: asserts consistent geometry tag / s_val / J / n_total across
%   files and DISJOINT M sets (a duplicated sector would double-count).
%   When the merged M set covers the full sweep 0..M_max, the FTLM sum
%   rule sum(w) == n_total is checked (and reported) -- the same
%   correctness gate the single-job full sweep prints.
%
%   SAVE_AS (optional): writes a merged results file (raw data + curves +
%   provenance) that REEVAL_OBSERVABLES_PG / OBSERVABLES_PER_M accept.
%
%   See also OBSERVABLES_PER_M, REEVAL_OBSERVABLES_PG, COMPUTE_OBSERVABLES_PG.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    if nargin < 3, save_as = ''; end

    %% Resolve the file list.
    if ischar(files) || isstring(files)
        d = dir(char(files));
        assert(~isempty(d), 'merge_observables_pg: no files match "%s".', char(files));
        files = arrayfun(@(x) fullfile(x.folder, x.name), d, 'UniformOutput', false);
    end
    files = files(:);
    n_f = numel(files);
    assert(n_f >= 1, 'merge_observables_pg: empty file list.');

    %% Load + consistency guards + concatenation.
    all_E = [];  all_w = [];  all_M = [];
    M_seen = [];  src = cell(n_f, 1);
    ref = load(files{1}, 's_val', 'J', 'n_total_save', 'M_max');
    for k = 1 : n_f
        S = load(files{k});
        for f = {'all_E', 'all_w', 'all_M'}
            assert(isfield(S, f{1}), ...
                'merge_observables_pg: %s has no raw field ''%s''.', files{k}, f{1});
        end
        assert(S.s_val == ref.s_val && S.J == ref.J ...
               && S.n_total_save == ref.n_total_save && S.M_max == ref.M_max, ...
            'merge_observables_pg: %s belongs to a DIFFERENT system than %s.', ...
            files{k}, files{1});
        Mk = unique(double(S.all_M(:)));
        dup = intersect(M_seen, Mk);
        if ~isempty(dup)   % NOT assert: assert evaluates dup(1) even when empty
            error('merge_observables_pg:dupM', ...
                ['M=%g appears in more than one file -- merging would ', ...
                 'double-count that sector.'], dup(1));
        end
        M_seen = [M_seen; Mk];                     %#ok<AGROW>
        all_E = [all_E; double(S.all_E(:))];       %#ok<AGROW>
        all_w = [all_w; double(S.all_w(:))];       %#ok<AGROW>
        all_M = [all_M; double(S.all_M(:))];       %#ok<AGROW>
        src{k} = files{k};
    end

    %% Sum rule: full coverage 0..M_max -> sum(w) must equal n_total.
    M_seen = sort(M_seen);
    full_sweep = isequal(M_seen(:)', 0 : double(ref.M_max));
    sum_w = sum(all_w);
    if full_sweep
        rel = abs(sum_w - ref.n_total_save) / ref.n_total_save;
        fprintf(['merge_observables_pg: %d files, FULL sweep M=0..%d. Sum rule: ', ...
                 'sum(w) = %.6g, n_total = %.6g, rel.err = %.2e\n'], ...
                n_f, ref.M_max, sum_w, ref.n_total_save, rel);
        assert(rel < 1e-10, 'merge_observables_pg: sum rule violated (%.2e).', rel);
    else
        fprintf(['merge_observables_pg: %d files, PARTIAL coverage (M in {%s}). ', ...
                 'sum(w) = %.6g (no full-sweep sum rule).\n'], ...
                n_f, strjoin(string(M_seen), ','), sum_w);
    end

    %% Total observables + optional merged save.
    T_range = double(T_range(:)');
    [C_T, chi_T, Z_eff] = compute_observables_pg(all_E, all_w, all_M, T_range);

    S_merged = struct('all_E', all_E, 'all_w', all_w, 'all_M', all_M, ...
        's_val', ref.s_val, 'J', ref.J, 'n_total_save', ref.n_total_save, ...
        'M_max', ref.M_max, 'T_range', T_range, 'C_T', C_T, 'chi_T', chi_T, ...
        'Z_eff', Z_eff, 'merged_from', {src}, 'full_sweep', full_sweep);
    if ~isempty(save_as)
        save(save_as, '-struct', 'S_merged', '-v7.3');
        fprintf('merge_observables_pg: merged results written to %s\n', save_as);
    end
end
