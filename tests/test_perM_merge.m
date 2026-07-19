function test_perM_merge()
%TEST_PERM_MERGE  One-job-per-M runs merged == single full-sweep run, EXACTLY.
%
%   The per-sector FTLM seeds are deterministic in the PHYSICAL (M, irrep)
%   pair (seed = 8e6 + M*1e4 + ig*100), so running each M sector in its own
%   driver call and concatenating the raw data via MERGE_OBSERVABLES_PG must
%   reproduce the single full-sweep run BIT-IDENTICALLY: same (E, w, M)
%   triples, same C/chi/Z_eff on any grid, same sum rule. This is the
%   correctness gate for the one-SLURM-job-per-M workflow.
%
%   Also covered: the double-count guard (merging a sector twice must
%   error) and an OBSERVABLES_PER_M smoke check.
%
%   System: icosahedron s=1/2 (M_max = 6, 7 sector runs + 1 full run, GPU).
%
%   See also MERGE_OBSERVABLES_PG, OBSERVABLES_PER_M, FTLM_OBSERVABLES_PG_GPU_IH.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    wd = fullfile(tempdir, 'ftlm_perm_merge_test');
    if exist(wd, 'dir'), rmdir(wd, 's'); end
    mkdir(wd);
    cu = onCleanup(@() rmdirq(wd));

    common = { ...
        'geometry  = ''icosahedron'';'
        's_val     = 0.5;'
        'J         = 1.0;'
        'R         = 4;'
        'M_lz      = 40;'
        'ed_thresh = 0;'
        'T_range   = logspace(-1, 1, 30);'};

    %% Full sweep in ONE run.
    dir_full = fullfile(wd, 'full');
    write_deck(fullfile(wd, 'in_full.m'), [common; {sprintf('output_dir = ''%s'';', esc(dir_full))}]);
    evalc('ftlm_observables_pg_gpu_Ih(fullfile(wd, ''in_full.m''))');
    F = load(fullfile(dir_full, 'ftlm_pg_gpu_Ih_icos_s1o2.mat'));

    %% One run per M sector (M = 0..6).
    files = cell(7, 1);
    for m = 0 : 6
        dm = fullfile(wd, sprintf('M%d', m));
        write_deck(fullfile(wd, sprintf('in_M%d.m', m)), ...
            [common; {sprintf('M_sectors = %d;', m); sprintf('output_dir = ''%s'';', esc(dm))}]);
        evalc('ftlm_observables_pg_gpu_Ih(fullfile(wd, sprintf(''in_M%d.m'', m)))');
        files{m + 1} = fullfile(dm, 'ftlm_pg_gpu_Ih_icos_s1o2.mat');
    end

    %% Merge and compare BIT-IDENTICALLY against the full run.
    T2 = linspace(0.05, 10, 137);
    out = evalc('[C2, chi2, Z2, SM] = merge_observables_pg(files, T2);');
    assert(contains(out, 'FULL sweep'), 'merge did not detect full coverage');
    a = sortrows([SM.all_E, SM.all_w, SM.all_M]);
    b = sortrows([F.all_E(:), F.all_w(:), F.all_M(:)]);
    assert(isequal(a, b), 'merged raw (E,w,M) triples differ from the full-sweep run');
    [Cf, chif, Zf] = compute_observables_pg(F.all_E, F.all_w, F.all_M, T2);
    assert(isequal(C2, Cf) && isequal(chi2, chif) && isequal(Z2, Zf), ...
        'merged observables differ from the full-sweep run');
    fprintf('  per-M merge == full sweep: %d triples bit-identical, curves identical\n', size(a, 1));

    %% Double-count guard: the same sector twice must ERROR.
    threw = false;
    try
        evalc('merge_observables_pg([files; files(1)], T2);');
    catch
        threw = true;
    end
    assert(threw, 'duplicate-M merge did not error (double counting!)');

    %% observables_per_M smoke: sector rows sane, M list complete.
    [Ml, C_M, chi_M, Z_M] = observables_per_M(files{1}, T2);   %#ok<ASGLU>
    assert(isequal(Ml, 0) && all(isfinite(C_M(:))) && all(Z_M(:) > 0));
    [Ml2, ~, chi_M2] = observables_per_M(fullfile(dir_full, 'ftlm_pg_gpu_Ih_icos_s1o2.mat'), T2);
    assert(isequal(Ml2(:)', 0:6), 'per-M split of the full run misses sectors');
    assert(max(abs(chi_M2(1, :))) == 0, 'M=0 sector must have zero chi contribution');
    fprintf('  PASS: one-job-per-M workflow reproduces the full sweep exactly.\n');
end


% ----------------------------------------------------------------
function write_deck(path, lines)
    fid = fopen(path, 'w');
    fprintf(fid, '%s\n', lines{:});
    fclose(fid);
end

function s = esc(s)
    % Only quotes need doubling: the path is passed as a sprintf ARGUMENT
    % (not format), and MATLAB single-quoted strings do not escape backslash.
    s = strrep(s, '''', '''''');
end

function rmdirq(d)
    try, rmdir(d, 's'); catch, end
end
