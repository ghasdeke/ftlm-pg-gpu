function merged = ftlm_orchestrate_sectors(input_file, n_workers, gpu_ids)
%FTLM_ORCHESTRATE_SECTORS  Sector-parallel multi-GPU FTLM (companion project B).
%
%   MERGED = FTLM_ORCHESTRATE_SECTORS(INPUT_FILE, N_WORKERS) splits the irrep
%   block list of a normal FTLM_OBSERVABLES_PG_GPU_IH input across N_WORKERS
%   parallel PROCESS workers (one GPU each) and merges the partial spectra into
%   one result with the global sum rule checked. NO kernel change is involved:
%   the MEX holds per-PROCESS statics (1 GPU per process), which is exactly the
%   parpool('Processes') worker model.
%
%   MERGED = FTLM_ORCHESTRATE_SECTORS(INPUT_FILE, N_WORKERS, GPU_IDS) maps
%   worker w to gpuDevice(GPU_IDS(w)); default GPU_IDS = mod(w-1, #GPUs)+1
%   (round-robin; on a single-GPU machine all workers share device 1 --
%   correctness is unaffected, only useful for smoke tests).
%
%   Phases:
%     0  PARENT runs the driver once with precompute_only=true and a SHARED
%        precompute_dir: the per-M enumerate+collect caches are written ONCE
%        and every worker loads them in seconds (instead of re-deriving the
%        minutes-long precompute per worker). The same call returns the block
%        list (info.irrep_list/irrep_d).
%     1  The base irrep names are split cost-balanced (greedy, cost ~ d: the
%        measured per-block wall grows ~linearly in d; under spin-flip each
%        base name carries its Gamma+- pair on the same worker). Each worker
%        runs the UNMODIFIED driver on its subset via a generated input file
%        (base input + appended overrides: irrep_list / output_dir=worker_<w>
%        / shared precompute_dir).
%     2  Merge: concatenate all_E/all_w/all_M, recompute C(T)/chi(T)/Z_eff via
%        COMPUTE_OBSERVABLES_PG (identical formula to the driver), check the
%        GLOBAL sum rule (sum w == covered dimension, exact bookkeeping), and
%        save <output_dir>/<mat_name>_merged.mat with the driver's field names.
%
%   SEED NOTE: the per-block FTLM seed is 8e6 + M*1e4 + ig*100 with ig the
%   POSITION in the worker's (filtered) list -- a split run therefore draws
%   different (statistically equivalent) samples than one full run. Sum rules
%   are exact either way; converged observables agree within FTLM statistics.
%
%   See also FTLM_OBSERVABLES_PG_GPU_IH, COMPUTE_OBSERVABLES_PG.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    assert(exist(input_file, 'file') == 2, 'input file not found: %s', input_file);
    assert(n_workers >= 1 && n_workers == round(n_workers), 'n_workers must be a positive integer');
    base_txt = fileread(input_file);

    % The base output_dir (for the shared cache + the merged result): read it
    % the same way the driver does -- run the input in a scratch workspace.
    base_dir = orch_read_output_dir(input_file);
    if ~exist(base_dir, 'dir'), mkdir(base_dir); end
    pc_dir = fullfile(base_dir, 'orch_precompute');

    %% Phase 0: shared precompute (parent) + block list.
    f0 = fullfile(base_dir, 'orch_phase0_in.m');
    write_input(f0, base_txt, { ...
        'precompute_cache = true;', ...
        'precompute_only  = true;', ...
        'checkpoint       = false;', ...
        sprintf('precompute_dir = ''%s'';', fwd(pc_dir))});
    fprintf('=== orchestrate: phase 0 (shared precompute, parent process) ===\n');
    info = ftlm_observables_pg_gpu_Ih(f0);
    nb = numel(info.irrep_list);
    assert(nb >= 1, 'orchestrate: empty block list');
    if n_workers > nb
        fprintf('orchestrate: only %d blocks for %d workers -> using %d workers.\n', ...
                nb, n_workers, nb);
        n_workers = nb;
    end

    %% Cost-balanced greedy split (descending d onto the lightest worker).
    [~, order] = sort(info.irrep_d(:), 'descend');
    wcost  = zeros(n_workers, 1);
    wlists = cell(n_workers, 1);
    for w = 1:n_workers, wlists{w} = {}; end
    for k = order.'
        [~, w] = min(wcost);
        wlists{w}{end+1} = info.irrep_list{k};   %#ok<AGROW>
        wcost(w) = wcost(w) + double(info.irrep_d(k));
    end
    for w = 1:n_workers
        fprintf('  worker %d: cost %g, blocks {%s}\n', w, wcost(w), strjoin(wlists{w}, ', '));
    end

    %% Phase 1: one driver run per worker (process pool, 1 GPU each).
    try
        ngpu = gpuDeviceCount;
    catch
        ngpu = 1;
    end
    if nargin < 3 || isempty(gpu_ids)
        gpu_ids = mod((1:n_workers) - 1, max(ngpu, 1)) + 1;
    end
    assert(numel(gpu_ids) >= n_workers, 'gpu_ids must list one device per worker');

    wfiles = cell(n_workers, 1);
    wdirs  = cell(n_workers, 1);
    for w = 1:n_workers
        wdirs{w}  = fullfile(base_dir, sprintf('worker_%d', w));
        wfiles{w} = fullfile(base_dir, sprintf('orch_worker%d_in.m', w));
        names = ['{', strjoin(cellfun(@(s) ['''', s, ''''], wlists{w}, 'uni', 0), ', '), '}'];
        write_input(wfiles{w}, base_txt, { ...
            sprintf('irrep_list = %s;', names), ...
            sprintf('output_dir = ''%s'';', fwd(wdirs{w})), ...
            'precompute_cache = true;', ...
            sprintf('precompute_dir = ''%s'';', fwd(pc_dir))});
    end

    p = gcp('nocreate');
    if isempty(p) || p.NumWorkers < n_workers
        if ~isempty(p), delete(p); end
        p = parpool('Processes', n_workers);   %#ok<NASGU>
    end
    fprintf('=== orchestrate: phase 1 (%d workers) ===\n', n_workers);
    winfo = cell(n_workers, 1);
    parfor w = 1:n_workers
        gpuDevice(gpu_ids(w));                 % pin this worker's GPU
        winfo{w} = ftlm_observables_pg_gpu_Ih(wfiles{w});
    end

    %% Phase 2: merge + global sum rule + observables.
    fprintf('=== orchestrate: phase 2 (merge) ===\n');
    all_E = [];  all_w = [];  all_M = [];
    sector_M = [];  sector_G = [];  sector_dims = [];  sector_path = {};
    T_range = [];  worker_wall = zeros(n_workers, 1);
    for w = 1:n_workers
        r = load(fullfile(wdirs{w}, info.mat_name));
        all_E = [all_E; r.all_E];  all_w = [all_w; r.all_w];  all_M = [all_M; r.all_M]; %#ok<AGROW>
        sector_M = [sector_M; r.sector_M];  sector_G = [sector_G; r.sector_G];          %#ok<AGROW>
        sector_dims = [sector_dims; r.sector_dims];                                     %#ok<AGROW>
        sector_path = [sector_path; r.sector_path];                                     %#ok<AGROW>
        T_range = r.T_range;  worker_wall(w) = r.t_wall;
    end
    [C_T, chi_T, Z_eff] = compute_observables_pg(all_E, all_w, all_M, T_range);

    if info.is_full_sweep
        Z_expected = info.n_total;
        chk_label  = 'full dim';
    else
        A_M0 = round(info.N_sites * info.s_val);
        [D_chk, ~] = build_D_table(info.N_sites, round(2 * info.s_val), A_M0 + info.M_max);
        Z_expected = 0;
        for Mc = info.M_run
            Z_expected = Z_expected + (1 + (Mc > 0)) * ...
                double(D_chk(info.N_sites + 1, A_M0 + Mc + 1));
        end
        chk_label = sprintf('dim(M in {%s})', num2str(info.M_run));
    end
    sum_w  = sum(all_w);
    relerr = abs(sum_w - Z_expected) / max(Z_expected, 1);
    fprintf('Merged sum rule: sum_i w_i = %.8g, %s = %.8g, rel.err = %.2e\n', ...
            sum_w, chk_label, Z_expected, relerr);

    merged = struct('T_range', T_range, 'C_T', C_T, 'chi_T', chi_T, 'Z_eff', Z_eff, ...
        'all_E', all_E, 'all_w', all_w, 'all_M', all_M, ...
        'sector_M', sector_M, 'sector_G', sector_G, 'sector_dims', sector_dims, ...
        'sum_w', sum_w, 'Z_expected', Z_expected, 'sum_rule_relerr', relerr, ...
        'n_workers', n_workers, 'worker_wall', worker_wall, ...
        'wall_max', max(worker_wall), 'wall_sum', sum(worker_wall));
    merged.sector_path  = sector_path;
    merged.worker_lists = wlists;

    merged_path = fullfile(base_dir, strrep(info.mat_name, '.mat', '_merged.mat'));
    save(merged_path, '-struct', 'merged', '-v7.3');
    fprintf('Merged result saved to: %s (wall: max %.1f s over workers, sum %.1f s)\n', ...
            merged_path, merged.wall_max, merged.wall_sum);
end

% ----------------------------------------------------------------
function write_input(path, base_txt, overrides)
%   Input files are plain scripts -> appended assignments OVERRIDE earlier ones.
    fid = fopen(path, 'w');
    assert(fid > 0, 'orchestrate: cannot write %s', path);
    fprintf(fid, '%s\n\n%% ---- ftlm_orchestrate_sectors overrides ----\n', base_txt);
    fprintf(fid, '%s\n', overrides{:});
    fclose(fid);
end

function d = orch_read_output_dir(input_file)
%   Evaluate the input script in this function's (scratch) workspace and pick
%   up output_dir exactly like the driver does (default '.').
    run(input_file);
    if ~exist('output_dir', 'var'), output_dir = '.'; end   %#ok<NODEF>
    d = output_dir;
end

function s = fwd(p)
    s = strrep(p, '\', '/');
end
