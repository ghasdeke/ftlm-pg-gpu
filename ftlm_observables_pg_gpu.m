function ftlm_observables_pg_gpu(input_file)
%FTLM_OBSERVABLES_PG_GPU  PG-FTLM with mixed GPU/CPU execution per irrep.
%
%   FTLM_OBSERVABLES_PG_GPU(INPUT_FILE) is the mixed-precision sister of
%   FTLM_OBSERVABLES_PG. It dispatches each (M, p) sector to the GPU
%   FP32 path (cuda_lanczos_clut_block_pg) when p is the trivial irrep
%   (p = 0), and falls back to the CPU FP64 path (run_ftlm_pg_sector +
%   lanczos_recursion_pg) for all other irreps. Aggregation across
%   sectors and computation of C(T), chi(T), Z_eff(T) is unchanged.
%
%   The current state of the GPU pipeline (Milestone B) only handles
%   real arithmetic (p = 0). The complex-k GPU kernel is the subject of
%   Milestone C. Until then, this driver yields full physical thermal
%   observables but uses GPU acceleration only for the totally symmetric
%   sector.
%
%   Input file syntax matches FTLM_OBSERVABLES_PG plus the additional
%   GPU-specific knobs:
%       B_gpu          (0)         GPU block-Lanczos block size (0 = adaptive)
%       L2_cache_bytes (48e6)      L2 size used by the adaptive heuristic
%
%   See also FTLM_OBSERVABLES_PG, RUN_FTLM_PG_SECTOR_GPU,
%            RUN_FTLM_PG_SECTOR.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    if nargin < 1 || isempty(input_file)
        error('ftlm_observables_pg_gpu:NoInput', ...
              'Usage: ftlm_observables_pg_gpu(''input.m'')');
    end

    fprintf('=== ftlm_observables_pg_gpu: PG-FTLM (mixed GPU FP32 / CPU FP64) ===\n');
    fprintf('Input file: %s\n\n', input_file);

    run(input_file);   %#ok<*NODEF>

    required = {'N_ring', 's_val', 'J', 'R', 'M_lz', 'T_range'};
    for k = 1 : numel(required)
        if ~exist(required{k}, 'var')
            error('ftlm_observables_pg_gpu:Missing', ...
                  'Required input missing: %s', required{k});
        end
    end

    if ~exist('only_M0',         'var'), only_M0         = false; end
    if ~exist('only_p0',         'var'), only_p0         = false; end
    if ~exist('merge_kbar',      'var'), merge_kbar      = true;  end
    if ~exist('use_spin_parity', 'var'), use_spin_parity = true;  end
    if ~exist('ed_thresh',       'var'), ed_thresh       = 0;     end
    if ~exist('output_dir',      'var'), output_dir      = '.';   end
    if ~exist('B_gpu',           'var'), B_gpu           = 0;     end
    if ~exist('L2_cache_bytes',  'var'), L2_cache_bytes  = 48e6;  end

    %% GPU init
    assert(gpuDeviceCount > 0, 'No CUDA-capable GPU found.');
    gpu_h = gpuDevice;
    reset(gpu_h);
    gpu_h = gpuDevice;
    fprintf('GPU: %s (%.1f GB VRAM)\n', gpu_h.Name, gpu_h.TotalMemory/1e9);
    assert(exist('cuda_lanczos_clut_block_pg', 'file') == 3, ...
        'cuda_lanczos_clut_block_pg MEX not found. Run build_pg_kernels first.');
    have_cplx_kernel = (exist('cuda_lanczos_clut_block_pg_cplx', 'file') == 3);
    if ~have_cplx_kernel
        fprintf(['Note: cuda_lanczos_clut_block_pg_cplx MEX not found.\n', ...
                 '      Complex-p sectors will fall back to the CPU FP64 path.\n', ...
                 '      Run build_pg_kernels to enable the complex GPU kernel.\n\n']);
    end

    %% Setup
    N       = N_ring;
    d_loc   = round(2*s_val + 1);
    n_total = double(d_loc)^N;
    M_max   = round(N * s_val);
    bonds   = adjacency_ring(N);

    two_s = round(2 * s_val);
    if mod(two_s, 2) == 0, s_str = sprintf('%d', two_s/2);
    else,                  s_str = sprintf('%do2', two_s); end

    fprintf('System:    N=%d ring, s=%s (d_loc=%d, n_total=%g, M_max=%d)\n', ...
        N, s_str, d_loc, n_total, M_max);
    fprintf('FTLM:      R=%d, M_lz=%d, T-grid: %d points in [%.3g, %.3g]\n', ...
        R, M_lz, numel(T_range), min(T_range), max(T_range));
    if merge_kbar
        p_max_loop = floor(N/2);
        fprintf('PG:        k <-> N-k pairing ON; p in 0..%d, mult_p=2 for paired sectors\n', p_max_loop);
    else
        p_max_loop = N - 1;
        fprintf('PG:        k <-> N-k pairing OFF; computing all %d momentum sectors\n', N);
    end
    if use_spin_parity
        fprintf('Parity:    (M=0, p) split into +/- subspaces (use_spin_parity=true)\n');
    else
        fprintf('Parity:    OFF (use_spin_parity=false)\n');
    end
    fprintf('\n');

    %% Main loop
    all_E = []; all_w = []; all_M = [];
    sector_M = []; sector_p = []; sector_dims = []; sector_path = {};
    t_start = tic;

    for M = 0 : M_max
        if only_M0 && M ~= 0, continue; end
        mult_M = 1 + (M > 0);

        for p = 0 : p_max_loop
            if only_p0 && p ~= 0, continue; end

            % k <-> N-k pair multiplicity: mult_p = 2 for p whose partner
            % N-p lies outside the canonical range we loop over (i.e., not
            % p = 0 and not p = N/2 for even N).
            if merge_kbar
                mult_p = 1 + (p > 0 && 2*p < N);
            else
                mult_p = 1;
            end

            [reps, orbit_lens, dim_sec] = enumerate_sector_with_translation( ...
                                              N, s_val, M, p);
            if dim_sec == 0, continue; end

            sector_M(end+1, 1)    = M;       %#ok<AGROW>
            sector_p(end+1, 1)    = p;       %#ok<AGROW>
            sector_dims(end+1, 1) = dim_sec; %#ok<AGROW>

            is_real = (p == 0) || (2*p == N);
            use_gpu_real = (p == 0);
            use_gpu_cplx = (p ~= 0) && have_cplx_kernel;

            % Parity reduction only acts on M = 0 sectors and only on the
            % FTLM branch (the ED branch is exact regardless).
            do_par_FTLM = use_spin_parity && (M == 0) && (dim_sec > ed_thresh);

            t_sec  = tic;
            E_sec  = [];
            w_sec  = [];
            par_str = '';
            path   = '';

            if dim_sec <= ed_thresh
                H = build_heisenberg_sparse_pg(reps, orbit_lens, bonds, ...
                                               s_val, J, N, p, n_total);
                Hd = full(H); Hd = 0.5 * (Hd + Hd');
                E_sec = sort(real(eig(Hd)));
                w_sec = ones(numel(E_sec), 1);
                path = 'ED';
            else
                % Decide on sigma iteration.
                % With parity on, R is split between the +/- sub-sectors
                % so total Lanczos work remains comparable to the
                % no-parity case (instead of doubling).
                if do_par_FTLM
                    [P_idx, P_phase, D_plus, D_minus] = parity_action_pg( ...
                        reps, orbit_lens, N, s_val, p);
                    sigma_list   = [+1, -1];
                    D_sigma_list = [D_plus, D_minus];
                    R_plus  = ceil(R/2);
                    R_minus = R - R_plus;
                    R_sigma_list = [R_plus, R_minus];
                    par_str = sprintf(' [P+%d/-%d]', D_plus, D_minus);
                else
                    sigma_list   = 0;
                    D_sigma_list = dim_sec;
                    R_sigma_list = R;
                end

                H_sparse = [];
                for s_idx = 1 : numel(sigma_list)
                    R_eff_user = R_sigma_list(s_idx);
                    if R_eff_user == 0, continue; end
                    if sigma_list(s_idx) == 0
                        parity_struct = [];
                    else
                        D_s = D_sigma_list(s_idx);
                        if D_s <= 0, continue; end
                        parity_struct = struct('sigma',   sigma_list(s_idx), ...
                                                'D_sigma', D_s, ...
                                                'P_idx',   P_idx, ...
                                                'P_phase', P_phase);
                    end

                    if use_gpu_real
                        [E_s, w_s, ~] = run_ftlm_pg_sector_gpu( ...
                            reps, orbit_lens, bonds, s_val, J, N, ...
                            dim_sec, R_eff_user, M_lz, B_gpu, L2_cache_bytes, ...
                            gpu_h, parity_struct);
                        path = 'GPU-r(FP32)';
                    elseif use_gpu_cplx
                        [E_s, w_s, ~] = run_ftlm_pg_sector_gpu_cplx( ...
                            reps, orbit_lens, bonds, s_val, J, N, p, ...
                            dim_sec, R_eff_user, M_lz, B_gpu, L2_cache_bytes, ...
                            gpu_h, parity_struct);
                        path = 'GPU-c(FP32)';
                    else
                        if isempty(H_sparse)
                            H_sparse = build_heisenberg_sparse_pg( ...
                                reps, orbit_lens, bonds, s_val, J, N, p, n_total);
                        end
                        seed = 1e6 + M * 1e4 + p * 100 + ...
                               (sigma_list(s_idx) > 0);
                        [E_s, w_s] = run_ftlm_pg_sector(H_sparse, dim_sec, ...
                            R_eff_user, M_lz, is_real, seed, parity_struct);
                        path = 'CPU(FP64)';
                    end
                    E_sec = [E_sec; E_s];  %#ok<AGROW>
                    w_sec = [w_sec; w_s];  %#ok<AGROW>
                end
            end

            % Apply both multiplicities: mult_M (M <-> -M) and mult_p (k <-> N-k).
            % Parity sub-sector weights are already scaled by D_sigma/R_eff
            % inside the sector worker; here we only multiply by mult_M*mult_p.
            w_sec = w_sec * (mult_M * mult_p);
            all_E = [all_E; E_sec];                          %#ok<AGROW>
            all_w = [all_w; w_sec];                          %#ok<AGROW>
            all_M = [all_M; M * ones(numel(E_sec), 1)];       %#ok<AGROW>
            sector_path{end+1, 1} = path;                     %#ok<AGROW>

            arith = ternary(is_real, 'real', 'cplx');
            fprintf('  M=%2d p=%2d (%s, x%d) dim=%6d%s %-12s t=%.2fs\n', ...
                M, p, arith, mult_p, dim_sec, par_str, path, toc(t_sec));
        end
    end
    t_wall = toc(t_start);
    fprintf('\nTotal wall time: %.2f s\n', t_wall);

    %% Observables
    [C_T, chi_T, Z_eff] = compute_observables_pg(all_E, all_w, all_M, T_range);

    %% Sum-rule sanity print
    Z_inf = sum(all_w);
    fprintf('Sum-rule check: sum_i w_i = %.6g, full dim = %.6g, rel.err = %.2e\n', ...
        Z_inf, n_total, abs(Z_inf - n_total) / max(n_total, 1));

    %% Save
    mat_name = sprintf('ftlm_pg_gpu_ring_%d_s%s.mat', N, s_str);
    mat_path = fullfile(output_dir, mat_name);
    n_total_save = n_total;     %#ok<NASGU>
    save(mat_path, ...
        'T_range', 'C_T', 'chi_T', 'Z_eff', ...
        'N', 's_val', 'J', 'R', 'M_lz', 'M_max', 'n_total_save', ...
        'only_M0', 'only_p0', 'merge_kbar', 'use_spin_parity', 'ed_thresh', ...
        'B_gpu', 'L2_cache_bytes', ...
        'sector_M', 'sector_p', 'sector_dims', 'sector_path', ...
        't_wall', '-v7.3');
    fprintf('Results saved to: %s\n', mat_path);
end

% ----------------------------------------------------------------
function bonds = adjacency_ring(N)
    bonds = zeros(N, 2);
    for i = 1 : N - 1
        bonds(i, :) = [i, i+1];
    end
    bonds(N, :) = [N, 1];
end

function s = ternary(cond, a, b)
    if cond, s = a; else, s = b; end
end
