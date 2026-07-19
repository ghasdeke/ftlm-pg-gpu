function ftlm_observables_pg(input_file)
%FTLM_OBSERVABLES_PG  Sector-FTLM with C_N translation symmetry for spin rings.
%
%   FTLM_OBSERVABLES_PG(INPUT_FILE) computes the specific heat C(T), the
%   magnetic susceptibility chi(T), and the effective partition function
%   Z_eff(T) of an N-site Heisenberg ring with local spin S_VAL, exploiting
%   the joint conservation of total S^z and lattice translation T (cyclic
%   group C_N). The Hilbert space is decomposed into simultaneous
%   eigenspaces of S^z (label M) and the C_N momentum (label k = 2*pi*p/N
%   with p = 0..N-1), and FTLM is run independently in each (M, p) block.
%
%   This is the pure-MATLAB CPU reference path for Milestones B/C of the
%   PG-extension branch (mit_pg). Sectors with real H (p = 0 or 2p = N)
%   use real arithmetic; all other sectors use complex Hermitian
%   arithmetic. The corresponding GPU FP32 kernel will replace
%   run_ftlm_pg_sector in a later milestone.
%
%   Aggregation rule:
%       Z(beta)  = sum_{M, p, r, k} mult_M * w_k * exp(-beta * E_k)
%       <O>(beta)= sum_{M, p, r, k} mult_M * w_k * O_k * exp(-beta * E_k) / Z
%   where (E_k, w_k) come from per-(M, p, r) Lanczos diagonalization and
%   mult_M = 1 + (M > 0) absorbs the M <-> -M spin-inversion multiplicity.
%
%   INPUT_FILE is a plain MATLAB script. Required variables:
%       N_ring     ring length (integer >= 3)
%       s_val      local spin (0.5, 1, 1.5, ...)
%       J          uniform nearest-neighbor exchange coupling
%       R          number of FTLM random vectors per (M, p) sector
%       M_lz       Lanczos iterations per random vector
%       T_range    temperature grid (positive, units of J/k_B)
%
%   Optional inputs (defaults in parentheses):
%       only_M0          (false)  restrict to M = 0
%       only_p0          (false)  restrict to p = 0
%       merge_kbar       (true)   skip the complex-conjugate partner sector
%                                 (k, N - k) and instead double-weight the
%                                 computed one. Exact for real H because
%                                 spec(H_pg^p) = spec(H_pg^{N-p}); halves
%                                 the number of complex-k sectors that
%                                 need to run.
%       use_spin_parity  (true)   In M = 0 sectors, exploit the spin-
%                                 inversion Z_2 symmetry (P: m_i -> -m_i)
%                                 by projecting random FTLM starting
%                                 vectors onto the +/- parity subspaces.
%                                 R is then SPLIT between the two
%                                 sub-sectors (R_+ = ceil(R/2),
%                                 R_- = R - R_+), so the total Lanczos
%                                 work matches the no-parity case
%                                 instead of doubling. Yields per-block
%                                 spectra labeled by parity but does
%                                 not change SpMV cost; an actual
%                                 ~2x speedup would require restricting
%                                 the basis to one sub-sector (future
%                                 work). Only acts on M = 0 sectors;
%                                 set false for explicit verification.
%       ed_thresh        (0)      dim_(M,p) <= ed_thresh -> dense ED
%       output_dir       ('.')    directory for the output .mat file
%
%   Output: a single .mat file named ftlm_pg_ring_<N>_s<s_str>.mat in
%   OUTPUT_DIR, containing T_range, C_T, chi_T, Z_eff, configuration
%   and per-sector diagnostics.
%
%   See also ENUMERATE_SECTOR_WITH_TRANSLATION,
%            BUILD_HEISENBERG_SPARSE_PG,
%            RUN_FTLM_PG_SECTOR,
%            COMPUTE_OBSERVABLES_PG.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    if nargin < 1 || isempty(input_file)
        error('ftlm_observables_pg:NoInput', ...
              'Usage: ftlm_observables_pg(''input.m'')');
    end
    if exist(input_file, 'file') ~= 2
        error('ftlm_observables_pg:InputNotFound', ...
              'Input file not found: %s', input_file);
    end

    fprintf('=== ftlm_observables_pg: PG-FTLM (CPU FP64) for spin rings ===\n');
    fprintf('Input file: %s\n\n', input_file);

    run(input_file);   %#ok<*NODEF>

    % --- Required inputs ---
    required = {'N_ring', 's_val', 'J', 'R', 'M_lz', 'T_range'};
    for k = 1 : numel(required)
        if ~exist(required{k}, 'var')
            error('ftlm_observables_pg:Missing', ...
                  'Required input variable missing in %s: %s', input_file, required{k});
        end
    end

    % --- Optional inputs ---
    if ~exist('only_M0',         'var'), only_M0         = false; end
    if ~exist('only_p0',         'var'), only_p0         = false; end
    if ~exist('merge_kbar',      'var'), merge_kbar      = true;  end
    if ~exist('use_spin_parity', 'var'), use_spin_parity = true;  end
    if ~exist('ed_thresh',       'var'), ed_thresh       = 0;     end
    if ~exist('output_dir',      'var'), output_dir      = '.';   end

    % --- Validation ---
    assert(N_ring >= 3 && N_ring == round(N_ring), 'N_ring must be integer >= 3.');
    assert(s_val > 0 && abs(2*s_val - round(2*s_val)) < 1e-12, ...
        's_val must be a positive half-integer.');
    assert(isfinite(J) && isnumeric(J) && isscalar(J), 'J must be a finite scalar.');
    assert(R >= 1 && R == round(R), 'R must be a positive integer.');
    assert(M_lz >= 1 && M_lz == round(M_lz), 'M_lz must be a positive integer.');
    assert(all(T_range > 0) && all(isfinite(T_range)), ...
        'T_range must contain positive finite values.');
    T_range = T_range(:).';

    %% System setup
    N       = N_ring;
    d_loc   = round(2*s_val + 1);
    n_total = double(d_loc)^N;
    M_max   = round(N * s_val);
    bonds   = adjacency_ring(N);

    two_s = round(2 * s_val);
    if mod(two_s, 2) == 0
        s_str = sprintf('%d', two_s/2);
    else
        s_str = sprintf('%do2', two_s);
    end

    fprintf('System:    N=%d ring, s=%s (d_loc=%d, n_total=%g, M_max=%d)\n', ...
        N, s_str, d_loc, n_total, M_max);
    fprintf('FTLM:      R=%d, M_lz=%d, T-grid: %d points in [%.3g, %.3g]\n', ...
        R, M_lz, numel(T_range), min(T_range), max(T_range));
    if only_M0, fprintf('Restricted to M=0 sectors\n'); end
    if only_p0, fprintf('Restricted to p=0 (trivial irrep)\n'); end
    if merge_kbar
        p_max_loop = floor(N/2);
        fprintf('Using k <-> N-k pair symmetry: computing p in 0..%d, ', p_max_loop);
        fprintf('with mult_p = 2 for paired sectors.\n');
    else
        p_max_loop = N - 1;
        fprintf('Computing all %d momentum sectors explicitly (merge_kbar=false).\n', N);
    end
    if use_spin_parity
        fprintf('Spin-parity reduction ON: (M=0, p) sectors split into +/- subspaces.\n');
    else
        fprintf('Spin-parity reduction OFF.\n');
    end
    fprintf('\n');

    %% Main loop over (M, p) sectors
    all_E       = zeros(0, 1);
    all_w       = zeros(0, 1);
    all_M       = zeros(0, 1);
    sector_M    = zeros(0, 1);
    sector_p    = zeros(0, 1);
    sector_dims = zeros(0, 1);

    t_start = tic;

    for M = 0 : M_max
        if only_M0 && M ~= 0, continue; end
        mult_M = 1 + (M > 0);

        for p = 0 : p_max_loop
            if only_p0 && p ~= 0, continue; end

            % k <-> N-k pair multiplicity. Self-conjugate irreps (p = 0, and
            % p = N/2 for even N) have mult_p = 1; all complex-conjugate
            % partners outside the canonical range contribute via mult_p = 2.
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
            t_sec = tic;

            % Decide whether to split into parity sub-sectors for FTLM.
            % Parity reduction only applies to M = 0 (P maps M to -M).
            do_par_FTLM = use_spin_parity && (M == 0) && (dim_sec > ed_thresh);

            E_sec = [];
            w_sec = [];
            par_str = '';

            if dim_sec <= ed_thresh
                % --- Dense ED branch (unaffected by parity setting) ---
                H = build_heisenberg_sparse_pg(reps, orbit_lens, bonds, ...
                                               s_val, J, N, p, n_total);
                Hd = full(H);
                Hd = 0.5 * (Hd + Hd');
                E_sec = sort(real(eig(Hd)));
                w_sec = ones(numel(E_sec), 1);
                method_str = 'ED';
            elseif do_par_FTLM
                % --- FTLM with parity reduction ---
                % R is split between the +/- parity sub-sectors so that
                % the total Lanczos work is comparable to the no-parity
                % case (instead of doubling). The split is balanced;
                % asymmetric (D_plus, D_minus) pairs still share R = R_+
                % + R_- evenly.
                [P_idx, P_phase, D_plus, D_minus] = parity_action_pg( ...
                    reps, orbit_lens, N, s_val, p);
                H = build_heisenberg_sparse_pg(reps, orbit_lens, bonds, ...
                                               s_val, J, N, p, n_total);
                sigma_list    = [+1, -1];
                D_sigma_list  = [D_plus, D_minus];
                R_plus  = ceil(R/2);
                R_minus = R - R_plus;
                R_sigma_list  = [R_plus, R_minus];
                for s_idx = 1:2
                    sigma_val = sigma_list(s_idx);
                    D_sigma   = D_sigma_list(s_idx);
                    R_sigma   = R_sigma_list(s_idx);
                    if D_sigma <= 0 || R_sigma == 0, continue; end
                    parity_struct = struct('sigma',   sigma_val, ...
                                            'D_sigma', D_sigma, ...
                                            'P_idx',   P_idx, ...
                                            'P_phase', P_phase);
                    seed = 1e6 + M * 1e4 + p * 100 + (sigma_val > 0);
                    [E_p, w_p] = run_ftlm_pg_sector(H, dim_sec, R_sigma, M_lz, ...
                                                    is_real, seed, parity_struct);
                    E_sec = [E_sec; E_p];   %#ok<AGROW>
                    w_sec = [w_sec; w_p];   %#ok<AGROW>
                end
                method_str = sprintf('FTLM R=%d+%d, M_lz=%d', ...
                                     R_plus, R_minus, ...
                                     min(M_lz, max(D_plus, D_minus)));
                par_str = sprintf(' [P+%d/-%d]', D_plus, D_minus);
            else
                % --- FTLM without parity ---
                H = build_heisenberg_sparse_pg(reps, orbit_lens, bonds, ...
                                               s_val, J, N, p, n_total);
                seed = 1e6 + M * 1e4 + p * 100;
                [E_sec, w_sec] = run_ftlm_pg_sector(H, dim_sec, R, M_lz, ...
                                                    is_real, seed);
                method_str = sprintf('FTLM R=%d, M_lz=%d', ...
                                     min(R, dim_sec), min(M_lz, dim_sec));
            end

            % Apply both multiplicities to weights:
            %   mult_M : M <-> -M spin-inversion (always 1 for M = 0)
            %   mult_p : k <-> N-k complex-conjugate sector pair (merge_kbar)
            w_sec = w_sec * (mult_M * mult_p);

            all_E = [all_E; E_sec];                       %#ok<AGROW>
            all_w = [all_w; w_sec];                       %#ok<AGROW>
            all_M = [all_M; M * ones(numel(E_sec), 1)];    %#ok<AGROW>

            arith_str = ternary(is_real, 'real', 'cplx');
            fprintf('  M=%2d p=%2d (%s, x%d) dim=%6d%s %-22s t=%.2fs\n', ...
                M, p, arith_str, mult_p, dim_sec, par_str, method_str, toc(t_sec));
        end
    end
    t_wall = toc(t_start);
    fprintf('\nTotal wall time: %.2f s\n', t_wall);

    %% Observables
    [C_T, chi_T, Z_eff] = compute_observables_pg(all_E, all_w, all_M, T_range);

    %% Save results
    mat_name = sprintf('ftlm_pg_ring_%d_s%s.mat', N, s_str);
    mat_path = fullfile(output_dir, mat_name);
    n_total_save = n_total;     %#ok<NASGU>
    save(mat_path, ...
        'T_range', 'C_T', 'chi_T', 'Z_eff', ...
        'N', 's_val', 'J', 'R', 'M_lz', 'M_max', 'n_total_save', ...
        'only_M0', 'only_p0', 'merge_kbar', 'use_spin_parity', 'ed_thresh', ...
        'sector_M', 'sector_p', 'sector_dims', ...
        't_wall', '-v7.3');
    fprintf('Results saved to: %s\n', mat_path);

    %% Brief partition-function sanity print
    Z_inf = sum(all_w);   % should equal full Hilbert dimension n_total
    fprintf('Sum-rule check: sum_i w_i = %.6g, full dim = %.6g, rel.err = %.2e\n', ...
        Z_inf, n_total, abs(Z_inf - n_total) / max(n_total, 1));
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
