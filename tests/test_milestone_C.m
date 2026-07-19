function test_milestone_C()
%TEST_MILESTONE_C  Verify the complex-k PG SpMV and the complex GPU kernel.
%
%   Part 1 (always runs, pure MATLAB): validates spmv_pg_cplx_matlab —
%   the CPU-side reference of the per-thread CUDA logic for arbitrary p
%   — against H_pg * V with H_pg from the verified sparse builder.
%
%   Part 2 (only if cuda_lanczos_clut_block_pg_cplx is on the path):
%   launches one Block-Lanczos pass on a small complex (M, p) sector,
%   recovers the smallest Ritz value, and compares it against full ED
%   of H_pg in that sector.
%
%   Run from MATLAB inside the mit_pg directory:
%       >> test_milestone_C

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    fprintf('=== Milestone C: complex-k PG SpMV verification ===\n\n');

    cases = {
        struct('N', 4,  's_val', 0.5, 'J',  1.0)
        struct('N', 6,  's_val', 0.5, 'J',  1.0)
        struct('N', 8,  's_val', 0.5, 'J',  1.0)
        struct('N', 4,  's_val', 1.0, 'J',  1.0)
        struct('N', 10, 's_val', 0.5, 'J',  1.0)
    };

    %% Part 1: spmv_pg_cplx_matlab vs sparse H_pg, all p (incl. complex)
    fprintf('Part 1: per-thread complex SpMV vs sparse H_pg (all p)\n');
    fprintf('---------------------------------------------------------\n');
    overall = true;
    for tc = 1 : numel(cases)
        c = cases{tc};
        fprintf('Case %d: N=%d, s=%g, J=%g\n', tc, c.N, c.s_val, c.J);
        pass = run_spmv_case(c.N, c.s_val, c.J);
        overall = overall && pass;
        fprintf('  => %s\n', ternary(pass, 'PASS', 'FAIL'));
    end
    fprintf('\n');

    %% Part 2: complex GPU kernel vs full ED in a small complex (M, p) sector
    fprintf('Part 2: complex GPU Block-Lanczos vs full ED\n');
    fprintf('---------------------------------------------------------\n');
    if exist('cuda_lanczos_clut_block_pg_cplx', 'file') ~= 3
        fprintf('  cuda_lanczos_clut_block_pg_cplx MEX not found on path.\n');
        fprintf('  Run build_pg_kernels in MATLAB to enable Part 2.\n\n');
    else
        c = cases{3};  % N=8, s=1/2
        % pick a complex sector: (M = 0, p = 1)
        pass = run_gpu_cplx_case(c.N, c.s_val, c.J, 0, 1);
        overall = overall && pass;
        fprintf('  => %s\n\n', ternary(pass, 'PASS', 'FAIL'));
    end

    fprintf('=========================================================\n');
    fprintf('OVERALL Milestone C: %s\n', ternary(overall, 'PASS', 'FAIL'));
end

% ----------------------------------------------------------------
function pass = run_spmv_case(N, s_val, J)
    d_loc   = round(2*s_val + 1);
    n_total = double(d_loc)^N;
    M_max   = round(N * s_val);
    bonds   = adjacency_ring(N);
    tol     = 1e-10 * abs(J) * N;

    pass = true;
    n_fail = 0;

    for M = 0 : M_max
        for p = 0 : N - 1
            [reps, L, dim] = enumerate_sector_with_translation(N, s_val, M, p);
            if dim == 0, continue; end

            H = full(build_heisenberg_sparse_pg(reps, L, bonds, ...
                                                 s_val, J, N, p, n_total));
            % Random complex test vectors
            rng(1000*p + M + 7);
            v_re = randn(dim, 1); v_im = randn(dim, 1);
            V_re = randn(dim, 4); V_im = randn(dim, 4);
            V_complex_single = v_re + 1i*v_im;
            V_complex_block  = V_re + 1i*V_im;

            % Reference
            w_ref_s = H * V_complex_single;
            W_ref_b = H * V_complex_block;

            % MATLAB per-thread SpMV
            [w_re_s, w_im_s] = spmv_pg_cplx_matlab(reps, L, bonds, s_val, ...
                                                    J, N, p, v_re, v_im);
            [W_re_b, W_im_b] = spmv_pg_cplx_matlab(reps, L, bonds, s_val, ...
                                                    J, N, p, V_re, V_im);
            w_thr_s = w_re_s + 1i*w_im_s;
            W_thr_b = W_re_b + 1i*W_im_b;

            err1 = max(abs(w_thr_s - w_ref_s));
            err2 = max(abs(W_thr_b(:) - W_ref_b(:)));
            if err1 > tol || err2 > tol
                fprintf('  M=%+d p=%2d dim=%4d  single max|dw|=%.2e  block max|dW|=%.2e  FAIL\n', ...
                    M, p, dim, err1, err2);
                pass = false;
                n_fail = n_fail + 1;
            end
        end
    end
    if pass
        fprintf('  All (M, p) sectors PASS (machine precision)\n');
    else
        fprintf('  %d sectors FAILED\n', n_fail);
    end
end

% ----------------------------------------------------------------
function pass = run_gpu_cplx_case(N, s_val, J, M, p)
    d_loc   = round(2*s_val + 1);
    n_total = double(d_loc)^N;
    bonds   = adjacency_ring(N);

    [reps, L, dim] = enumerate_sector_with_translation(N, s_val, M, p);
    fprintf('  System: N=%d, s=%g, J=%g, (M=%d, p=%d) sector dim=%d\n', ...
            N, s_val, J, M, p, dim);
    H = full(build_heisenberg_sparse_pg(reps, L, bonds, s_val, J, N, p, n_total));
    H = 0.5 * (H + H');
    E_ref = sort(real(eig(H)));
    E0_ref = E_ref(1);

    gpu_h = gpuDevice; reset(gpu_h); gpu_h = gpuDevice;
    fprintf('  GPU: %s (%.1f GB)\n', gpu_h.Name, gpu_h.TotalMemory/1e9);

    [block_base, block_mask] = build_CLT_inline(reps, n_total);
    basis_int  = int32(reps);
    orbit_int  = int32(L);
    bonds_flat = int32(reshape(bonds' - 1, [], 1));

    M_lz_run = min(80, dim);
    B_test   = min(4, dim);

    cuda_lanczos_clut_block_pg_cplx('init', ...
        gpuArray(block_base), gpuArray(block_mask), ...
        gpuArray(basis_int), gpuArray(orbit_int), ...
        bonds_flat, N, d_loc, s_val, J, dim, B_test, p);
    wait(gpu_h);

    rng(7 + p);
    V0_re = gpuArray(single(randn(dim, B_test)));
    V0_im = gpuArray(single(randn(dim, B_test)));
    [AL, BE] = cuda_lanczos_clut_block_pg_cplx('block_lanczos', V0_re, V0_im, M_lz_run);

    E0_gpu = +inf;
    for b = 1 : B_test
        a = double(AL(:, b)); be = double(BE(:, b));
        n = numel(a);
        T = diag(a) + diag(be(1:n-1), 1) + diag(be(1:n-1), -1);
        E0_gpu = min(E0_gpu, min(eig(T)));
    end

    cuda_lanczos_clut_block_pg_cplx('cleanup');

    err = abs(E0_gpu - E0_ref);
    rel = err / max(abs(E0_ref), 1);
    fprintf('  E0 (ref FP64)  = %.10f\n', E0_ref);
    fprintf('  E0 (GPU FP32)  = %.10f\n', E0_gpu);
    fprintf('  |dE0|          = %.3e  (rel %.3e)\n', err, rel);
    pass = rel < 1e-4;
end

% ----------------------------------------------------------------
function [block_base, block_mask] = build_CLT_inline(reps, n_total)
    BLOCK_SIZE = 32;
    n_blocks   = ceil(n_total / BLOCK_SIZE);
    states     = double(reps(:));
    blks = floor(states / BLOCK_SIZE) + 1;
    bits = mod(states, BLOCK_SIZE);
    block_base      = int32(-ones(n_blocks, 1));
    [ub, fi]        = unique(blks, 'first');
    block_base(ub)  = int32(fi - 1);
    bit_vals   = pow2(bits);
    mask_sums  = accumarray(blks, bit_vals, [n_blocks, 1]);
    block_mask = uint32(mask_sums);
end

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
