function test_milestone_B()
%TEST_MILESTONE_B  Verify the k=0 PG SpMV and GPU kernel against the
%                  verified MATLAB sparse builder from Milestone A.
%
%   This test does NOT require the CUDA kernel to be built. It runs
%   purely in MATLAB and validates the per-thread SpMV logic (function
%   spmv_pg_matlab) against H_pg * v computed from the verified sparse
%   builder build_heisenberg_sparse_pg.
%
%   If the GPU kernel has been built (cuda_lanczos_clut_block_pg MEX is
%   on the path), the test additionally launches one block-Lanczos pass
%   in a small (M, 0) sector, recovers the Ritz values, and compares
%   them against full ED of H_pg in that sector. Otherwise that part is
%   skipped with a notice.
%
%   Run from MATLAB inside the mit_pg directory:
%       >> test_milestone_B
%
%   Passing this test implies:
%       - The CUDA SpMV (whose per-thread logic mirrors spmv_pg_matlab)
%         computes the correct symmetry-adapted matrix-vector product.
%       - The Block-Lanczos infrastructure (interleaved layout, fused
%         reductions, pointer swap) inherited from the release kernel
%         continues to work on the rep basis.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    fprintf('=== Milestone B: k=0 PG SpMV verification ===\n\n');

    cases = {
        struct('N', 4,  's_val', 0.5, 'J',  1.0)
        struct('N', 6,  's_val', 0.5, 'J',  1.0)
        struct('N', 8,  's_val', 0.5, 'J',  1.0)
        struct('N', 4,  's_val', 1.0, 'J',  1.0)
        struct('N', 10, 's_val', 0.5, 'J',  1.0)
    };

    %% Part 1: MATLAB per-thread SpMV vs. sparse H_pg
    fprintf('Part 1: per-thread SpMV (spmv_pg_matlab) vs sparse H_pg\n');
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

    %% Part 2: GPU kernel vs full ED on a small (M, 0) sector
    fprintf('Part 2: GPU Block-Lanczos vs full ED (small (M, p=0) sector)\n');
    fprintf('------------------------------------------------------------\n');
    if exist('cuda_lanczos_clut_block_pg', 'file') ~= 3
        fprintf('  cuda_lanczos_clut_block_pg MEX not found on path.\n');
        fprintf('  Run build_pg_kernels in MATLAB to enable Part 2.\n');
        fprintf('  Skipping Part 2.\n\n');
    else
        c = cases{3};   % N=8, s=1/2
        pass = run_gpu_case(c.N, c.s_val, c.J);
        overall = overall && pass;
        fprintf('  => %s\n\n', ternary(pass, 'PASS', 'FAIL'));
    end

    fprintf('=========================================================\n');
    fprintf('OVERALL Milestone B: %s\n', ternary(overall, 'PASS', 'FAIL'));
end

% ----------------------------------------------------------------
function pass = run_spmv_case(N, s_val, J)
    d_loc   = round(2*s_val + 1);
    n_total = double(d_loc)^N;
    M_max   = round(N * s_val);
    bonds   = adjacency_ring(N);
    tol     = 1e-10 * abs(J) * N;

    pass = true;

    for M = 0 : M_max
        [reps, L, dim] = enumerate_sector_with_translation(N, s_val, M, 0);
        if dim == 0, continue; end

        H_ref = full(build_heisenberg_sparse_pg(reps, L, bonds, ...
                                                 s_val, J, N, 0, n_total));
        % Random vectors: single and block.
        rng(42 + M);
        v_single = randn(dim, 1);
        V_block  = randn(dim, 4);

        % Reference
        w_ref_s = H_ref * v_single;
        W_ref_b = H_ref * V_block;

        % MATLAB per-thread SpMV
        w_thr_s = spmv_pg_matlab(reps, L, bonds, s_val, J, N, v_single);
        W_thr_b = spmv_pg_matlab(reps, L, bonds, s_val, J, N, V_block);

        err1 = max(abs(w_thr_s - w_ref_s));
        err2 = max(abs(W_thr_b(:) - W_ref_b(:)));
        st1 = ternary(err1 < tol, 'OK ', 'FAIL');
        st2 = ternary(err2 < tol, 'OK ', 'FAIL');
        fprintf('  M=%+d dim=%4d  single max|dw|=%.2e [%s]  block max|dW|=%.2e [%s]\n', ...
            M, dim, err1, st1, err2, st2);
        if err1 > tol || err2 > tol
            pass = false;
        end
    end
end

% ----------------------------------------------------------------
function pass = run_gpu_case(N, s_val, J)
%RUN_GPU_CASE  Drive the CUDA kernel through one Block-Lanczos pass on a
%  small (M = 0, p = 0) sector and check the smallest Ritz value against
%  the exact ground-state energy of H_pg in that sector.
    d_loc   = round(2*s_val + 1);
    n_total = double(d_loc)^N;
    bonds   = adjacency_ring(N);

    [reps, L, dim] = enumerate_sector_with_translation(N, s_val, 0, 0);
    fprintf('  System: N=%d, s=%g, J=%g, (M=0, p=0) sector dim=%d\n', ...
            N, s_val, J, dim);
    H_ref = full(build_heisenberg_sparse_pg(reps, L, bonds, s_val, J, N, 0, n_total));
    E_ref = sort(eig(H_ref));
    E0_ref = E_ref(1);

    % GPU device & init
    gpu_h = gpuDevice; reset(gpu_h); gpu_h = gpuDevice;
    fprintf('  GPU: %s (%.1f GB)\n', gpu_h.Name, gpu_h.TotalMemory/1e9);

    [block_base, block_mask] = build_CLT_inline(reps, n_total);
    basis_int  = int32(reps);
    orbit_int  = int32(L);
    bonds_flat = int32(reshape(bonds' - 1, [], 1));

    M_lz_run = min(80, dim);
    B_test   = min(4, dim);

    cuda_lanczos_clut_block_pg('init', ...
        gpuArray(block_base), gpuArray(block_mask), ...
        gpuArray(basis_int), gpuArray(orbit_int), ...
        bonds_flat, N, d_loc, s_val, J, dim, B_test);
    wait(gpu_h);

    rng(7);
    V0_gpu = gpuArray(single(randn(dim, B_test)));
    [AL, BE] = cuda_lanczos_clut_block_pg('block_lanczos', V0_gpu, M_lz_run);

    % Lowest Ritz value across the B chains
    E0_gpu = +inf;
    for b = 1 : B_test
        a = double(AL(:, b)); be = double(BE(:, b));
        n = numel(a);
        T = diag(a) + diag(be(1:n-1), 1) + diag(be(1:n-1), -1);
        E0_gpu = min(E0_gpu, min(eig(T)));
    end

    cuda_lanczos_clut_block_pg('cleanup');

    err = abs(E0_gpu - E0_ref);
    rel = err / max(abs(E0_ref), 1);
    fprintf('  E0 (ref FP64)  = %.10f\n', E0_ref);
    fprintf('  E0 (GPU FP32)  = %.10f\n', E0_gpu);
    fprintf('  |dE0|          = %.3e  (rel %.3e)\n', err, rel);
    % FP32 + Lanczos roundoff: ~1e-5 is plenty of margin.
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
