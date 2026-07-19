function test_milestone_G_GPU()
%TEST_MILESTONE_G_GPU  GPU I_h kernel smoke test against the CPU CLT SpMV.
%
%   Two-stage check:
%
%   (1) SpMV-only path. For each (M, Gamma) block on the s=1/2
%       icosahedron, builds the CLT and a random complex block X,
%       applies H_block via CUDA_LANCZOS_CLUT_BLOCK_PG_IH('spmv')
%       on the GPU (FP32), and compares the result against the
%       CPU FP64 reference SPMV_PG_IH_CLT_MATLAB(clt, X). The
%       relative error must lie at the FP32 round-off level
%       (~ 1e-5 .. 1e-4 for these block dimensions).
%
%   (2) Block-Lanczos path. For one representative larger block
%       (M=0, H_g) runs M_lz steps of block-Lanczos on the GPU and
%       returns the tridiagonal coefficients. The smallest Ritz value
%       must match the smallest eigenvalue of the dense H_block
%       (computed via build_heisenberg_sparse_Ih_gamma2 + eig) to a
%       few FP32 ULP, confirming the orthogonalisation loop is wired
%       up correctly end-to-end.
%
%   Requires a CUDA-capable GPU and the MEX file produced by
%   BUILD_PG_KERNELS. Run from MATLAB inside mit_pg/:
%       >> build_pg_kernels        % once, to compile the CU files
%       >> test_milestone_G_GPU

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    fprintf('=== Milestone G GPU: I_h kernel smoke test ===\n\n');

    if exist('cuda_lanczos_clut_block_pg_Ih', 'file') ~= 3
        fprintf(2, 'MEX file cuda_lanczos_clut_block_pg_Ih not found.\n');
        fprintf(2, 'Run build_pg_kernels first.\n');
        return;
    end

    try
        gpu_h = gpuDevice();
        fprintf('GPU: %s, compute capability %d.%d, %.1f GB free\n\n', ...
            gpu_h.Name, gpu_h.ComputeCapability(1), gpu_h.ComputeCapability(end), ...
            gpu_h.AvailableMemory / 1e9);
    catch
        fprintf(2, 'No CUDA-capable GPU detected.\n');
        return;
    end

    group = icosahedron_Ih_full();
    bonds = adjacency_icosahedron_Ih();
    s_val = 0.5; J = 1.0;
    B_test = 4;
    tol_spmv = 5e-4;     % FP32 cumulative roundoff on these dims

    irreps = build_irreps_table(group);

    %% Stage 1: SpMV-only path
    fprintf('--- Stage 1: SpMV vs CPU CLT reference ---\n');
    rng(99);
    overall_spmv = true;
    for M = 0 : 3
        fprintf('  M = %d\n', M);
        for k = 1 : numel(irreps)
            ir = irreps{k};
            [super_reps, V_per_rep, eig_per_rep, n_per_rep, ~] = ...
                enumerate_sector_with_Ih_gamma2(s_val, M, ir.data, ir.d, group);
            n_basis = double(sum(n_per_rep));
            if n_basis == 0
                continue;
            end

            clt = build_clt_pg_Ih(super_reps, V_per_rep, eig_per_rep, ...
                n_per_rep, bonds, s_val, J, ir.data, ir.d, group);

            X = randn(n_basis, B_test) + 1i * randn(n_basis, B_test);
            X_re = single(real(X));
            X_im = single(imag(X));

            Y_cpu = spmv_pg_Ih_clt_matlab(clt, X);

            Y_gpu = gpu_spmv(clt, X_re, X_im, gpu_h);

            err_abs = norm(Y_cpu - Y_gpu, 'fro');
            nrm     = norm(Y_cpu, 'fro');
            err_rel = err_abs / max(nrm, 1e-20);
            ok = err_rel < tol_spmv;
            overall_spmv = overall_spmv && ok;
            fprintf('    %-4s (d=%d): n_basis = %3d, ||Y_cpu - Y_gpu||_rel = %.2e  [%s]\n', ...
                ir.name, ir.d, n_basis, err_rel, ternary(ok, 'OK', 'FAIL'));
        end
    end

    %% Stage 2: Block-Lanczos lowest Ritz vs dense eig
    fprintf('\n--- Stage 2: Block-Lanczos lowest Ritz vs dense ED ---\n');
    overall_lz = true;
    M_test = 0;
    target_irreps = {'A_g', 'H_g'};
    R_test  = 4;
    Mlz_lz  = 80;
    tol_lz  = 5e-4;
    for k = 1 : numel(target_irreps)
        name = target_irreps{k};
        ir   = lookup_irrep(irreps, name);

        [super_reps, V_per_rep, eig_per_rep, n_per_rep, ~] = ...
            enumerate_sector_with_Ih_gamma2(s_val, M_test, ir.data, ir.d, group);
        n_basis = double(sum(n_per_rep));
        if n_basis == 0
            fprintf('    %-4s: empty, skip\n', ir.name);
            continue;
        end

        clt = build_clt_pg_Ih(super_reps, V_per_rep, eig_per_rep, ...
            n_per_rep, bonds, s_val, J, ir.data, ir.d, group);

        H = build_heisenberg_sparse_Ih_gamma2(super_reps, V_per_rep, ...
            eig_per_rep, n_per_rep, bonds, s_val, J, ir.data, ir.d, group);
        H = full(H);
        H = 0.5 * (H + H');
        E_dense = sort(real(eig(H)));
        E0_ref  = E_dense(1);

        [E_sec, ~, ~] = run_ftlm_pg_sector_gpu_Ih(clt, R_test, Mlz_lz, 0, ...
                                                   12345 + k, gpu_h);
        E0_gpu = min(E_sec);

        err_E0 = abs(E0_gpu - E0_ref) / max(abs(E0_ref), 1e-12);
        ok = err_E0 < tol_lz;
        overall_lz = overall_lz && ok;
        fprintf('    M=%d %-4s: n_basis = %3d, E0_dense = %.6f, E0_GPU = %.6f, rel.err = %.2e  [%s]\n', ...
            M_test, ir.name, n_basis, E0_ref, E0_gpu, err_E0, ternary(ok, 'OK', 'FAIL'));
    end

    fprintf('\n=========================================================\n');
    overall = overall_spmv && overall_lz;
    fprintf('OVERALL Milestone G GPU: %s\n', ternary(overall, 'PASS', 'FAIL'));
end


% ----------------------------------------------------------------
function Y = gpu_spmv(clt, X_re, X_im, gpu_h)
% Apply a single FP32 GPU SpMV using the 'spmv' mode of the kernel.
    n_basis   = clt.n_basis;
    d_irrep   = clt.d_irrep;
    n_reps    = clt.n_reps;
    n_entries = double(sum(clt.entries_per_rep));
    [~, B] = size(X_re);

    % clt.M is now a [d x d x n_entries] padded complex double tensor
    % (build_clt_pg_Ih, Plan A). The previous per-entry cell unpack
    % collapses into two single-precision element-wise conversions.
    M_re = single(real(clt.M));
    M_im = single(imag(clt.M));

    assert_kernel_abi();
    diag_gpu       = gpuArray(single(clt.diag_vals));
    % int64 offsets (64-bit basis-offset ABI v2; the kernel validates class).
    rep_offs_gpu   = gpuArray(int64(clt.rep_offsets));
    n_per_rep_gpu  = gpuArray(int32(clt.n_per_rep));
    entries_n_gpu  = gpuArray(int32(clt.entries_per_rep));
    entry_offs_gpu = gpuArray(int64(clt.entry_offsets));
    src_idx_gpu    = gpuArray(int32(clt.src_idx - 1));
    M_re_gpu       = gpuArray(M_re(:));
    M_im_gpu       = gpuArray(M_im(:));

    cuda_lanczos_clut_block_pg_Ih('init', ...
        diag_gpu, rep_offs_gpu, n_per_rep_gpu, ...
        entries_n_gpu, entry_offs_gpu, src_idx_gpu, ...
        M_re_gpu, M_im_gpu, ...
        n_basis, n_reps, n_entries, d_irrep, B);
    wait(gpu_h);

    [Y_re, Y_im] = cuda_lanczos_clut_block_pg_Ih('spmv', ...
                       gpuArray(X_re), gpuArray(X_im));

    cuda_lanczos_clut_block_pg_Ih('cleanup');

    Y = double(Y_re) + 1i * double(Y_im);
end


% ----------------------------------------------------------------
function irreps = build_irreps_table(group)
    irreps = {};
    irreps{end+1} = struct('name', 'A_g', 'd', 1, 'data', group.Ag);
    irreps{end+1} = struct('name', 'A_u', 'd', 1, 'data', group.Au);
    irreps{end+1} = struct('name', 'T1g', 'd', 3, 'data', group.T1g);
    irreps{end+1} = struct('name', 'T1u', 'd', 3, 'data', group.T1u);
    irreps{end+1} = struct('name', 'T2g', 'd', 3, 'data', group.T2g);
    irreps{end+1} = struct('name', 'T2u', 'd', 3, 'data', group.T2u);
    irreps{end+1} = struct('name', 'F_g', 'd', 4, 'data', group.Fg);
    irreps{end+1} = struct('name', 'F_u', 'd', 4, 'data', group.Fu);
    irreps{end+1} = struct('name', 'H_g', 'd', 5, 'data', group.Hg);
    irreps{end+1} = struct('name', 'H_u', 'd', 5, 'data', group.Hu);
end

function ir = lookup_irrep(irreps, name)
    ir = [];
    for k = 1 : numel(irreps)
        if strcmp(irreps{k}.name, name)
            ir = irreps{k};
            return;
        end
    end
    error('test_milestone_G_GPU:irrep', 'unknown irrep %s', name);
end

function s = ternary(cond, a, b)
    if cond, s = a; else, s = b; end
end
