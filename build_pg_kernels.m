function build_pg_kernels()
%BUILD_PG_KERNELS  Compile all GPU kernels for the PG-FTLM branch.
%
%   Run from MATLAB inside the mit_pg directory:
%       >> build_pg_kernels
%
%   Currently builds:
%       cuda_lanczos_clut_block_pg.cu   (Milestone B: k=0 real FP32)
%
%   Later milestones will add the complex-k counterpart and any CR-style
%   alternative kernels. The release CPU reference kernel cpu_lanczos_omp.c
%   is not built here; see release/build_all.m for that.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    fprintf('=== Building PG-FTLM CUDA kernels ===\n\n');

    %% cuda_lanczos_clut_block_pg.cu (k=0, real FP32)
    fprintf('Building cuda_lanczos_clut_block_pg.cu (k=0 real FP32) ...\n');
    try
        mexcuda -lcublas cuda_lanczos_clut_block_pg.cu
        fprintf('  OK\n\n');
    catch ME
        fprintf(2, '  FAILED: %s\n\n', ME.message);
        rethrow(ME);
    end

    %% cuda_lanczos_clut_block_pg_cplx.cu (k!=0, complex FP32)
    fprintf('Building cuda_lanczos_clut_block_pg_cplx.cu (k!=0 complex FP32) ...\n');
    try
        mexcuda -lcublas cuda_lanczos_clut_block_pg_cplx.cu
        fprintf('  OK\n\n');
    catch ME
        fprintf(2, '  FAILED: %s\n\n', ME.message);
        rethrow(ME);
    end

    %% cuda_lanczos_clut_block_pg_Ih.cu (I_h, all 10 irreps, complex FP32)
    fprintf('Building cuda_lanczos_clut_block_pg_Ih.cu (I_h all irreps, complex FP32) ...\n');
    try
        mexcuda -lcublas cuda_lanczos_clut_block_pg_Ih.cu
        fprintf('  OK\n\n');
    catch ME
        fprintf(2, '  FAILED: %s\n\n', ME.message);
        rethrow(ME);
    end

    fprintf('Build complete.\n');
    fprintf('Files produced (Windows): cuda_lanczos_clut_block_pg.mexw64\n');
    fprintf('                          cuda_lanczos_clut_block_pg_cplx.mexw64\n');
    fprintf('                          cuda_lanczos_clut_block_pg_Ih.mexw64\n');
    fprintf('Files produced (Linux):   cuda_lanczos_clut_block_pg.mexa64\n');
    fprintf('                          cuda_lanczos_clut_block_pg_cplx.mexa64\n');
    fprintf('                          cuda_lanczos_clut_block_pg_Ih.mexa64\n');
end
