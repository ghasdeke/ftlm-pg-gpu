function build_all(build_legacy, verbose)
%BUILD_ALL  One-command build of the symmetry-adapted GPU-FTLM native code.
%
%   BUILD_ALL              builds the production kernels.
%   BUILD_ALL(true)        also builds the legacy ring (k=0 / complex-k) kernels.
%   BUILD_ALL(false, true) verbose: print ptxas register/spill statistics for
%                          the production kernel (-Xptxas -v; identical binary).
%
%   Production native code (required for the GPU pipeline):
%     1. cuda_lanczos_clut_block_pg_Ih.cu  -- the block-Lanczos SpMV kernel
%        (one CUDA file, ~6 device kernels) used by ftlm_observables_pg_gpu_Ih
%        for ALL geometries / spins / irreps / lookup methods.  (mexcuda)
%     2. schnack_query_mex.cpp             -- the combinatorial-ranking lookup
%        used when n_total > 2^32 (e.g. N=36).  CPU C++ MEX, std::thread.  (mex)
%
%   Requirements: MATLAB + Parallel Computing Toolbox, a CUDA-capable GPU and a
%   matching CUDA toolkit (mexcuda), and a C++ compiler (mex -setup C++). The
%   schnack MEX uses std::thread (NOT OpenMP) to avoid the MSVC vcomp / MATLAB
%   libiomp5 teardown clash.
%
%   See also BUILD_PG_KERNELS, BUILD_SCHNACK_QUERY_MEX, RUN_ALL_TESTS.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    if nargin < 1, build_legacy = false; end
    if nargin < 2, verbose = false; end
    here = fileparts(mfilename('fullpath'));
    old  = cd(here);  cleanup = onCleanup(@() cd(old));   %#ok<NASGU>

    fprintf('=== build_all: symmetry-adapted GPU-FTLM ===\n\n');

    %% Toolchain checks (fail early with an actionable message).
    if exist('mexcuda', 'file') ~= 2 && exist('mexcuda', 'builtin') ~= 5
        error('build_all:noMexcuda', ...
            'mexcuda not found -- the Parallel Computing Toolbox + a CUDA toolkit are required.');
    end
    try
        ngpu = gpuDeviceCount;
    catch
        ngpu = 0;
    end
    if ngpu < 1
        warning('build_all:noGPU', ...
            'No CUDA GPU detected. The kernel will still COMPILE, but the GPU driver needs a device at run time.');
    end

    %% 1. Production CUDA kernel. (No -lcublas: the Lanczos algebra is entirely
    %  hand-rolled kernels; the handle-only cuBLAS dependency was removed 2026-07.)
    %  verbose=true surfaces ptxas register/spill statistics per kernel --
    %  the data any MAX_D / launch-bounds / occupancy decision must be based on.
    if verbose
        old_flags = getenv('MW_NVCC_FLAGS');
        setenv('MW_NVCC_FLAGS', '-Xptxas -v,-warn-lmem-usage,-warn-spills');
        restore_flags = onCleanup(@() setenv('MW_NVCC_FLAGS', old_flags)); %#ok<NASGU>
    end
    build_one(@() mexcuda('cuda_lanczos_clut_block_pg_Ih.cu'), ...
              'cuda_lanczos_clut_block_pg_Ih.cu  (production block-Lanczos SpMV)');

    %% 2. schnack lookup MEX (CPU, std::thread).
    fprintf('Building schnack_query_mex.cpp (combinatorial-ranking lookup, std::thread) ...\n');
    try
        build_schnack_query_mex();          % self-contained (compiler-flag logic inside)
        fprintf('  OK\n\n');
    catch ME
        fprintf(2, '  FAILED: %s\n\n', ME.message);
        rethrow(ME);
    end

    %% 3. mmap_file MEX (out-of-core entry streaming; read-only file mapping).
    build_one(@() mex('mmap_file.cpp'), ...
              'mmap_file.cpp  (out-of-core: memory-map the entry table from NVMe)');

    %% 4. Legacy ring kernels (optional).
    if build_legacy
        build_one(@() mexcuda('-lcublas', 'cuda_lanczos_clut_block_pg.cu'), ...
                  'cuda_lanczos_clut_block_pg.cu  (legacy ring, k=0 real)');
        build_one(@() mexcuda('-lcublas', 'cuda_lanczos_clut_block_pg_cplx.cu'), ...
                  'cuda_lanczos_clut_block_pg_cplx.cu  (legacy ring, complex-k)');
    end

    fprintf('Build complete. Next: run_all_tests\n');
end

% ----------------------------------------------------------------
function build_one(fn, label)
    fprintf('Building %s ...\n', label);
    try
        fn();
        fprintf('  OK\n\n');
    catch ME
        fprintf(2, '  FAILED: %s\n\n', ME.message);
        rethrow(ME);
    end
end
