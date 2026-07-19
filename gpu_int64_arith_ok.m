function tf = gpu_int64_arith_ok()
%GPU_INT64_ARITH_OK  Does the GPU support elementwise int64 arithmetic?
%
%   TF = GPU_INT64_ARITH_OK() probes (once per MATLAB session) whether the
%   selected GPU device supports the gpuArray int64 operations used by the
%   base-d_loc digit-decomposition loops: MOD plus the paired exact
%   subtraction/division. On a natively supported architecture this is
%   always true. Under CUDA FORWARD COMPATIBILITY (a GPU released after
%   the MATLAB version, e.g. an NVIDIA B200 on R2024b with
%   MW_CUDA_FORWARD_COMPATIBILITY=1) MATLAB's JIT-recompiled GPU libraries
%   LACK int64 mod -- "Input arguments of type 'int64' are not supported
%   for 'mod'" -- and the callers must take a float-exact or host path
%   instead (observed on the B200 node 2026-07-01; int64 add/sub/mul/
%   compare/bitshift and int64<->double casts DO work there, so only the
%   digit loops are affected).
%
%   Set the environment variable FTLM_FORCE_NO_GPU_INT64=1 to force
%   TF = false. This emulates the forward-compatibility limitation on a
%   natively supported GPU and is how TEST_GPU_INT64_FALLBACK exercises the
%   fallback paths on the dev box. The env var is re-read on every call
%   (cheap); only the hardware probe itself is cached.
%
%   CACHING MODEL: the probe result is cached once per MATLAB session and is
%   NOT re-run on gpuDevice(k) switches. This matches the pipeline's process
%   model -- ONE GPU per process (the kernel MEX holds per-process statics;
%   multi-GPU runs use process workers, see FTLM_ORCHESTRATE_SECTORS) -- so a
%   mid-session device switch to a different architecture is unsupported
%   anyway ('clear functions' resets the cache if you must). Callers also
%   assume int64-subscript gather D(idx) works on forward-compat devices;
%   that op is empirically proven on the B200 by the unrank routines in
%   ENUMERATE_M_ORBITS_IH_GPU (GATE A of the node bring-up).
%
%   See also MIN_IMAGE_IH_GPU, IS_SUPER_REP_IH_GPU, SCHNACK_RANK_GPU,
%            COLLECT_CLT_ENTRIES_IH_GPU, TEST_GPU_INT64_FALLBACK.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    persistent hw_ok

    if strcmp(getenv('FTLM_FORCE_NO_GPU_INT64'), '1')
        tf = false;
        return;
    end

    if isempty(hw_ok)
        try
            x  = gpuArray(int64([7; 8]));
            dg = mod(x, int64(3));                    % the op forward compat drops
            q  = (x - dg) / int64(3);                 % the paired exact division
            hw_ok = isequal(gather(dg), int64([1; 2])) && ...
                    isequal(gather(q),  int64([2; 2]));
        catch
            hw_ok = false;
        end
    end
    tf = hw_ok;
end
