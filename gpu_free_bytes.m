function b = gpu_free_bytes()
%GPU_FREE_BYTES  Free VRAM (bytes) of the selected GPU, with a test override.
%
%   ALL VRAM-adaptive sizing decisions read the free memory through this
%   helper, so the env var FTLM_FAKE_FREE_VRAM_GB can emulate ANY card size
%   (8-GB workstation ... 200-GB B200) on the dev box. Rationale: the
%   failure class "behavior depends on the GPU size" has no natural local
%   test coverage (two 2^31 incidents shipped despite a green suite). TEST_GPU_SIZING_INVARIANTS
%   sweeps the sizing formulas across fake sizes.
%
%   Returns Inf when no GPU is present/queryable: VRAM-adaptive callers then
%   fall back to their own defaults, and ELEMENT-limit caps (2^31-1 per
%   gpuArray variable) must be applied by the caller independently -- they
%   do not depend on VRAM.
%
%   See also ADAPTIVE_ENUM_CHUNK, RUN_FTLM_PG_SECTOR_GPU_IH,
%            BUILD_ENTRY_SKELETON_IH, TEST_GPU_SIZING_INVARIANTS.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    fake = getenv('FTLM_FAKE_FREE_VRAM_GB');
    if ~isempty(fake)
        b = str2double(fake) * 1e9;
        assert(isfinite(b) && b > 0, ...
            'gpu_free_bytes: FTLM_FAKE_FREE_VRAM_GB must be a positive number, got "%s".', fake);
        return;
    end
    try
        b = double(gpuDevice().AvailableMemory);
    catch
        b = Inf;
    end
end
