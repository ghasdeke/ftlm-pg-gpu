function CHUNK = adaptive_enum_chunk(N_sites, use_gpu)
%ADAPTIVE_ENUM_CHUNK  Enumeration-scan chunk length, sized to the free VRAM.
%
%   Per-state device budget: ~N_SITES*8 B (the [N_sites x n] double digits
%   array in IS_SUPER_REP_IH_GPU) + ~72 B (unrank/min_image int64 temporaries
%   and the stage-1 block-matmul share). Half the free memory, clamped to
%   [4e6, 256e6] -- AND, independent of VRAM, hard-capped so the digit
%   matrix stays below MATLAB's 2^31-1 elements-per-gpuArray limit
%   (0.9 * 2^31 / N_SITES; a production run hit exactly this).
%   The cap applies on EVERY path, including the no-probe fallback.
%
%   Chunking only partitions the scan: kept set and ordering are
%   byte-identical for any chunk size (gated by the suite).
%
%   See also GPU_FREE_BYTES, ENUMERATE_M_ORBITS_IH_GPU,
%            TEST_GPU_SIZING_INVARIANTS.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    CHUNK = int64(32e6);                    % CPU / no-probe default
    if use_gpu
        free_b = gpu_free_bytes();
        if isfinite(free_b)
            bps   = N_sites * 8 + 72;
            CHUNK = int64(min(max(round(0.5 * free_b / bps), 4e6), 256e6));
        end
    end
    % Device-INDEPENDENT element cap -- outside any probe/fallback branch.
    CHUNK = min(CHUNK, int64(floor(2^31 / double(N_sites) * 0.9)));
end
