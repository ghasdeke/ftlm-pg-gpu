function mem_snapshot(label, gpu_h)
%MEM_SNAPSHOT  Print a one-line GPU + host memory snapshot.
%
%   MEM_SNAPSHOT(LABEL, GPU_H) waits for any pending GPU operations to
%   complete, then prints LABEL plus the current free / used GPU memory
%   (from GPU_H) and the MATLAB-side resident memory (Windows-only via
%   memory()).
%
%   The wait() call is essential: querying gpuDevice().AvailableMemory
%   on a busy device returns whatever was free at the LAST sync point,
%   so without the wait we'd see misleading numbers during the Lanczos
%   loop.
%
%   This helper exists to diagnose memory-budget questions for the
%   icosidodecahedron-scale runs. It is called only when the input
%   file sets `mem_diag = true`; in production the cost is one
%   wait(gpu_h) + a printf, so a few ms per call.
%
%   Example output (single line, truncated for clarity):
%       [after init_skel_ref       ] GPU free 17.50 GB / used  2.50 GB  | host MATLAB  4.20 GB

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    if nargin < 2 || isempty(gpu_h)
        gpu_h = gpuDevice();
    end

    wait(gpu_h);                              % flush queued ops for accurate readings
    free_b  = gpu_h.AvailableMemory;
    total_b = gpu_h.TotalMemory;
    used_b  = total_b - free_b;
    free_gb = free_b  / 1e9;
    used_gb = used_b  / 1e9;

    if ispc
        try
            m = memory;
            host_used_gb = m.MemUsedMATLAB / 1e9;
            host_str = sprintf('host MATLAB %5.2f GB', host_used_gb);
        catch
            host_str = 'host MATLAB    n/a';
        end
    else
        host_str = 'host MATLAB    n/a';
    end

    fprintf('    [%-30s] GPU free %5.2f GB / used %5.2f GB  | %s\n', ...
            label, free_gb, used_gb, host_str);
end
