function s = pc_gather_struct(s)
%PC_GATHER_STRUCT  Gather any gpuArray fields of a struct to host.
%   Used before writing the precompute cache (cache_M / entries_M) to disk so
%   the cached file is portable and re-loadable without a live GPU context. For
%   the production 'host' collect path the fields are already host arrays, so
%   this is a no-op; it is defensive for any path that returns gpuArray fields.
%
%   See also FTLM_OBSERVABLES_PG_GPU_IH (precompute_cache option).
% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    f = fieldnames(s);
    for i = 1:numel(f)
        v = s.(f{i});
        if isa(v, 'gpuArray')
            s.(f{i}) = gather(v);
        end
    end
end
