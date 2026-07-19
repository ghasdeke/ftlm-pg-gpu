function [gb, node_gb] = host_avail_gb()
%HOST_AVAIL_GB  Usable host-RAM budget of THIS process in GB (NaN if unknown).
%   GB = HOST_AVAIL_GB() returns min(node availability, cgroup limit, SLURM
%   allocation):
%     * node: Windows memory().MemAvailableAllArrays; Linux MemAvailable
%       from /proc/meminfo.
%     * cgroup: v2 memory.max / v1 memory.limit_in_bytes, walked UP the
%       hierarchy (SLURM sets the limit at the job/step level while the
%       leaf often says 'max').
%     * SLURM: SLURM_MEM_PER_NODE, or SLURM_MEM_PER_CPU x SLURM_CPUS_ON_NODE.
%   Inside a SLURM step the cgroup is the BINDING limit, not the node total:
%   the 2026-07 dodecahedron collect passed a node-level check and was then
%   OOM-killed by slurmstepd at the (much smaller) cgroup cap (b6b9a5f
%   postmortem). Callers MUST keep a conservative fallback for GB = NaN
%   (no detector available at all).
%
%   [GB, NODE_GB] = HOST_AVAIL_GB() additionally returns the uncapped node
%   availability (NaN if unreadable), so callers can report "capped by the
%   job allocation" separately (ESTIMATE_FEASIBILITY's verbose note).
%
%   Extracted verbatim from ESTIMATE_FEASIBILITY (2026-07 audit) so every
%   RAM-adaptive decision (e.g. EBS_FINALIZE's parallel-worker cap) shares
%   ONE cgroup/SLURM-aware detector.
%
%   See also ESTIMATE_FEASIBILITY, EBS_FINALIZE.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    node_gb = NaN;
    if ispc
        try, m = memory; node_gb = m.MemAvailableAllArrays / 1e9; catch, end
    else
        try
            txt = fileread('/proc/meminfo');
            tok = regexp(txt, 'MemAvailable:\s*(\d+)\s*kB', 'tokens', 'once');
            if ~isempty(tok), node_gb = str2double(tok{1}) * 1024 / 1e9; end
        catch
        end
    end
    cap = min([cgroup_limit_gb(), slurm_mem_gb()]);   % NaNs ignored by min([...])
    gb  = min([node_gb, cap]);
end


% ----------------------------------------------------------------
function gb = cgroup_limit_gb()
%CGROUP_LIMIT_GB  Memory limit of THIS process's cgroup in GB; NaN if none.
%   Reads /proc/self/cgroup, then walks the hierarchy UPWARD taking the
%   minimum finite limit (SLURM sets it at the job/step level while the
%   leaf often says 'max'). Supports cgroup v2 (memory.max) and v1
%   (memory.limit_in_bytes; values >= 2^60 are "unlimited" sentinels).
    gb = NaN;
    try
        cg = fileread('/proc/self/cgroup');
    catch
        return;                                     % Windows / no cgroups
    end
    lim = NaN;
    tok = regexp(cg, '(?m)^0::(.*)$', 'tokens', 'once');
    if ~isempty(tok)                                % cgroup v2
        lim = cg_walkup_min(['/sys/fs/cgroup' strtrim(tok{1})], 'memory.max');
    else                                            % cgroup v1 (memory controller)
        tok = regexp(cg, '(?m)^\d+:memory:(.*)$', 'tokens', 'once');
        if ~isempty(tok)
            lim = cg_walkup_min(['/sys/fs/cgroup/memory' strtrim(tok{1})], ...
                                'memory.limit_in_bytes');
        end
    end
    if isfinite(lim), gb = lim / 1e9; end
end

function lim = cg_walkup_min(dirpath, fname)
    lim = NaN;
    for depth = 1 : 32
        f = [dirpath '/' fname];
        if exist(f, 'file') == 2
            v = str2double(strtrim(fileread(f)));
            if isfinite(v) && v > 0 && v < 2^60
                lim = min([lim, v]);                % min([...]) ignores NaN
            end
        end
        parent = fileparts(dirpath);
        if isempty(parent) || strcmp(parent, dirpath) || strcmp(parent, '/sys/fs')
            break;
        end
        dirpath = parent;
    end
end

function gb = slurm_mem_gb()
%SLURM_MEM_GB  Memory allocation of the current SLURM step in GB; NaN if none.
    gb = NaN;
    mpn = getenv('SLURM_MEM_PER_NODE');             % MB
    if ~isempty(mpn)
        v = str2double(mpn);
        if isfinite(v) && v > 0, gb = v * 1e6 / 1e9; return; end
    end
    mpc = getenv('SLURM_MEM_PER_CPU');              % MB
    ncp = getenv('SLURM_CPUS_ON_NODE');
    if ~isempty(mpc) && ~isempty(ncp)
        v = str2double(mpc) * str2double(ncp);
        if isfinite(v) && v > 0, gb = v * 1e6 / 1e9; end
    end
end
