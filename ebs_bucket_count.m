function nb = ebs_bucket_count(n_reps, proj_e)
%EBS_BUCKET_COUNT  Number of EBS disk buckets for an entries_on_disk collect.
%
%   NB = EBS_BUCKET_COUNT(N_REPS, PROJ_E) is the SINGLE source of the EBS
%   bucket count, shared by COLLECT_CLT_ENTRIES_IH (which sizes the REAL
%   buckets) and ESTIMATE_FEASIBILITY (whose entries_on_disk host floor
%   models the finalize in-RAM peak of ONE bucket, ~23 B/entry). The two
%   sites used to inline the same formula independently; a drift (e.g.
%   bucket target 2.5 -> 4 GB, or cap 192 -> 256) would silently skew the
%   od floor -- either spurious "Infeasible" aborts at tight sbatch --mem,
%   or a too-small floor that re-opens the b6b9a5f cgroup-OOM mode.
%
%   PROJ_E is the PROJECTED entry count (2 * n_reps * n_bonds at the call
%   sites): the 23 B/2.5-GB term bounds the finalize peak per bucket; the
%   reps-only term is the legacy lower bound; 192 buckets (4 open files
%   each) stays well under fd limits.
%
%   See also COLLECT_CLT_ENTRIES_IH, ESTIMATE_FEASIBILITY.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    nb = min(192, max([1, ceil(double(n_reps) / 1e7), ...
                          ceil(double(proj_e) * 23 / 2.5e9)]));
end
