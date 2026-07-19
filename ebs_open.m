function h = ebs_open(work_dir, n_reps, n_buckets, with_c)
%EBS_OPEN  Open an incremental external bucket-sort of CLT entries by target rep.
%   H = EBS_OPEN(WORK_DIR, N_REPS, N_BUCKETS) creates N_BUCKETS contiguous
%   tgt-rep range buckets (3 SoA scratch files each: src/tgt/g) under WORK_DIR
%   and returns a handle H. Feed entry chunks with EBS_PUSH, then EBS_FINALIZE
%   writes the sorted [src][g] table to disk. This lets the collect bond loop
%   stream entries to disk without ever holding them all in RAM (the Stage-2
%   lever that removes the ~260 GB in-RAM collect-sort peak).
%
%   H = EBS_OPEN(WORK_DIR, N_REPS, N_BUCKETS, WITH_C) with WITH_C = true adds a
%   4th SoA scratch file per bucket for the per-entry uint8 c-index (s >= 1:
%   the off-diagonal coefficient varies; a uint8 index into a tiny c_table).
%   EBS_FINALIZE then writes [src][g][c_idx] (7 B/entry instead of 6).
%
%   See also EBS_PUSH, EBS_FINALIZE, EXTERNAL_BUCKET_SORT, COLLECT_CLT_ENTRIES_IH.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    if nargin < 4 || isempty(with_c), with_c = false; end
    if ~exist(work_dir, 'dir'), mkdir(work_dir); end
    h.work_dir    = work_dir;
    h.n_reps      = double(n_reps);
    h.n_buckets   = double(n_buckets);
    h.bucket_size = ceil(h.n_reps / h.n_buckets);
    h.with_c      = with_c;
    h.epr         = zeros(h.n_reps, 1, 'int32');     % per-rep entry histogram
    h.psrc = strings(h.n_buckets, 1);  h.ptgt = strings(h.n_buckets, 1);  h.pg = strings(h.n_buckets, 1);
    h.fsrc = zeros(h.n_buckets, 1);    h.ftgt = zeros(h.n_buckets, 1);    h.fg = zeros(h.n_buckets, 1);
    h.pc   = strings(h.n_buckets, 1);  h.fc   = zeros(h.n_buckets, 1);
    for b = 1:h.n_buckets
        h.psrc(b) = fullfile(work_dir, sprintf('ebs_%04d_src.bin', b));
        h.ptgt(b) = fullfile(work_dir, sprintf('ebs_%04d_tgt.bin', b));
        h.pg(b)   = fullfile(work_dir, sprintf('ebs_%04d_g.bin',   b));
        h.fsrc(b) = fopen(h.psrc(b), 'w');
        h.ftgt(b) = fopen(h.ptgt(b), 'w');
        h.fg(b)   = fopen(h.pg(b),   'w');
        assert(h.fsrc(b) > 0 && h.ftgt(b) > 0 && h.fg(b) > 0, ...
            'ebs_open: cannot open bucket scratch files in %s', work_dir);
        if with_c
            h.pc(b) = fullfile(work_dir, sprintf('ebs_%04d_c.bin', b));
            h.fc(b) = fopen(h.pc(b), 'w');
            assert(h.fc(b) > 0, ...
                'ebs_open: cannot open c-index scratch file in %s', work_dir);
        end
    end
end
