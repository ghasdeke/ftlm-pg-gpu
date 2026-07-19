function [out_path, entries_per_rep, n_entries] = external_bucket_sort( ...
        src, tgt, g, n_reps, out_path, work_dir, n_buckets, chunk, c_idx)
%EXTERNAL_BUCKET_SORT  Bounded-RAM external stable-ish sort of CLT entries by
%   target rep, written straight to disk -- the Stage-2 lever that removes the
%   ~260 GB in-RAM collect-sort peak. Thin wrapper over the incremental
%   EBS_OPEN / EBS_PUSH / EBS_FINALIZE API: feeds the full arrays in CHUNK rows
%   (the collect bond loop instead feeds its per-(bond,sign) chunks directly).
%
%   [OUT_PATH, ENTRIES_PER_REP, N] = EXTERNAL_BUCKET_SORT(SRC, TGT, G, N_REPS,
%   OUT_PATH, WORK_DIR, N_BUCKETS, CHUNK) writes OUT_PATH = [src int32][g uint16]
%   sorted (grouped) by TGT and returns the per-rep entry counts. The entries are
%   grouped by tgt (any intra-rep order -- the SpMV sums them), peak RAM ~ one
%   bucket. 6 B/entry: s=1/2 (constant c).
%
%   ... = EXTERNAL_BUCKET_SORT(..., CHUNK, C_IDX) with a non-empty uint8 C_IDX
%   (per-entry index into a small c_table; s >= 1) additionally sorts/writes a
%   third section: OUT_PATH = [src int32][g uint16][c_idx uint8] (7 B/entry).
%
%   See also EBS_OPEN, EBS_PUSH, EBS_FINALIZE, SPILL_ENTRIES_MMAP.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    ne = numel(src);
    assert(numel(tgt) == ne && numel(g) == ne, 'src/tgt/g length mismatch');
    if nargin < 7 || isempty(n_buckets), n_buckets = max(1, ceil(ne / 2e7)); end
    if nargin < 8 || isempty(chunk),     chunk     = 5e6; end
    if nargin < 9, c_idx = zeros(0, 1, 'uint8'); end
    with_c = ~isempty(c_idx);
    assert(~with_c || numel(c_idx) == ne, 'c_idx length mismatch');

    h = ebs_open(work_dir, n_reps, n_buckets, with_c);
    for s0 = 1:chunk:ne
        s1 = min(s0 + chunk - 1, ne);
        if with_c
            h = ebs_push(h, src(s0:s1), tgt(s0:s1), g(s0:s1), c_idx(s0:s1));
        else
            h = ebs_push(h, src(s0:s1), tgt(s0:s1), g(s0:s1));
        end
    end
    [out_path, entries_per_rep, n_entries] = ebs_finalize(h, out_path);
end
