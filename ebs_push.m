function h = ebs_push(h, src, tgt, g, c_idx)
%EBS_PUSH  Append an entry chunk to the incremental external bucket sort.
%   H = EBS_PUSH(H, SRC, TGT, G) partitions the chunk (SRC int-like, TGT in
%   1..n_reps, G uint16-like) into the tgt-rep buckets, appends each to its
%   per-bucket scratch files, and accumulates the per-rep histogram. Called once
%   per (bond, sign) chunk by the collect bond loop; the chunk can then be freed.
%
%   H = EBS_PUSH(H, SRC, TGT, G, C_IDX) additionally appends the per-entry uint8
%   c-index (s >= 1) in lockstep; requires the handle opened with WITH_C
%   (EBS_OPEN). With WITH_C the C_IDX argument is mandatory and must match the
%   chunk length; without it C_IDX must be omitted or empty.
%
%   See also EBS_OPEN, EBS_FINALIZE.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    if isempty(src), return; end
    if nargin < 5, c_idx = zeros(0, 1, 'uint8'); end
    sc = int32(src(:));  tc = int32(tgt(:));  gc = uint16(g(:));
    if h.with_c
        assert(numel(c_idx) == numel(sc), ...
            'ebs_push: c_idx length %d ~= chunk length %d (with_c handle).', ...
            numel(c_idx), numel(sc));
        cc = uint8(c_idx(:));
    else
        assert(isempty(c_idx), ...
            'ebs_push: c_idx given but the handle was opened without with_c.');
    end
    % Chunk-local histogram (2026-07 audit): accumarray over the FULL rep
    % range materialised an n_reps x 1 DOUBLE (+ its int32 copy) PER PUSH
    % (~4.3 GB transient at dodec-M-sector scale, once per (bond,sign)
    % chunk). The unique-compressed form is bounded by the chunk's number
    % of distinct reps and yields the IDENTICAL int32 counts.
    [ut, ~, iu] = unique(tc);
    h.epr(ut) = h.epr(ut) + int32(accumarray(iu, 1));
    if h.n_buckets == 1
        fwrite_chk(h.fsrc(1), sc, 'int32');  fwrite_chk(h.ftgt(1), tc, 'int32');  fwrite_chk(h.fg(1), gc, 'uint16');
        if h.with_c, fwrite_chk(h.fc(1), cc, 'uint8'); end
        return;
    end
    % ONE sort instead of n_buckets full-chunk mask scans (2026-07 audit):
    % O(n log n) vs O(n x 192) at the wave-B bucket cap. The explicit
    % running-index tiebreak makes the within-bucket order PROVABLY the
    % chunk's original order -- exactly what the sequential mask selection
    % sc(bk == b) produced -> scratch files stay BYTE-IDENTICAL (gate:
    % tests/test_external_bucket_sort checks stability explicitly).
    bk = min(h.n_buckets, floor(double(tc - 1) / h.bucket_size) + 1);
    [~, p] = sortrows([bk, (1:numel(bk))']);
    bs = bk(p);
    edges = [0; find(diff(bs) ~= 0); numel(bs)];
    for k = 1 : numel(edges) - 1
        idx = p(edges(k) + 1 : edges(k+1));
        b   = bs(edges(k) + 1);
        fwrite_chk(h.fsrc(b), sc(idx), 'int32');
        fwrite_chk(h.ftgt(b), tc(idx), 'int32');
        fwrite_chk(h.fg(b),   gc(idx), 'uint16');
        if h.with_c, fwrite_chk(h.fc(b), cc(idx), 'uint8'); end
    end
end

function fwrite_chk(fid, data, prec)
%FWRITE_CHK  fwrite that FAILS on a short write (2026-07 audit).
%   MATLAB's fwrite does NOT error on ENOSPC -- it returns a short count;
%   an unchecked short scratch write would silently truncate the entry
%   table. (Local twin of the helper in EBS_FINALIZE.)
    n = fwrite(fid, data, prec);
    assert(n == numel(data), ...
        'ebs: short write %d/%d (%s) -- disk full?', n, numel(data), prec);
end
