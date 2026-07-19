function [out_path, entries_per_rep, n_entries] = ebs_finalize(h, out_path)
%EBS_FINALIZE  Finish the incremental external bucket sort -> sorted entry file.
%   [OUT_PATH, ENTRIES_PER_REP, N] = EBS_FINALIZE(H, OUT_PATH) sorts each bucket
%   by target rep ONE AT A TIME in RAM, appends sorted src/g (and, for a WITH_C
%   handle, the uint8 c-index) to per-column scratch streams, then
%   stream-concatenates them into OUT_PATH = [ all src int32 ][ all g uint16 ]
%   (+ [ all c_idx uint8 ] when WITH_C) -- the spill/mmap layout, 6 B/entry or
%   7 B/entry. Deletes all scratch files. Peak RAM ~ one bucket. The intra-rep
%   order is irrelevant (the SpMV sums a rep's entries), so any grouping-by-tgt
%   is correct.
%
%   PARALLEL SAFETY: the parallel finalize auto-enables only on a LOCAL
%   filesystem with a LOCAL (single-node) pool. Bucket regions are byte-exact,
%   NOT page-aligned -- on a network FS (NFS/BeeGFS/Lustre/GPFS/CIFS) every
%   client page-caches with read-modify-write, so two nodes writing different
%   byte ranges of the SAME page silently clobber each other at every bucket
%   boundary. FTLM_EBS_PAR=N forces parallel regardless (explicit opt-in).
%
%   See also EBS_OPEN, EBS_PUSH, SPILL_ENTRIES_MMAP, HOST_AVAIL_GB.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    for b = 1:h.n_buckets
        try, fclose(h.fsrc(b)); catch, end
        try, fclose(h.ftgt(b)); catch, end
        try, fclose(h.fg(b));   catch, end
        if h.with_c, try, fclose(h.fc(b)); catch, end, end
    end

    %% DIRECT-WRITE finalize (2026-07-03, wave-2 [10]): the three region
    %  sizes are known up front, so every bucket's sorted columns are
    %  fseek+fwritten STRAIGHT into their [src][g][c] regions of OUT_PATH.
    %  No per-column scratch streams, no final concatenation pass: the
    %  on-disk transient drops from ~2x table (+ a full extra read+write;
    %  this doubled transient blew the 600-GB home quota on 2026-07-03) to
    %  ~1x table + one bucket, and buckets are deleted as consumed.
    %  Output file is BYTE-IDENTICAL (same bucket order, same permutation).
    sizes = zeros(h.n_buckets, 1);
    for b = 1:h.n_buckets
        d = dir(h.psrc(b));  if ~isempty(d), sizes(b) = d.bytes / 4; end
    end
    N = sum(sizes);
    if h.with_c, total_bytes = 7 * N; else, total_bytes = 6 * N; end
    % Pre-size the output file: MATLAB's fseek FAILS (status -1, position
    % unchanged) beyond EOF, so region writes into a growing 'w' file would
    % silently land at the wrong offset (caught by test_external_bucket_sort).
    % Java setLength extends instantly (sparse where the FS supports it);
    % without a JVM, fall back to a chunked zero-fill append.
    presize_file(out_path, total_bytes);

    %% PARALLEL finalize (2026-07-10): every bucket writes DISJOINT
    %  [src][g][c]-Regionen, deren Offsets vorab aus cumsum(sizes) bekannt
    %  sind -- die Buckets sind damit vollstaendig unabhaengig (eigenes
    %  Datei-Handle pro Worker) und das Ergebnis ist BYTE-IDENTISCH zum
    %  seriellen Pfad (gleiche per-Bucket-Sortierpermutation, gleiche
    %  Regionen). Ein erfolgreicher Worker loescht seine Scratches sofort
    %  und hinterlaesst einen .done-Marker, den der serielle Fallback
    %  ueberspringt (two-phase commit; haelt den Disk-Peak auf serieller
    %  Hoehe statt Scratch+Output koexistent, wave-B: ~271 + 172 GB).
    %  Env FTLM_EBS_PAR: ''/auto -> parallel ab 4 GB Tabelle wenn PCT da;
    %  '0' -> seriell erzwingen; 'N' -> Pool/Concurrency mit N Workern.
    offs = cumsum([0; sizes(1:end-1)]);
    par_workers = 0;
    pe = getenv('FTLM_EBS_PAR');
    if strcmpi(strtrim(pe), 'auto'), pe = ''; end   % K6c: 'auto' == ungesetzt
    if h.n_buckets >= 4
        if ~isempty(pe)
            par_workers = round(str2double(pe));
            if ~isfinite(par_workers), par_workers = 0; end
        elseif total_bytes > 4e9 && ~isempty(ver('parallel'))
            par_workers = -1;   % auto: worker count from the host budget below
        end
    end
    if par_workers ~= 0
        try
            % Safety guards (2026-07 audit): both error into the existing
            % catch below -> serial fallback.
            pool = gcp('nocreate');
            if ~isempty(pool) && isprop(pool, 'Cluster') && ...
                    ~isa(pool.Cluster, 'parallel.cluster.Local')
                % An adopted MULTI-NODE pool (ftlm_orchestrate_sectors creates
                % pools!) must never run this: per-node client caches make the
                % non-page-aligned bucket regions false-share pages (see the
                % PARALLEL SAFETY help note). Thread pools have no Cluster
                % property and are inherently single-node -> allowed.
                error('ebs_finalize:nonLocalPool', 'non-local parallel pool -> serial finalize');
            end
            if isempty(pe) && fs_is_network(fileparts(char(out_path)))
                % Network-FS guard in AUTO mode only; explicit FTLM_EBS_PAR=N
                % stays an opt-in override for setups known to be safe.
                error('ebs_finalize:netFS', 'network filesystem output -> serial finalize');
            end
            % Worker cap from the LIVE host budget (2026-07 audit): each
            % worker peaks at ~28 B/entry of its bucket (read columns + sort
            % permutation + permuted copies). An unsized parpool() defaults
            % to the core count (64+ on cluster nodes) -> ~190 GB transient
            % at the 2.5-GB bucket target = the b6b9a5f cgroup-OOM mode.
            % host_avail_gb() is cgroup/SLURM-aware; without any detector
            % fall back to a conservative 16 GB. Spend at most 25% of the
            % budget on finalize workers, never more than 12.
            bucket_peak = 28 * max(sizes) + 512e6;
            avail_gb = host_avail_gb();
            if ~isfinite(avail_gb), avail_gb = 16; end
            nw_cap = max(1, min(floor(0.25 * avail_gb * 1e9 / bucket_peak), 12));
            if par_workers > 0, nw = par_workers; else, nw = nw_cap; end
            if isempty(pool), parpool(min(nw, h.n_buckets)); end
            psrc = h.psrc; ptgt = h.ptgt; pg = h.pg; pc = h.pc; wc = h.with_c;
            % parfor's M argument throttles concurrency even on an adopted
            % larger (orchestrator) pool -- pool size alone would not.
            parfor (b = 1:h.n_buckets, nw)
                finalize_one_bucket(out_path, psrc(b), ptgt(b), pg(b), ...
                                    wc, pc(b), sizes(b), offs(b), N, b);
            end
            % Size check BEFORE the marker cleanup: on failure the catch
            % falls back to the serial path, which still needs the markers.
            db = dir(char(out_path));  ob = 0;
            if ~isempty(db), ob = db.bytes; end
            assert(ob == total_bytes, ...
                'ebs_finalize: output size %d ~= expected %d', ob, total_bytes);
            for b = 1:h.n_buckets
                del([char(h.psrc(b)) '.done']);
                % Workers already deleted their scratch; this sweeps the
                % scratch of EMPTY buckets (workers return early on nb=0).
                del(h.psrc(b));  del(h.ptgt(b));  del(h.pg(b));
                if h.with_c, del(h.pc(b)); end
            end
            entries_per_rep = h.epr;
            n_entries = N;
            return;
        catch err
            warning('ebs_finalize:parFallback', ...
                'Paralleles Finalize fehlgeschlagen (%s) -- serieller Fallback.', ...
                err.message);
        end
    end

    of = fopen(out_path, 'r+');  assert(of > 0, 'ebs_finalize: cannot open %s', out_path);
    cur = 0;
    for b = 1:h.n_buckets
        nb = sizes(b);
        % Two-phase commit (2026-07 audit): a .done marker means a parallel
        % worker already wrote this bucket's disjoint regions completely and
        % byte-identically (and deleted its scratch) -- just advance cur.
        done_p = [char(h.psrc(b)) '.done'];
        if exist(done_p, 'file')
            cur = cur + nb;
            del(done_p);
            del(h.psrc(b));  del(h.ptgt(b));  del(h.pg(b));
            if h.with_c, del(h.pc(b)); end
            continue;
        end
        if nb > 0
            fid = fopen(h.psrc(b), 'r'); sb = fread(fid, nb, '*int32');  fclose(fid);
            fid = fopen(h.ptgt(b), 'r'); tb = fread(fid, nb, '*int32');  fclose(fid);
            fid = fopen(h.pg(b),   'r'); gb = fread(fid, nb, '*uint16'); fclose(fid);
            [~, p] = sort(tb);
            fseek(of, 4 * cur, 'bof');            fwrite_chk(of, sb(p), 'int32');
            fseek(of, 4 * N + 2 * cur, 'bof');    fwrite_chk(of, gb(p), 'uint16');
            if h.with_c
                fid = fopen(h.pc(b), 'r'); cb = fread(fid, nb, '*uint8'); fclose(fid);
                assert(numel(cb) == nb, 'ebs_finalize: c-index bucket %d short', b);
                fseek(of, 6 * N + cur, 'bof');    fwrite_chk(of, cb(p), 'uint8');
            end
            cur = cur + nb;
        end
        del(h.psrc(b));  del(h.ptgt(b));  del(h.pg(b));
        if h.with_c, del(h.pc(b)); end
    end
    st = fclose(of);
    assert(st == 0, 'ebs_finalize: close of %s failed (flush error -- disk full?)', out_path);
    db = dir(char(out_path));  ob = 0;
    if ~isempty(db), ob = db.bytes; end
    assert(ob == total_bytes, 'ebs_finalize: output size %d ~= expected %d', ob, total_bytes);

    entries_per_rep = h.epr;
    n_entries = cur;
end

function del(p)
%DEL  Best-effort-Loeschung: komplett STILL. delete() meldet auf Windows
%  transiente Worker-Handles als WARNUNG (MATLAB:DELETE:Permission), nicht
%  als Fehler -- das try/catch faengt sie nicht, sie ueberschrieb lastwarn
%  und brach damit die Warn-Assertion des Injektions-Subtests (2026-07-11).
    ws = warning('off', 'MATLAB:DELETE:Permission');
    wf = warning('off', 'MATLAB:DELETE:FileNotFound');
    try, if exist(p, 'file'), delete(p); end, catch, end
    warning(ws); warning(wf);
end

function fwrite_chk(fid, data, prec)
%FWRITE_CHK  fwrite that FAILS on a short write (2026-07 audit).
%   MATLAB's fwrite does NOT error on ENOSPC -- it returns a short count.
%   The pre-sized output is SPARSE (setLength allocates no blocks), so a
%   full disk mid-finalize would otherwise leave silent zero-holes that
%   mmap_cidx later reads as valid entries (silent physics corruption).
    n = fwrite(fid, data, prec);
    assert(n == numel(data), ...
        'ebs: short write %d/%d (%s) -- disk full? (presize is sparse; blocks are allocated at write time)', ...
        n, numel(data), prec);
end

function finalize_one_bucket(out_path, psrc, ptgt, pgf, with_c, pcf, nb, off, N, b_idx)
%FINALIZE_ONE_BUCKET  Ein Bucket -> seine disjunkten Regionen von OUT_PATH.
%  Identische Sortierung/Schreiblogik wie der serielle Pfad; off = Entry-
%  Offset des Buckets (cumsum der Vorgaenger), N = Gesamt-Entryzahl.
    if nb <= 0, return; end
    fb = getenv('FTLM_EBS_FAIL_BUCKET');   % test-only fault injection: throw
    if ~isempty(fb) && str2double(fb) == b_idx   % BEFORE any write, so the
        error('ebs_finalize:testFault', ...      % serial fallback can redo it
            'injected test fault in bucket %d (FTLM_EBS_FAIL_BUCKET)', b_idx);
    end
    fid = fopen(psrc, 'r'); sb = fread(fid, nb, '*int32');  fclose(fid);
    fid = fopen(ptgt, 'r'); tb = fread(fid, nb, '*int32');  fclose(fid);
    fid = fopen(pgf,  'r'); gb = fread(fid, nb, '*uint16'); fclose(fid);
    [~, p] = sort(tb);
    of = fopen(out_path, 'r+');
    assert(of > 0, 'ebs_finalize: worker cannot open %s', out_path);
    fseek(of, 4 * off, 'bof');            fwrite_chk(of, sb(p), 'int32');
    fseek(of, 4 * N + 2 * off, 'bof');    fwrite_chk(of, gb(p), 'uint16');
    if with_c
        fid = fopen(pcf, 'r'); cb = fread(fid, nb, '*uint8'); fclose(fid);
        assert(numel(cb) == nb, 'ebs_finalize: c-index bucket short (%s)', pcf);
        fseek(of, 6 * N + off, 'bof');    fwrite_chk(of, cb(p), 'uint8');
    end
    st = fclose(of);
    assert(st == 0, 'ebs_finalize: worker close of %s failed (flush error)', out_path);
    % Two-phase commit (2026-07 audit): the regions are fully written ->
    % drop a .done marker (the serial fallback skips such buckets) and
    % delete OUR scratch NOW, so scratch shrinks while the output grows
    % (parallel disk peak ~= serial peak instead of scratch+output
    % coexisting until after the parfor).
    fm = fopen([char(psrc) '.done'], 'w');
    if fm > 0, fclose(fm); end
    del(psrc);  del(ptgt);  del(pgf);
    if with_c, del(pcf); end
end

function tf = fs_is_network(dirpath)
%FS_IS_NETWORK  True when DIRPATH sits on a known network filesystem.
%   Linux only (GNU stat -f -c %T); everywhere else -- and whenever stat
%   fails -- returns false, i.e. the guard is a no-op (matches the pre-guard
%   behaviour; Windows dev box and local /tmp NVMe stay parallel-eligible).
    tf = false;
    if ~isunix, return; end
    [st, fst] = system(['stat -f -c %T "' char(dirpath) '"']);
    if st == 0 && ~isempty(regexp(strtrim(fst), ...
            '^(nfs|beegfs|lustre|cifs|smb2?|fuse|gpfs)', 'once'))
        tf = true;
    end
end

function presize_file(p, nbytes)
    try
        raf = java.io.RandomAccessFile(p, 'rw');
        raf.setLength(nbytes);
        raf.close();
    catch
        fid = fopen(p, 'w');  assert(fid > 0, 'ebs_finalize: cannot create %s', p);
        CH = 64 * 2^20;  z = zeros(CH, 1, 'uint8');  left = nbytes;
        while left > 0
            k = min(left, CH);
            fwrite_chk(fid, z(1:k), 'uint8');
            left = left - k;
        end
        fclose(fid);
    end
end
