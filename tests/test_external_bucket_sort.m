function test_external_bucket_sort()
%TEST_EXTERNAL_BUCKET_SORT  The bounded-RAM external sort (Stage-2 collect lever)
%   produces byte-for-byte the same sorted entry table as MATLAB's in-RAM stable
%   sort(tgt), for any number of buckets and multi-chunk input. The tie-heavy
%   tgt (n_reps << n_entries) exercises stability: equal-tgt entries must keep
%   their original order, exactly as MATLAB's stable sort does.
%
%   See also EXTERNAL_BUCKET_SORT, SPILL_ENTRIES_MMAP.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    rng(7);
    n_reps = 1000;  ne = 50000;               % heavy ties (~50 entries/rep)
    src = int32(randi(2^30, ne, 1));
    tgt = int32(randi(n_reps, ne, 1));
    g   = uint16(randi(120, ne, 1));

    % Reference: the original entry multiset + the per-rep histogram. The SpMV
    % sums each rep's entries, so the external output must contain exactly the
    % same (tgt,src,g) triples GROUPED by tgt (any intra-rep order), with the
    % same entries_per_rep -> the (M,Gamma) Hamiltonian block is identical.
    epr_ref = int32(accumarray(double(tgt), 1, [n_reps, 1]));
    ref = sortrows([double(tgt), double(src), double(g)]);

    wd = tempname;
    cu = onCleanup(@() rmdirq(wd));
    ok = true;
    for nb = [1 4 7 13]
        op = [tempname, '.bin'];
        [op, epr, n] = external_bucket_sort(src, tgt, g, n_reps, op, wd, nb, 9999);  % chunk<ne -> multi-chunk
        fid = fopen(op, 'r');
        so = fread(fid, ne, '*int32');  go = fread(fid, ne, '*uint16');
        fclose(fid);  delete(op);
        tgt_out = repelem((1:n_reps)', double(epr));     % tgt implied by the grouped layout
        ext = sortrows([double(tgt_out), double(so), double(go)]);
        de = isequal(ext, ref) && isequal(epr, epr_ref) && (n == ne);
        fprintf('  n_buckets=%2d : entries grouped-by-tgt match=%d\n', nb, de);
        ok = ok && de;
    end
    assert(ok, 'external_bucket_sort entries/grouping != reference');

    %% WITH_C variant (s>=1, companion A): a per-entry uint8 c-index rides along
    %  -> the file gains a third section [src int32][g uint16][c_idx uint8]
    %  (7 B/entry) and the (tgt,src,g,c) quadruples must match grouped by tgt.
    cidx  = uint8(randi(7, ne, 1));
    ref_c = sortrows([double(tgt), double(src), double(g), double(cidx)]);
    for nb = [1 4 7 13]
        op = [tempname, '.bin'];
        [op, epr, n] = external_bucket_sort(src, tgt, g, n_reps, op, wd, nb, 9999, cidx);
        d = dir(op);
        sz_ok = (d.bytes == ne * 7);                     % [src 4][g 2][c 1]
        fid = fopen(op, 'r');
        so = fread(fid, ne, '*int32');  go = fread(fid, ne, '*uint16');
        co = fread(fid, ne, '*uint8');
        fclose(fid);  delete(op);
        tgt_out = repelem((1:n_reps)', double(epr));
        ext = sortrows([double(tgt_out), double(so), double(go), double(co)]);
        de = isequal(ext, ref_c) && isequal(epr, epr_ref) && (n == ne) && sz_ok;
        fprintf('  n_buckets=%2d : entries+c_idx grouped-by-tgt match=%d (7 B/entry=%d)\n', nb, de, sz_ok);
        ok = ok && de;
    end
    assert(ok, 'external_bucket_sort with c_idx != reference');

    %% PARALLELES Finalize (2026-07-10): FTLM_EBS_PAR=2 (parfor ueber Buckets,
    %  disjunkte Regionen) muss eine BYTE-IDENTISCHE Datei zum seriellen Pfad
    %  (FTLM_EBS_PAR=0) liefern. Uebersprungen ohne Parallel Computing Toolbox.
    if ~isempty(ver('parallel'))
        env_old = getenv('FTLM_EBS_PAR');
        cu_env = onCleanup(@() setenv('FTLM_EBS_PAR', env_old));
        setenv('FTLM_EBS_PAR', '0');
        op_s = [tempname, '.bin'];
        [op_s, epr_s, n_s] = external_bucket_sort(src, tgt, g, n_reps, op_s, wd, 13, 9999, cidx);
        setenv('FTLM_EBS_PAR', '2');
        op_p = [tempname, '.bin'];
        [op_p, epr_p, n_p] = external_bucket_sort(src, tgt, g, n_reps, op_p, wd, 13, 9999, cidx);
        fs = fopen(op_s, 'r'); bs = fread(fs, inf, '*uint8'); fclose(fs);
        fp = fopen(op_p, 'r'); bp = fread(fp, inf, '*uint8'); fclose(fp);
        delete(op_s); delete(op_p);
        de = isequal(bs, bp) && isequal(epr_s, epr_p) && (n_s == n_p);
        fprintf('  paralleles Finalize (2 Worker): byte-identisch=%d\n', de);
        assert(de, 'paralleles ebs_finalize != serielles Ergebnis');
    else
        fprintf('  paralleles Finalize: uebersprungen (keine Parallel Computing Toolbox)\n');
    end

    %% LEERE BUCKETS + Rep-LUECKEN (K6c, 2026-07): n_buckets >> belegte Reps
    %  bzw. tgt-Werte mit Luecken -- ebs_finalize muss leere Buckets
    %  (sizes(b)==0) und leere Zwischen-Reps korrekt ueberspringen; die
    %  Offset-Kette cumsum(sizes) enthaelt dann Nullen. Die stabile
    %  per-Bucket-Sortierung erhaelt die Push-Reihenfolge innerhalb jeder Rep
    %  fuer JEDES n_buckets -> Datei BYTE-identisch zur nb=1-Referenz.
    % Restore the TRUE original: with PCT the par-finalize section above has
    % already captured it (env_old) and left '2' set -- re-capturing here
    % would "restore" that transient at teardown.
    if exist('env_old', 'var'), env_old2 = env_old;
    else,                       env_old2 = getenv('FTLM_EBS_PAR'); end
    cu_env2 = onCleanup(@() setenv('FTLM_EBS_PAR', env_old2));
    setenv('FTLM_EBS_PAR', '0');
    rng(11);
    ne_e  = 200;
    src_e = int32(randi(2^30, ne_e, 1));
    g_e   = uint16(randi(120, ne_e, 1));
    vals  = int32([1; 7; 9]);
    efix = struct('label', {'nb=64 > 10 Reps', 'Rep-Luecken {1,7,9}'}, ...
                  'tgt',   {int32(randi(10, ne_e, 1)), vals(randi(3, ne_e, 1))}, ...
                  'nb',    {64, 5});
    for f = efix
        epr_e = int32(accumarray(double(f.tgt), 1, [10, 1]));
        [b_ref1, epr_r1] = ebs_run_bytes(src_e, f.tgt, g_e, 10, wd, 1);
        [b_out,  epr_o]  = ebs_run_bytes(src_e, f.tgt, g_e, 10, wd, f.nb);
        de = isequal(b_out, b_ref1) && isequal(epr_r1, epr_e) && isequal(epr_o, epr_e);
        fprintf('  leere Buckets (%s, nb=%d): byte-identisch=%d\n', f.label, f.nb, de);
        assert(de, 'ebs_finalize: leere Buckets/Reps aendern das Ergebnis (%s)', f.label);
    end

    if ~isempty(ver('parallel'))
        %% Leere Buckets im PARALLELEN Finalize: deckt den nb<=0-Fruehausstieg
        %  von finalize_one_bucket (parfor-Worker) ab.
        setenv('FTLM_EBS_PAR', '2');
        [b_p, epr_p2] = ebs_run_bytes(src_e, efix(1).tgt, g_e, 10, wd, 64);
        setenv('FTLM_EBS_PAR', '0');
        [b_s, epr_s2] = ebs_run_bytes(src_e, efix(1).tgt, g_e, 10, wd, 64);
        de = isequal(b_p, b_s) && isequal(epr_p2, epr_s2);
        fprintf('  leere Buckets parallel (nb=64): byte-identisch=%d\n', de);
        assert(de, 'parallele leere Buckets != seriell');

        %% PARFOR-ABBRUCH-INJEKTION (offizieller Hook FTLM_EBS_FAIL_BUCKET,
        %  2026-07-Audit): Bucket 7 wirft im Worker VOR jedem Write -> der
        %  parfor bricht ab -> catch in ebs_finalize -> Warnung
        %  'ebs_finalize:parFallback' -> serieller Fallback uebernimmt
        %  (Two-Phase-Commit: fertige .done-Buckets werden uebersprungen,
        %  der Rest neu geschrieben). Datei byte-identisch zur seriellen
        %  Referenz, Scratch inkl. .done-Marker geraeumt. Der Pool wird
        %  ERST NACH dem setenv gestartet (frische Worker erben das Env des
        %  Clients; ein laufender Pool saehe die Variable nicht).
        delete(gcp('nocreate'));
        wd_fb = tempname;          % frisches Scratch-Dir fuer die Leerheits-Pruefung
        cu_fb = onCleanup(@() rmdirq(wd_fb));
        fb_old = getenv('FTLM_EBS_FAIL_BUCKET');
        cu_fbe = onCleanup(@() setenv('FTLM_EBS_FAIL_BUCKET', fb_old));
        setenv('FTLM_EBS_PAR', '0');
        [b_ref13, epr_ref13] = ebs_run_bytes(src, tgt, g, n_reps, wd_fb, 13, cidx);
        setenv('FTLM_EBS_PAR', '2');
        setenv('FTLM_EBS_FAIL_BUCKET', '7');
        lastwarn('');
        wtxt = evalc(['[b_fb, epr_fb] = ebs_run_bytes(src, tgt, g, n_reps, ' ...
                      'wd_fb, 13, cidx);']);
        setenv('FTLM_EBS_FAIL_BUCKET', '');
        % Robust gegen NACHFOLGENDE Warnungen (Windows-delete-Races
        % ueberschrieben lastwarn, 2026-07-11): im gefangenen Output nach
        % der Fallback-Warnung suchen statt lastwarn zu vertrauen.
        assert(contains(wtxt, 'parFallback') || contains(wtxt, 'serieller Fallback'), ...
            'Fehler-Injektion loeste den Fallback nicht aus (Output: %s)', wtxt(1:min(200,end)));
        % Pool ZUERST schliessen: auf Windows halten Worker transiente
        % Datei-/Verzeichnis-Handles, bis sie enden -- der Raeumcheck lief
        % sonst in eine Race (2026-07-11). Danach mit kurzem Retry pruefen.
        delete(gcp('nocreate'));
        scraps = dir(fullfile(wd_fb, 'ebs_*'));
        for rtry = 1:10
            if isempty(scraps), break; end
            pause(0.5);
            % Nach Pool-Ende sind Worker-Handles frei -> Nachloeschen der
            % Dateien, deren best-effort-del im Fallback an Windows-Locks
            % scheiterte (auf Linux ist die Liste hier immer schon leer).
            for sc = scraps'
                try, delete(fullfile(wd_fb, sc.name)); catch, end
            end
            scraps = dir(fullfile(wd_fb, 'ebs_*'));
        end
        assert(isempty(scraps), 'Fallback hat %d Scratch-/Marker-Dateien nicht geraeumt', numel(scraps));
        de = isequal(b_fb, b_ref13) && isequal(epr_fb, epr_ref13);
        fprintf('  parfor-Abbruch (FAIL_BUCKET=7): Fallback seriell, byte-identisch=%d\n', de);
        assert(de, 'parFallback-Ergebnis != serielle Referenz');

        %% VERTRAGSTESTS FTLM_EBS_PAR (Doku in ebs_finalize):
        %  (i)   'N' mit n_buckets<4 -> immer seriell, KEIN Pool;
        %  (iii) 'auto' -> wie '' (bei <4 GB Tabelle: seriell, kein Pool).
        setenv('FTLM_EBS_PAR', '2');
        [b_c2, epr_c2] = ebs_run_bytes(src, tgt, g, n_reps, wd_fb, 2, cidx);
        assert(isempty(gcp('nocreate')), 'n_buckets<4 startete faelschlich einen Pool');
        setenv('FTLM_EBS_PAR', '0');
        [b_c2s, epr_c2s] = ebs_run_bytes(src, tgt, g, n_reps, wd_fb, 2, cidx);
        assert(isequal(b_c2, b_c2s) && isequal(epr_c2, epr_c2s), ...
            'nb<4-Seriell-Erzwingung liefert falsches Ergebnis');
        setenv('FTLM_EBS_PAR', 'auto');
        [b_au, epr_au] = ebs_run_bytes(src, tgt, g, n_reps, wd_fb, 13, cidx);
        assert(isempty(gcp('nocreate')), '''auto'' startete bei Mini-Tabelle einen Pool');
        assert(isequal(b_au, b_ref13) && isequal(epr_au, epr_ref13), ...
            '''auto'' (Mini-Tabelle, seriell) != serielle Referenz');
        fprintf('  Vertragstests: nb<4 seriell, ''auto'' Mini-Tabelle seriell -- OK\n');
    else
        fprintf('  parfor-Abbruch/Vertragstests: uebersprungen (keine Parallel Computing Toolbox)\n');
    end

    %% (ii)/(iv) auch OHNE PCT lauffaehig: '0' und Nicht-Zahlen (str2double ->
    %  NaN -> par_workers=0) -> seriell, kein Crash, identisches Ergebnis.
    setenv('FTLM_EBS_PAR', 'quatsch');
    [b_q, epr_q] = ebs_run_bytes(src, tgt, g, n_reps, wd, 13, cidx);
    setenv('FTLM_EBS_PAR', '0');
    [b_0, epr_0] = ebs_run_bytes(src, tgt, g, n_reps, wd, 13, cidx);
    assert(isequal(b_q, b_0) && isequal(epr_q, epr_0), ...
        'FTLM_EBS_PAR=''quatsch'' (NaN) != seriell');
    fprintf('  FTLM_EBS_PAR=''quatsch'' -> seriell, kein Crash -- OK\n');

    fprintf(['PASS: external_bucket_sort == in-RAM entry set, grouped by tgt ', ...
             '(multi-bucket/chunk, +c_idx, par-finalize, leere Buckets, ', ...
             'parFallback, Env-Vertrag).\n']);
end

% ----------------------------------------------------------------
function [bytes, epr] = ebs_run_bytes(src, tgt, g, n_reps, wd, nb, cidx)
%   Ein external_bucket_sort-Lauf -> Ausgabedatei als uint8-Byte-Spalte
%   (Byte-Identitaets-Vergleiche) + entries_per_rep; Datei wird geloescht.
    op = [tempname, '.bin'];
    if nargin < 7 || isempty(cidx)
        [op, epr] = external_bucket_sort(src, tgt, g, n_reps, op, wd, nb, 9999);
    else
        [op, epr] = external_bucket_sort(src, tgt, g, n_reps, op, wd, nb, 9999, cidx);
    end
    fid = fopen(op, 'r');  bytes = fread(fid, inf, '*uint8');  fclose(fid);
    delete(op);
end

function rmdirq(d)
    try, if exist(d, 'dir'), rmdir(d, 's'); end, catch, end
end
