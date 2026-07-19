function tl = build_tiled_entries(entries, rep_offsets, B, target_bytes)
%BUILD_TILED_ENTRIES  R2-v1: (src_tile, tgt)-sortierte Entry-Tabelle + Run-CSR.
%
%   TL = BUILD_TILED_ENTRIES(ENTRIES, REP_OFFSETS, B, TARGET_BYTES) ordnet die
%   tgt-sortierte Entry-Tabelle in src-Tiles um, deren V-Fenster (interleaved,
%   B Spalten, fp32) je <= TARGET_BYTES bleibt -- der getilte SpMV-Kernel
%   haelt so alle V[src]-Gathers eines Launches L2-resident. Innerhalb eines
%   Tiles liegen gleiche tgt als zusammenhaengende RUNS; der Kernel faehrt
%   einen Thread pro (Run, kp) und akkumuliert ADDITIV in W[tgt].
%
%   Numerik: Klasse C -- die Summationsreihenfolge pro tgt wird zur
%   Tile-Ordnung (deterministisch: aufsteigende Tiles, im Tile aufsteigende
%   Entry-Position). Die Sum Rule bleibt exakt; E/w aendern sich auf
%   FP32-Rundungsniveau (Validierungskontrakt wie beim Realify-Uebergang).
%
%   Eingaben:
%     entries      Struct aus COLLECT_CLT_ENTRIES_IH (host, tgt-sortiert):
%                  src_sorted/tgt_sorted/g_sorted [+ c_idx | c_sorted].
%     rep_offsets  int64 [n_reps+1] Basis-Offsets (aufsteigend; letzter
%                  Eintrag = n_basis) ODER [n_reps] (dann wird n_basis aus
%                  max nicht benoetigt -- nur Differenzen der Fensterkanten).
%     B            Krylov-Blockbreite, fuer die das Fenster gelten soll.
%     target_bytes Fensterbudget in Bytes (z.B. 40e6).
%
%   Ausgabe TL:
%     .perm        uint32/uint64 Permutation der Entries (fuer src/g/c-Arrays)
%     .tile_lo     int32 [n_tiles]   erster Rep des Tiles (1-basiert)
%     .tile_hi     int32 [n_tiles]   letzter Rep + 1
%     .run_ptr     int64 [n_runs+1]  Entry-Offsets der Runs (0-basiert)
%     .run_tgt     int32 [n_runs]    Ziel-Rep des Runs (1-basiert)
%     .tile_run_ptr int64 [n_tiles+1] Run-Offsets pro Tile (0-basiert)
%     .n_tiles, .n_runs, .window_bytes_max
%
%   Siehe auch COLLECT_CLT_ENTRIES_IH, BUILD_ENTRY_SKELETON_IH.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    % Eingabe entweder als Entries-Struct (roh, per-M) ODER als Struct mit
    % .src/.tgt-Vektoren in der Nummerierung des KONSUMENTEN. WICHTIG: Der
    % getilte Kernel liest die per-Irrep-CLT-Arrays (gefiltert + auf aktive
    % Reps umnummeriert) -- fuer ihn MUESSEN src/tgt aus dem CLT abgeleitet
    % werden (tgt via repelem(entries_per_rep), src aus clt.src_idx/srcg),
    % nicht aus den rohen per-M-Entries.
    if isfield(entries, 'src') && isfield(entries, 'tgt')
        src = entries.src(:);
        tgt = entries.tgt(:);
    else
        src = entries.src_sorted(:);
        tgt = entries.tgt_sorted(:);
    end
    n_e = numel(src);
    assert(n_e == numel(tgt), 'src/tgt Laengen inkonsistent');
    ro  = double(rep_offsets(:));
    if numel(ro) >= 2 && ro(1) == 0
        % [n_reps+1]-Form (0-basierte Offsets + Endmarke) ODER [n_reps]-Form
        % mit fuehrender 0 -- beide liefern Fensterkanten via Differenzen.
    end
    n_reps = numel(ro) - 1;
    if n_reps < double(max(tgt))
        % rep_offsets kam als [n_reps] ohne Endmarke: konservativ letzte
        % Kante extrapolieren (gleichmaessige d-Bloecke).
        n_reps = numel(ro);
        ro(end + 1) = ro(end) + (ro(end) - ro(max(end - 1, 1)));
    end

    %% 1) src-Tiles: gierig entlang der Rep-Achse, Fenster <= target_bytes.
    win = @(lo, hi) (ro(hi + 1) - ro(lo)) * double(B) * 4;   % [lo, hi] inkl., 1-basiert
    tile_lo = zeros(0, 1); tile_hi = zeros(0, 1);
    lo = 1;
    while lo <= n_reps
        hi = lo;
        % groesstmoegliches hi mit Fenster <= Budget (min. 1 Rep pro Tile)
        step = max(1, floor(n_reps / 64));
        while hi < n_reps && win(lo, min(hi + step, n_reps)) <= target_bytes
            hi = min(hi + step, n_reps);
        end
        while hi < n_reps && win(lo, hi + 1) <= target_bytes
            hi = hi + 1;
        end
        tile_lo(end + 1, 1) = lo;  tile_hi(end + 1, 1) = hi; %#ok<AGROW>
        lo = hi + 1;
    end
    n_tiles = numel(tile_lo);

    %% 2) Tile-Index pro Entry (ueber src) + stabile Sortierung nach
    %  (tile, tgt). discretize ist O(n log n_tiles); die Sortierung traegt
    %  den uint64-Schluessel tile*2^32 + tgt (beide < 2^31).
    edges = [tile_lo; n_reps + 1];
    tile_of = discretize(double(src), edges);
    key  = uint64(tile_of) * uint64(2^32) + uint64(tgt);
    [key_s, perm] = sort(key);            % stabil fuer gleiche Schluessel
    clear key tile_of;

    %% 3) Run-CSR: Grenzen dort, wo sich der Schluessel aendert.
    chg = [true; diff(double(key_s)) ~= 0];
    run_start = find(chg);                            % 1-basiert in Entry-Strom
    n_runs = numel(run_start);
    run_ptr = int64([run_start - 1; n_e]);            % 0-basiert, [n_runs+1]
    run_tgt = int32(bitand(key_s(run_start), uint64(2^32 - 1)));
    run_tile = double(bitshift(key_s(run_start), -32));
    clear key_s chg run_start;

    %% 4) Runs -> Tiles (CSR ueber die Run-Achse).
    tile_run_ptr = int64([0; cumsum(accumarray(run_tile, 1, [n_tiles, 1]))]);

    wmax = 0;
    for k = 1 : n_tiles, wmax = max(wmax, win(tile_lo(k), tile_hi(k))); end

    tl = struct('perm', perm, 'tile_lo', int32(tile_lo), 'tile_hi', int32(tile_hi), ...
                'run_ptr', run_ptr, 'run_tgt', run_tgt, 'tile_run_ptr', tile_run_ptr, ...
                'n_tiles', n_tiles, 'n_runs', n_runs, 'window_bytes_max', wmax);
end
