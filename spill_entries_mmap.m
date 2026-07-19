function [clt, mh] = spill_entries_mmap(clt, fpath)
%SPILL_ENTRIES_MMAP  Out-of-core streaming: write the streaming entry table to a
%   binary file on disk and memory-map it, so the per-SpMV rep-tile reads page
%   from NVMe (reclaimable OS page cache) instead of sitting RESIDENT in host RAM.
%
%   [CLT, MH] = SPILL_ENTRIES_MMAP(CLT, FPATH) writes CLT's per-entry arrays
%   (packed uint32 srcg, OR unpacked int32 src + uint16 g) to FPATH, maps it
%   read-only via MMAP_FILE, stores the raw base+offset host pointers in CLT
%   (clt.mmap_srcg, or clt.mmap_src/clt.mmap_g -- uint64 scalars), and FREES the
%   in-RAM copies (clt.srcg / clt.src_idx+clt.g_idx -> empty). RUN_FTLM_PG_SECTOR_GPU_IH
%   passes those uint64 pointers to the kernel's init_skel_stream, which treats a
%   uint64 scalar entry arg as a raw mapped-file pointer (see HOST_PTR_ARG in the
%   .cu). MH is the map handle; close with mmap_file('close', MH) when the M
%   sector is done, then delete FPATH.
%
%   Only the STREAMING path is supported. Both coefficient layouts spill:
%   s=1/2 (constant c) writes [src][g] or [srcg]; s>=1 (indexed c, non-empty
%   clt.c_idx) appends the per-entry uint8 c-index as a trailing block and sets
%   clt.mmap_cidx (the small c_table stays in clt, resident on the GPU). The
%   entries must already be sorted by target rep (they are, out of collect).
%
%   See also MMAP_FILE, RUN_FTLM_PG_SECTOR_GPU_IH, BUILD_ENTRY_SKELETON_IH.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    assert(exist('mmap_file', 'file') == 3, ...
        'spill_entries_mmap: mmap_file MEX not built (mex mmap_file.cpp).');

    packed = isfield(clt, 'srcg') && ~isempty(clt.srcg);
    with_c = isfield(clt, 'c_idx') && ~isempty(clt.c_idx);
    fid = fopen(fpath, 'w');
    if fid < 0, error('spill_entries_mmap:open', 'cannot open %s for writing', fpath); end
    cu = onCleanup(@() fclose_if_open(fid));

    if packed
        srcg = gather(clt.srcg);  ne = numel(srcg);
        nw = fwrite(fid, srcg, 'uint32');
        assert(nw == ne, 'spill_entries_mmap: short write (srcg)');
        base_c = uint64(ne) * uint64(4);                  % c block after [srcg]
    else
        src = gather(clt.src_idx);  g = uint16(gather(clt.g_idx));  ne = numel(src);
        assert(numel(g) == ne, 'spill_entries_mmap: src/g length mismatch');
        nw1 = fwrite(fid, src, 'int32');
        nw2 = fwrite(fid, g,   'uint16');
        assert(nw1 == ne && nw2 == ne, 'spill_entries_mmap: short write (src/g)');
        base_c = uint64(ne) * uint64(6);                  % c block after [src][g]
    end
    if with_c
        ci  = uint8(gather(clt.c_idx));
        assert(numel(ci) == ne, 'spill_entries_mmap: c_idx length mismatch');
        nwc = fwrite(fid, ci, 'uint8');
        assert(nwc == ne, 'spill_entries_mmap: short write (c_idx)');
    end
    clear cu;                                 % flush + close the file

    [base, nb] = mmap_file('open', fpath);
    mh = base;
    nb_expect = double(base_c) + double(with_c) * ne;     % base_c == src(+g) bytes
    assert(double(nb) == nb_expect, 'spill_entries_mmap: mapped size != written bytes');
    if packed
        clt.mmap_srcg = base;                 % offset 0
        clt.srcg = zeros(0, 1, 'uint32');     % free resident
    else
        clt.mmap_src = base;                              % src at offset 0
        clt.mmap_g   = base + uint64(ne) * uint64(4);     % g starts after the int32 src block
        clt.src_idx  = zeros(0, 1, 'int32');   % free resident
        clt.g_idx    = zeros(0, 1, 'uint16');
    end
    if with_c
        clt.mmap_cidx = base + base_c;        % uint8 c-index block (s>=1)
        clt.c_idx     = zeros(0, 1, 'uint8'); % free resident (c_table stays in clt)
    end
end

function fclose_if_open(fid)
    try, if fid >= 0, fclose(fid); end, catch, end
end
