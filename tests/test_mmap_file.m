function test_mmap_file()
%TEST_MMAP_FILE  Verify the read-only memory-mapping MEX round-trips bit-identically,
%   including mid-file tile slices (the access pattern the streaming SpMV uses).
%
%   See also MMAP_FILE.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    assert(exist('mmap_file', 'file') == 3, 'mmap_file MEX not built (run: mex mmap_file.cpp).');

    n   = 100000;
    rng(1);
    src = int32(randi(2^30, n, 1));
    g   = uint16(randi(65535, n, 1));
    cidx = uint8(randi(200, n, 1));

    f = [tempname, '.bin'];
    fid = fopen(f, 'w');
    fwrite(fid, src,  'int32');     % bytes [0,            4n)
    fwrite(fid, g,    'uint16');    % bytes [4n,           6n)
    fwrite(fid, cidx, 'uint8');     % bytes [6n,           7n)
    fclose(fid);

    [ptr, nbytes] = mmap_file('open', f);
    cleanup = onCleanup(@() cleanupFcn(ptr, f));
    assert(double(nbytes) == 4*n + 2*n + 1*n, 'mapped size wrong: %d', nbytes);

    % full read-back
    src_rb  = mmap_file('read', ptr, 0,       n, 'int32');
    g_rb    = mmap_file('read', ptr, 4*n,     n, 'uint16');
    cidx_rb = mmap_file('read', ptr, 4*n+2*n, n, 'uint8');
    assert(isequal(src,  src_rb),  'src round-trip mismatch');
    assert(isequal(g,    g_rb),    'g round-trip mismatch');
    assert(isequal(cidx, cidx_rb), 'cidx round-trip mismatch');

    % mid-file tile slice (the streaming access pattern): entries [lo, hi)
    lo = 31234; cnt = 5000;
    tile = mmap_file('read', ptr, 4*lo, cnt, 'int32');   % int32 -> 4 bytes/elem
    assert(isequal(tile, src(lo+1 : lo+cnt)), 'tile slice mismatch');

    fprintf('test_mmap_file PASS (mapped %d bytes, slices bit-identical)\n', nbytes);
end

function cleanupFcn(ptr, f)
    try, mmap_file('close', ptr); catch, end
    if exist(f, 'file'), delete(f); end
end
