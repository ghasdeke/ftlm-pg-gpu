function clt = build_clt_lookup(basis, n_total)
%BUILD_CLT_LOOKUP  Bitmap-compressed state -> basis-index lookup table.
%
%   CLT = BUILD_CLT_LOOKUP(BASIS, N_TOTAL) builds a compressed lookup
%   table that, for any 0-based state index s in [0, N_TOTAL), answers:
%
%       (i)  whether s is in BASIS,
%       (ii) if so, its 1-based index in BASIS.
%
%   The structure is the same one used in RELEASE/FTLM_OBSERVABLES.M
%   (Lin-style block bitmap) and reduces memory from N_TOTAL * 4 bytes
%   (dense int32 lookup) to N_TOTAL / 32 * 8 bytes -> exact 16x
%   compression. For s = 1/2 on the icosidodecahedron (N_TOTAL = 2^30)
%   this is 256 MB instead of 4 GB; for s = 2 on the icosahedron
%   (N_TOTAL = 5^12 ~ 2.44e8) it is 60 MB instead of ~ 1 GB.
%
%   The basis array must be sorted ascending and contain only valid
%   0-based state indices in [0, N_TOTAL). Duplicates are not allowed.
%
%   Output CLT struct fields:
%       block_size    int32 = 32 (the block granularity)
%       n_total       double; the N_TOTAL passed in (echoed for sanity)
%       n_blocks      int32 = ceil(N_TOTAL / 32)
%       block_base    [n_blocks x 1] int32; for block b (1-based),
%                     block_base(b) is the 0-based index in BASIS of the
%                     first state in that block, or -1 if no states fall
%                     in block b.
%       block_mask    [n_blocks x 1] uint32; bit j (j = 0..31) is 1 iff
%                     state 32 * (b - 1) + j is in BASIS.
%
%   Queries: use QUERY_CLT_LOOKUP.
%
%   Limits. The block_base / block_mask arrays scale with N_TOTAL / 32.
%   For s >= 1 on the icosidodecahedron, N_TOTAL = 3^30 ~ 2e14 -> the
%   bitmap itself becomes infeasible (~ 50 TB). Such systems require a
%   different lookup strategy (binary search over BASIS, or a true hash
%   table). For the s = 1/2 / icosidodecahedron target this routine is
%   the right tool.
%
%   See also QUERY_CLT_LOOKUP, COLLECT_CLT_ENTRIES_IH.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    BLOCK_SIZE = 32;

    n_total  = double(n_total);
    n_blocks = ceil(n_total / BLOCK_SIZE);

    if isempty(basis)
        clt.block_size = int32(BLOCK_SIZE);
        clt.n_total    = n_total;
        clt.n_blocks   = int32(n_blocks);
        clt.block_base = int32(-ones(n_blocks, 1));
        clt.block_mask = zeros(n_blocks, 1, 'uint32');
        return;
    end

    states = double(basis(:));
    assert(states(end) < n_total, ...
        'build_clt_lookup: state %g out of range (n_total = %g)', states(end), n_total);

    blks = floor(states / BLOCK_SIZE) + 1;        % 1-based block index
    bits = mod(states, BLOCK_SIZE);               % 0..31

    %% block_base(b) = first basis-array position landing in block b (0-based)
    block_base = int32(-ones(n_blocks, 1));
    [ub, fi]   = unique(blks, 'first');
    block_base(ub) = int32(fi - 1);

    %% block_mask(b) = sum of 2^bit_j over basis states in block b
    %  Use 'pow2(bits)' (exact double up to bit <= 52, more than enough
    %  for 32-bit bitmasks). accumarray sums into the right bin per block.
    bit_vals  = pow2(bits);
    mask_sums = accumarray(blks, bit_vals, [n_blocks, 1]);
    block_mask = uint32(mask_sums);

    %% Pack
    clt.block_size = int32(BLOCK_SIZE);
    clt.n_total    = n_total;
    clt.n_blocks   = int32(n_blocks);
    clt.block_base = block_base;
    clt.block_mask = block_mask;
end
