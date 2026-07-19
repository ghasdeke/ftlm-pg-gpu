function idx = query_clt_lookup(clt, states)
%QUERY_CLT_LOOKUP  Batched query of the compressed lookup table.
%
%   IDX = QUERY_CLT_LOOKUP(CLT, STATES) returns the 1-based position of
%   each input state inside the basis array passed to BUILD_CLT_LOOKUP,
%   or 0 if the state is not in the basis. STATES is a vector of 0-based
%   state indices (int64 or convertible). IDX is int32 with the same
%   shape as STATES.
%
%   Algorithm. Each state s decomposes into a (block, bit) pair:
%       block = s / 32 + 1           (1-based)
%       bit   = s mod 32             (0..31)
%   Then:
%       presence(s)  = bit-`bit` of clt.block_mask(block) is set
%       index(s)     = clt.block_base(block) + popcount(
%                        clt.block_mask(block) AND ((1 << bit) - 1)) + 1
%   The popcount counts how many set bits sit BELOW position `bit` in
%   that block, which is exactly the offset of state s among the
%   in-basis states of the block.
%
%   The implementation is fully vectorised over STATES and uses a
%   SWAR-based byte-sum popcount that does not rely on modular uint32
%   multiplication (MATLAB uint32 multiply saturates rather than wraps).
%
%   See also BUILD_CLT_LOOKUP, COLLECT_CLT_ENTRIES_IH.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    BLOCK_SIZE = double(clt.block_size);
    assert(BLOCK_SIZE == 32, 'query_clt_lookup expects 32-state blocks');

    if isempty(states)
        idx = zeros(0, 1, 'int32');
        return;
    end

    states_d = double(states(:));               % 0-based, fits in double for <= 2^52
    blks     = floor(states_d / BLOCK_SIZE) + 1;
    bits     = mod(states_d, BLOCK_SIZE);       % 0..31, double

    %% Gather block masks / bases per query.
    masks_at = clt.block_mask(blks);            % uint32 column
    base_at  = clt.block_base(blks);            % int32 column (-1 if empty)

    %% Is the bit set?
    bit_set = bitand(bitshift(masks_at, -bits), uint32(1)) > 0;

    %% Lower-bit mask = (1 << bit) - 1. At bit=0 this is 0; at bit=31
    %  this is 0x7FFFFFFF. Computing via 2.^bits is exact in double for
    %  bits in 0..31.
    lower_mask = uint32(pow2(bits)) - uint32(1);
    masked     = bitand(masks_at, lower_mask);
    pop        = popcount_u32(masked);

    idx = zeros(numel(states_d), 1, 'int32');
    valid = bit_set & (base_at >= 0);
    idx(valid) = base_at(valid) + int32(pop(valid)) + int32(1);
end


% ----------------------------------------------------------------
function pop = popcount_u32(x)
% Vectorised population count for uint32 input.
% Standard SWAR reduction: two-bit pairs -> four-bit nibbles -> bytes,
% then sum the four bytes explicitly. (We do NOT use the classic
% multiply-by-0x01010101 trick because MATLAB's uint32 multiplication
% saturates instead of wrapping, which breaks the trick.)
    x = uint32(x);
    x = x - bitand(bitshift(x, -1), uint32(1431655765));                 % 0x55555555
    x = bitand(x, uint32(858993459)) + ...
        bitand(bitshift(x, -2), uint32(858993459));                       % 0x33333333
    x = bitand(x + bitshift(x, -4), uint32(252645135));                   % 0x0F0F0F0F

    b0 = bitand(x, uint32(255));
    b1 = bitand(bitshift(x, -8),  uint32(255));
    b2 = bitand(bitshift(x, -16), uint32(255));
    b3 = bitand(bitshift(x, -24), uint32(255));
    pop = b0 + b1 + b2 + b3;
end
