function test_clt_lookup()
%TEST_CLT_LOOKUP  Round-trip test for the bitmap-compressed lookup.
%
%   For several (n_total, basis) configurations:
%       1. Builds a dense int32 reference lookup the old way.
%       2. Builds the bitmap CLT via BUILD_CLT_LOOKUP.
%       3. Queries every state index 0..n_total-1 via QUERY_CLT_LOOKUP.
%       4. Verifies the bitmap result equals the dense reference for
%          every queried state (in-basis -> 1-based index match;
%          not-in-basis -> 0 match).
%       5. For a large random configuration, verifies a sampled subset
%          (avoiding the full 1 e9 sweep that exhaustive coverage would
%          require on the icosidodecahedron scale).
%
%   Also reports the memory ratio (dense / bitmap) so the 16x
%   compression is visible.
%
%   Run from mit_pg/. No arguments.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    fprintf('=== Regression: bitmap CLT lookup ===\n\n');

    overall = true;

    %% Case 1: tiny exhaustive check (boundary bits, edge of block).
    fprintf('--- Case 1: tiny exhaustive (n_total = 100) ---\n');
    n_total = 100;
    rng(11);
    basis = sort(randperm(n_total, 27)) - 1;   % 0-based, 27 in-basis states
    overall = overall && roundtrip_check(basis, n_total, []);

    %% Case 2: full bit-31 / block-boundary coverage.
    fprintf('--- Case 2: bit-31 and block boundaries (n_total = 256) ---\n');
    n_total = 256;
    basis = [0, 1, 30, 31, 32, 33, 62, 63, 64, 100, 127, 128, 254, 255];
    overall = overall && roundtrip_check(basis, n_total, []);

    %% Case 3: medium (n_total = 1e5), full sweep
    fprintf('--- Case 3: medium random basis (n_total = 1e5) ---\n');
    n_total = 1e5;
    rng(23);
    basis = sort(randperm(n_total, 7321)) - 1;
    overall = overall && roundtrip_check(basis, n_total, []);

    %% Case 4: icosahedron s=2 scale (n_total ~ 2.44e8, basis ~ 165 k)
    fprintf('--- Case 4: icosahedron s=2 scale ---\n');
    n_total = 244140625;
    rng(37);
    n_basis = 165000;
    basis = sort(int64(randperm(n_total, n_basis)) - 1);
    sample = sort([basis(:); int64(randperm(n_total, 50000) - 1)']);
    overall = overall && roundtrip_check(basis, n_total, double(sample));

    %% Case 5: icosidodecahedron s=1/2 scale (n_total = 2^30, basis ~ 1.3 M)
    fprintf('--- Case 5: icosidodecahedron s=1/2 scale ---\n');
    n_total = 2^30;
    rng(53);
    n_basis = 1.3e6;
    basis = sort(int64(randperm(n_total, n_basis)) - 1);
    sample = sort([basis(:); int64(randperm(n_total, 200000) - 1)']);
    overall = overall && roundtrip_check(basis, n_total, double(sample));

    %% Summary
    fprintf('\n=========================================\n');
    fprintf('OVERALL: %s\n', tern(overall, 'PASS', 'FAIL'));
end


% ----------------------------------------------------------------
function ok = roundtrip_check(basis, n_total, sample_states)
% Build dense lookup + CLT lookup; compare on either full sweep or
% provided sample_states; report timing + memory ratio.

    basis = double(basis(:));

    t0 = tic;
    if n_total <= 1e8
        dense = zeros(n_total, 1, 'int32');
        dense(basis + 1) = int32(1 : numel(basis));
        dense_built = true;
    else
        % Don't build the full 4 GB dense for the icosidodecahedron-scale
        % test; we'll only need values at the sample_states and can
        % compute them analytically: in-basis iff the state appears in
        % sorted basis (binary search); index = 1-based position.
        dense = [];
        dense_built = false;
    end
    t_dense = toc(t0);

    t0 = tic;
    clt = build_clt_lookup(basis, n_total);
    t_clt = toc(t0);

    %% Memory comparison
    dense_bytes = n_total * 4;
    clt_bytes   = double(clt.n_blocks) * 8;   % int32 + uint32 per block
    ratio       = dense_bytes / clt_bytes;
    fprintf('   dense  : %s   build %.3f s\n', fmt_bytes(dense_bytes), t_dense);
    fprintf('   bitmap : %s   build %.3f s   (compression %.2fx)\n', ...
            fmt_bytes(clt_bytes), t_clt, ratio);

    %% Query
    if isempty(sample_states)
        % Full sweep: query every state in [0, n_total).
        q = (0 : n_total - 1).';
    else
        q = sample_states(:);
    end

    t0 = tic;
    got = query_clt_lookup(clt, q);
    t_q  = toc(t0);

    if dense_built
        expected = dense(double(q) + 1);
    else
        expected = zeros(numel(q), 1, 'int32');
        [in_basis, pos] = ismember(double(q), basis);
        expected(in_basis) = int32(pos(in_basis));
    end

    ok = isequal(got, expected);
    n_mism = sum(got ~= expected);
    fprintf('   query  : %d states, %.3f s, %d mismatches   [%s]\n\n', ...
            numel(q), t_q, n_mism, tern(ok, 'OK', 'FAIL'));
end


function s = fmt_bytes(n)
    if n < 1024,         s = sprintf('%d B',     n);
    elseif n < 1024^2,   s = sprintf('%.1f KB',  n/1024);
    elseif n < 1024^3,   s = sprintf('%.1f MB',  n/1024^2);
    else,                s = sprintf('%.2f GB',  n/1024^3);
    end
end

function s = tern(c, a, b)
    if c, s = a; else, s = b; end
end
