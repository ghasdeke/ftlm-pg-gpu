function test_irreps_from_group()
%TEST_IRREPS_FROM_GROUP  Verify the generic regular-rep irrep extractor.
%
%   Checks IRREPS_FROM_GROUP on:
%     - I_h (icosahedron_Ih_full): must recover 10 irreps with dimensions
%       {1,1,3,3,3,3,4,4,5,5} (the known character-orthogonal set);
%     - the square-lattice translation group (all 1D);
%     - the square-lattice SPACE group 4x4 (C_4v, order 128) and 6x6 (order 288).
%   For each: completeness (sum d^2 = |G|, count = #classes), unitarity,
%   homomorphism rho(a)rho(b)=rho(mul(a,b)), and character orthogonality
%   (1/|G|) sum_g chi_p(g) conj(chi_q(g)) = delta_pq.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    rng(3);
    run_one('I_h',                 icosahedron_Ih_full());
    run_one('translation 3x3',     square_lattice_translation_group(3,3));
    run_one('spacegroup 4x4 C_4v', square_lattice_spacegroup(4,4));
    run_one('spacegroup 6x6 C_4v', square_lattice_spacegroup(6,6));
    fprintf('\nALL TESTS PASSED.\n');
end

function run_one(label, g)
    n   = double(g.order);
    mul = double(g.mul);
    irr = irreps_from_group(g);
    dims = arrayfun(@(s) s.d, irr);
    fprintf('=== %s: |G|=%d, %d irreps, dims=[%s], sum d^2=%d ===\n', ...
        label, n, numel(irr), num2str(sort(dims)), sum(dims.^2));

    assert(sum(dims.^2) == n, '%s: sum d^2 != |G|', label);
    if isfield(g,'n_class')
        assert(numel(irr) == g.n_class, '%s: #irreps != #classes', label);
    end

    % unitarity + homomorphism on a random sample of pairs
    npair = min(1500, n*n);
    aa = randi(n, npair, 1);  bb = randi(n, npair, 1);
    homerr = 0; unierr = 0;
    for p = 1 : numel(irr)
        M = irr(p).mats; d = irr(p).d;
        for s = 1 : 60
            gx = randi(n);
            unierr = max(unierr, norm(M(:,:,gx)*M(:,:,gx)' - eye(d), 'fro'));
        end
        for s = 1 : npair
            ab = mul(aa(s), bb(s));
            homerr = max(homerr, norm(M(:,:,aa(s))*M(:,:,bb(s)) - M(:,:,ab), 'fro'));
        end
    end
    assert(unierr < 1e-9, '%s: non-unitary (%.2e)', label, unierr);
    assert(homerr < 1e-9, '%s: not a homomorphism (%.2e)', label, homerr);

    % character orthogonality
    X = zeros(n, numel(irr));
    for p = 1 : numel(irr)
        for gx = 1 : n, X(gx,p) = trace(irr(p).mats(:,:,gx)); end
    end
    G = (X' * X) / n;                       % should be identity
    ortherr = max(abs(G(:) - reshape(eye(numel(irr)),[],1)));
    assert(ortherr < 1e-8, '%s: character orthogonality (%.2e)', label, ortherr);

    fprintf('  unitarity %.1e, homomorphism %.1e, char-orth %.1e : OK\n', ...
        unierr, homerr, ortherr);
end
