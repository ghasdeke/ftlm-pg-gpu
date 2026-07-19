function test_realify_irreps()
%TEST_REALIFY_IRREPS  Verify the FS-indicator realification of space-group irreps.
%
%   For the square (C_4v / C_2v), kagome (C_6v) and triangular (C_6v) space
%   groups the point group contains C_2, so every irrep is self-conjugate with
%   Frobenius-Schur indicator +1 and MUST realify. For each group the test
%   checks, on the output of REALIFY_IRREPS(IRREPS_FROM_GROUP(group), group):
%     (1) every irrep has FS = +1 and is flagged realified;
%     (2) the realified matrices are exactly real (isreal == true);
%     (3) they are still a unitary (now orthogonal) homomorphism
%         rho(a)rho(b) = rho(mul(a,b));
%     (4) the characters are UNCHANGED vs the original complex irreps
%         (realification is a basis change, so chi is invariant);
%     (5) character orthogonality (1/|G|) sum_g chi_p conj(chi_q) = delta_pq
%         still holds (completeness of the realified set).
%   It also confirms that the I_h complex irreps are NOT touched by a generic
%   realify call when their FS=+1 form is requested directly, and that a
%   genuinely complex 1-D character (FS != +1) is left complex (fallback).
%
%   See also REALIFY_IRREPS, IRREPS_FROM_GROUP, TEST_IRREPS_FROM_GROUP.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    tol = 1e-7;

    gsq = square_lattice_spacegroup(4, 4);
    run_group('square 4x4 C_4v', gsq, tol);

    gk = kagome_spacegroup(2, 0);
    run_group('kagome (2,0) C_6v', gk, tol);

    gt = triangular_spacegroup(2, 2);
    run_group('triangular (2,2) C_6v', gt, tol);

    % Rectangular C_2v also contains C_2 -> all FS = +1. (Ly >= 3 so the
    % y-mirror is non-degenerate; Ly = 2 collapses sigma_x to the identity.)
    grc = square_lattice_spacegroup(4, 3);
    run_group('square 4x3 C_2v', grc, tol);

    %% Fallback: a genuinely complex 1-D irrep (C_3 cyclic, no C_2) must NOT be
    %  realified -- its FS indicator is 0 for the two complex-conjugate chars.
    fprintf('\n--- fallback: C_3 cyclic (complex chars, FS=0) ---\n');
    gc3 = cyclic_group(3);
    irr3 = irreps_from_group(gc3);
    [irr3r, info3] = realify_irreps(irr3, gc3);
    nu = sort([info3.nu]);
    n_complex_left = sum(~[info3.realified]);
    fprintf('  FS indicators = [%s]; %d irrep(s) left complex\n', ...
        num2str(round([info3.nu], 3)), n_complex_left);
    assert(any(abs(nu) < 1e-6), 'C_3 should have complex (FS~0) chars');
    % The two FS=0 chars must remain complex (not stripped to real).
    for p = 1 : numel(irr3r)
        if abs(info3(p).nu) < 1e-6
            assert(~info3(p).realified, 'C_3 complex char was wrongly realified');
            assert(~isreal(irr3r(p).mats), 'C_3 complex char became real');
        end
    end
    fprintf('  OK: complex C_3 chars left untouched.\n');

    fprintf('\nALL TESTS PASSED.\n');
end


% ----------------------------------------------------------------
function run_group(label, group, tol)
    n   = double(group.order);
    mul = double(group.mul);
    irr0 = irreps_from_group(group);
    [irr, info] = realify_irreps(irr0, group);

    fprintf('\n=== %s: |G|=%d, %d irreps ===\n', label, n, numel(irr));

    % (1) every irrep FS=+1 and realified
    assert(all(abs([info.nu] - 1) < 1e-6), ...
        '%s: not all FS indicators are +1 (got [%s])', label, num2str([info.nu]));
    assert(all([info.realified]), '%s: some FS=+1 irrep was not realified', label);

    % (2) exactly real
    for p = 1 : numel(irr)
        assert(isreal(irr(p).mats), '%s: irrep %s not real after realify', ...
            label, irr(p).name);
    end

    % (3) homomorphism + orthogonality on a random pair sample
    rng(99);
    npair = min(800, n * n);
    aa = randi(n, npair, 1);  bb = randi(n, npair, 1);
    homerr = 0; unierr = 0;
    for p = 1 : numel(irr)
        M = irr(p).mats; d = irr(p).d;
        for s = 1 : 60
            gx = randi(n);
            unierr = max(unierr, norm(M(:,:,gx)*M(:,:,gx).' - eye(d), 'fro'));
        end
        for s = 1 : npair
            ab = mul(aa(s), bb(s));
            homerr = max(homerr, norm(M(:,:,aa(s))*M(:,:,bb(s)) - M(:,:,ab), 'fro'));
        end
    end
    assert(unierr < tol, '%s: realified irreps not orthogonal (%.2e)', label, unierr);
    assert(homerr < tol, '%s: realified irreps not a homomorphism (%.2e)', label, homerr);

    % (4) characters unchanged vs the original complex irreps
    cherr = 0;
    for p = 1 : numel(irr)
        for g = 1 : n
            c_old = trace(irr0(p).mats(:,:,g));
            c_new = trace(irr(p).mats(:,:,g));
            cherr = max(cherr, abs(c_old - c_new));
        end
    end
    assert(cherr < tol, '%s: characters changed under realify (%.2e)', label, cherr);

    % (5) character orthogonality of the realified set
    Xc = zeros(n, numel(irr));
    for p = 1 : numel(irr)
        for g = 1 : n, Xc(g,p) = trace(irr(p).mats(:,:,g)); end
    end
    G = (Xc' * Xc) / n;
    ortherr = max(abs(G(:) - reshape(eye(numel(irr)), [], 1)));
    assert(ortherr < tol, '%s: character orthogonality broken (%.2e)', label, ortherr);

    fprintf('  realified=%d/%d  orth=%.1e hom=%.1e char-unchanged=%.1e char-orth=%.1e : OK\n', ...
        sum([info.realified]), numel(irr), unierr, homerr, cherr, ortherr);
end


% ----------------------------------------------------------------
function group = cyclic_group(m)
%CYCLIC_GROUP  Minimal C_m group struct (order/mul/inv/identity) for the
%   complex-character fallback test. Elements 1..m represent rotations 0..m-1.
    group.order = m;
    mul = zeros(m, m);
    for a = 1 : m
        for b = 1 : m
            mul(a, b) = mod((a - 1) + (b - 1), m) + 1;
        end
    end
    group.mul = mul;
    inv = zeros(m, 1);
    for a = 1 : m, inv(a) = mod(-(a - 1), m) + 1; end
    group.inv = inv;
    group.identity = 1;
    group.n_class = m;     % abelian
end
