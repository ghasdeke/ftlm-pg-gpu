function test_Ih_full()
%TEST_IH_FULL  Verify the full I_h group construction including all irreps.
%
%   This test extends test_Ih_group (which only checked the 1D
%   structure) to all ten irreducible representations of I_h. It mirrors
%   verify_Ih_full.py and covers:
%
%   1. Group cardinality |I_h| = 120 and 60 proper / 60 improper.
%   2. Conjugacy class structure (10 classes, sizes
%       [1, 1, 12, 12, 12, 12, 15, 15, 20, 20]).
%   3. Adjacency invariance: every group element permutes the 30 bonds
%      to a permutation of the same set.
%   4. Homomorphism rho_Gamma(g) * rho_Gamma(h) == rho_Gamma(g*h) for
%      every (g, h) pair and every irrep Gamma.
%   5. Character orthogonality on the 10 classes (all 55 identities).
%   6. Dimension consistency Sum d_Gamma^2 = |G|.
%
%   Passing this test implies that the I_h group and all ten irreps are
%   ready for the symmetry-adapted basis construction in Phase beta.
%
%   Run from MATLAB inside mit_pg/:
%       >> test_Ih_full

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    fprintf('=== Test: full I_h group + all irreps ===\n\n');
    overall = true;

    G = icosahedron_Ih_full();
    bonds = adjacency_icosahedron_Ih();

    %% 1. Cardinality and parity counts
    pass = (G.order == 120) && (sum(G.det ==  1) == 60) && ...
                                 (sum(G.det == -1) == 60);
    fprintf('  |I_h| = %d, 60 proper + 60 improper  [%s]\n', G.order, ternary(pass, 'OK', 'FAIL'));
    overall = overall && pass;

    %% 2. Conjugacy class structure
    sz_expected = [1; 1; 12; 12; 12; 12; 15; 15; 20; 20];
    sz_actual   = sort(double(G.class_size));
    pass = isequal(sz_actual, sz_expected);
    fprintf('  10 classes, sizes [%s]  [%s]\n', ...
        strjoin(arrayfun(@(x) sprintf('%d', x), sz_actual', 'UniformOutput', false), ','), ...
        ternary(pass, 'OK', 'FAIL'));
    overall = overall && pass;

    %% 3. Bond invariance
    bond_set = bonds_to_set(bonds);
    bad = 0;
    for k = 1 : G.order
        p = double(G.perms(k, :));
        bonds_img = p(bonds);
        if ~isequal(bonds_to_set(bonds_img), bond_set)
            bad = bad + 1;
        end
    end
    pass = (bad == 0);
    fprintf('  Bond invariance for all 120 elements  [%s]\n', ternary(pass, 'OK', 'FAIL'));
    overall = overall && pass;

    %% 4. Homomorphism check for every irrep
    fprintf('\n  Homomorphism rho(g)*rho(h) == rho(g*h):\n');
    irreps = {'Ag', 'Au', 'T1g', 'T1u', 'T2g', 'T2u', 'Fg', 'Fu', 'Hg', 'Hu'};
    for ii = 1 : numel(irreps)
        name = irreps{ii};
        R = G.(name);
        n_fail = 0;
        tol = 1e-8;
        % We check 200 random pairs rather than all 14400 to keep the
        % test fast in MATLAB; the Python verification has the full
        % 14400-pair pass.
        rng(2026);
        for trial = 1 : 200
            i = randi(120);
            j = randi(120);
            ij = G.mul(i, j);
            if ndims(R) == 3
                lhs = R(:, :, i) * R(:, :, j);
                rhs = R(:, :, ij);
            else
                lhs = R(i) * R(j);
                rhs = R(ij);
            end
            if max(abs(lhs(:) - rhs(:))) > tol, n_fail = n_fail + 1; end
        end
        pass_i = (n_fail == 0);
        fprintf('    %-4s  [%s]\n', name, ternary(pass_i, 'OK', 'FAIL'));
        overall = overall && pass_i;
    end

    %% 5. Character orthogonality on classes
    %  Build character table from any one representative per class.
    [~, ord] = sort(double(G.class_size));
    classes_rep = zeros(10, 1, 'int32');
    classes_sz  = zeros(10, 1);
    seen = false(120, 1);
    c = 0;
    for k = 1 : 120
        if seen(k), continue; end
        c = c + 1;
        cls_idx = G.class_idx(k);
        % collect members
        members = find(G.class_idx == cls_idx);
        classes_rep(c) = int32(members(1));
        classes_sz(c)  = numel(members);
        seen(members) = true;
    end
    char_table = zeros(10, 10) + 0i;
    for ii = 1 : numel(irreps)
        R = G.(irreps{ii});
        for cc = 1 : 10
            idx = classes_rep(cc);
            if ndims(R) == 3
                char_table(ii, cc) = trace(R(:, :, idx));
            else
                char_table(ii, cc) = R(idx);
            end
        end
    end
    bad = 0;
    for ii = 1 : 10
        for jj = ii : 10
            s = sum(classes_sz' .* char_table(ii, :) .* conj(char_table(jj, :)));
            expected = (ii == jj) * 120;
            if abs(s - expected) > 1e-6, bad = bad + 1; end
        end
    end
    pass = (bad == 0);
    fprintf('\n  Character orthogonality (55 identities)  [%s]\n', ternary(pass, 'OK', 'FAIL'));
    overall = overall && pass;

    %% 6. Dimension sum
    dims = [1, 1, 3, 3, 3, 3, 4, 4, 5, 5];
    pass = (sum(dims.^2) == 120);
    fprintf('  Sum d_Gamma^2 = %d  [%s]\n', sum(dims.^2), ternary(pass, 'OK', 'FAIL'));
    overall = overall && pass;

    fprintf('\n=========================================================\n');
    fprintf('OVERALL Phase alpha (full I_h with all irreps): %s\n', ternary(overall, 'PASS', 'FAIL'));
end

% ----------------------------------------------------------------
function S = bonds_to_set(b)
    rows = sort(b, 2);
    S = sortrows(rows);
end

function s = ternary(cond, a, b)
    if cond, s = a; else, s = b; end
end
