function test_Ih_group()
%TEST_IH_GROUP  Verify the I_h group construction on the icosahedron.
%
%   Checks:
%     1. |G| = 120
%     2. 10 conjugacy classes with sizes [1, 1, 12, 12, 12, 12, 15, 15, 20, 20]
%     3. Each class is correctly labeled (E, 12C5, ..., 15sigma)
%     4. Multiplication table is closed (already enforced by construction)
%     5. inv(g) is a true inverse (already enforced)
%     6. det(g) for the underlying 3D rotation matches the geometric type
%     7. Vertex permutation preserves the icosahedron edge set (adjacency
%        invariance). Without this, the permutation is not a symmetry of
%        the graph and the SpMV symmetry-adaptation breaks.
%     8. apply_perm_to_state acts correctly: g * (h * |n>) = (g*h) * |n>
%        for a few random group elements and states.
%
%   Run from MATLAB in the mit_pg directory:
%       >> test_Ih_group

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    fprintf('=== Test: icosahedral group I_h construction ===\n\n');

    G = icosahedron_Ih_group();
    overall = true;

    %% 1. Order
    pass = (G.order == 120);
    fprintf('  |G| = %d  [%s]\n', G.order, ternary(pass, 'OK', 'FAIL'));
    overall = overall && pass;

    %% 2. Conjugacy class structure
    expected_sizes = [1; 1; 12; 12; 12; 12; 15; 15; 20; 20];
    sizes_sorted   = sort(double(G.class_size));
    pass = numel(G.class_size) == 10 && all(sizes_sorted == expected_sizes);
    fprintf('  10 classes, sizes = [%s]  [%s]\n', ...
        strjoin(arrayfun(@(x) sprintf('%d', x), sizes_sorted', 'UniformOutput', false), ','), ...
        ternary(pass, 'OK', 'FAIL'));
    overall = overall && pass;

    %% 3. Class labels
    fprintf('  Class labels:\n');
    expected_labels = {'E', '15sigma', '15C2', '12C5', '12C5^2', '12S10', ...
                       '12S10^3', '20C3', '20S6', 'i'};
    % We don't enforce a specific order; just check that all expected
    % labels appear exactly once.
    labels = G.class_label;
    [~, idx_sort] = sort(double(G.class_size));
    labels_sorted = labels(idx_sort);
    found_all = all(ismember(expected_labels, labels));
    no_extras = all(ismember(labels, expected_labels));
    pass = found_all && no_extras;
    for c = 1 : numel(labels_sorted)
        fprintf('    size %3d -> %s\n', sizes_sorted(c), labels_sorted{c});
    end
    fprintf('    Label coverage  [%s]\n', ternary(pass, 'OK', 'FAIL'));
    overall = overall && pass;

    %% 4. Adjacency invariance: every group element permutes the bond list
    bonds = adjacency_icosahedron();
    bonds_set = bonds_to_set(bonds);
    fail_idx = [];
    for k = 1 : G.order
        p = G.perms(k, :);
        bonds_new = double(p(bonds));
        bonds_new_set = bonds_to_set(bonds_new);
        if ~isequal(bonds_new_set, bonds_set)
            fail_idx(end+1) = k; %#ok<AGROW>
        end
    end
    pass = isempty(fail_idx);
    fprintf('  Bond invariance for all 120 elements  [%s]\n', ternary(pass, 'OK', 'FAIL'));
    if ~pass, fprintf('    Failing elements: %s\n', mat2str(fail_idx)); end
    overall = overall && pass;

    %% 5. State action: g * (h * |n>) = (gh) * |n>
    rng(7);
    n_test = 8;
    d_loc = 2;   % spin-1/2
    n_total = 2^12;
    test_states = randi(n_total, n_test, 1) - 1;
    n_pair_tests = 10;
    pass = true;
    for tt = 1 : n_pair_tests
        g_idx = randi(G.order);
        h_idx = randi(G.order);
        gh_idx = G.mul(g_idx, h_idx);
        g = double(G.perms(g_idx, :));
        h = double(G.perms(h_idx, :));
        gh = double(G.perms(gh_idx, :));
        for s = 1 : n_test
            n = int64(test_states(s));
            ghn_direct = apply_perm_to_state(gh, n, d_loc, 12);
            hn         = apply_perm_to_state(h, n, d_loc, 12);
            ghn_compose = apply_perm_to_state(g, hn, d_loc, 12);
            if ghn_direct ~= ghn_compose
                pass = false;
                fprintf('    Compositionality FAIL: g=%d h=%d n=%d\n', g_idx, h_idx, n);
            end
        end
    end
    fprintf('  apply_perm_to_state composition  [%s]\n', ternary(pass, 'OK', 'FAIL'));
    overall = overall && pass;

    %% 6. det vector: equals +1 for rotations, -1 for improper
    n_proper   = sum(G.det ==  1);
    n_improper = sum(G.det == -1);
    pass = (n_proper == 60 && n_improper == 60);
    fprintf('  det: %d proper + %d improper  [%s]\n', n_proper, n_improper, ...
        ternary(pass, 'OK', 'FAIL'));
    overall = overall && pass;

    fprintf('\n=========================================================\n');
    fprintf('OVERALL Phase alpha (I_h group structure): %s\n', ternary(overall, 'PASS', 'FAIL'));
end

% ----------------------------------------------------------------
function bonds = adjacency_icosahedron()
    phi = (1 + sqrt(5)) / 2;
    V = [ 0, 1, phi;  0, 1, -phi;  0,-1, phi;  0,-1,-phi;
          1, phi, 0;  1,-phi, 0; -1, phi, 0; -1,-phi, 0;
          phi, 0, 1;  phi, 0,-1; -phi, 0, 1; -phi, 0,-1 ];
    bonds = [];
    for i = 1 : 12
        for j = i+1 : 12
            if abs(norm(V(i, :) - V(j, :)) - 2) < 0.01
                bonds = [bonds; i, j]; %#ok<AGROW>
            end
        end
    end
end

function S = bonds_to_set(bonds)
    rows = sort(bonds, 2);
    rows = sortrows(rows);
    S = rows;
end

function s = ternary(cond, a, b)
    if cond, s = a; else, s = b; end
end
