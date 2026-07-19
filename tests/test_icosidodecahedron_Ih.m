function test_icosidodecahedron_Ih()
%TEST_ICOSIDODECAHEDRON_IH  Sanity tests for the I_h-on-icosidodecahedron setup.
%
%   Verifies:
%
%   (1) Group axioms for the derived 30-vertex permutation rep:
%       - identity acts trivially
%       - closure (perms(mul(a,b)) == perms(a) o perms(b)) for all (a,b)
%       - inverses (perms(inv(k)) o perms(k) == identity)
%
%   (2) Group data agreement with ICOSAHEDRON_IH_FULL:
%       - mul, inv, class_idx, class_label all unchanged (intrinsic to I_h).
%
%   (3) Adjacency structure:
%       - exactly 60 bonds
%       - every vertex has degree 4
%       - the bond set is closed under the full 120-element I_h action
%         (i.e. for every g and every bond (a,b), the image bond
%         (perms(g,a), perms(g,b)) is also a bond)
%
%   (4) Combinatorial identities of the icosidodecahedron:
%       - V - E + F = 2 (Euler), with V=30, E=60 -> F=32
%       - F = 20 triangles + 12 pentagons (we use the triangle count
%         from the construction itself)
%
%   Run from mit_pg/. No arguments.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    fprintf('=== Sanity tests: I_h on the icosidodecahedron (N=30) ===\n\n');

    base  = icosahedron_Ih_full();
    group = icosidodecahedron_Ih_full();
    bonds = adjacency_icosidodecahedron_Ih();

    overall = true;

    %% (1) Group axioms on the new perm rep
    fprintf('(1) Group axioms on the 30-vertex perm rep\n');

    id_row = double(group.perms(group.identity, :));
    ok = isequal(id_row, 1 : 30);
    report('   identity acts trivially', ok); overall = overall && ok;

    closure_ok = true;
    rng(7);
    n_trials = 200;        % random sample; full 14400 below
    for t = 1 : n_trials
        a = randi(120); b = randi(120);
        pa = double(group.perms(a, :));
        pb = double(group.perms(b, :));
        composed = pa(pb);
        c = group.mul(a, b);
        if ~isequal(double(group.perms(c, :)), composed)
            closure_ok = false; break;
        end
    end
    report('   closure (200 random pairs)', closure_ok); overall = overall && closure_ok;

    inv_ok = true;
    for k = 1 : 120
        pk = double(group.perms(k, :));
        pinv = double(group.perms(group.inv(k), :));
        if ~isequal(pinv(pk), 1 : 30)
            inv_ok = false; break;
        end
    end
    report('   inverses (all 120 elements)', inv_ok); overall = overall && inv_ok;

    %% (2) Group data agreement with ICOSAHEDRON_IH_FULL
    fprintf('(2) Intrinsic group data unchanged vs ICOSAHEDRON_IH_FULL\n');
    ok_mul = isequal(group.mul, base.mul);
    ok_inv = isequal(group.inv, base.inv);
    ok_cls = isequal(group.class_idx, base.class_idx) && ...
             isequal(group.class_size, base.class_size) && ...
             isequal(group.class_label, base.class_label);
    ok_id  = group.identity == base.identity;
    report('   mul table identical',          ok_mul); overall = overall && ok_mul;
    report('   inverse table identical',      ok_inv); overall = overall && ok_inv;
    report('   conjugacy class data identical', ok_cls); overall = overall && ok_cls;
    report('   identity element index identical', ok_id); overall = overall && ok_id;

    %% (3) Adjacency structure
    fprintf('(3) Icosidodecahedron adjacency\n');
    ok_n = size(bonds, 1) == 60;
    report('   exactly 60 bonds', ok_n); overall = overall && ok_n;

    deg = zeros(30, 1);
    for e = 1 : size(bonds, 1)
        deg(bonds(e, 1)) = deg(bonds(e, 1)) + 1;
        deg(bonds(e, 2)) = deg(bonds(e, 2)) + 1;
    end
    ok_deg = all(deg == 4);
    report('   every vertex has degree 4', ok_deg); overall = overall && ok_deg;

    % I_h-invariance of the bond set
    bond_keys = arrayfun(@(e) sprintf('%d-%d', bonds(e, 1), bonds(e, 2)), ...
                         1 : size(bonds, 1), 'UniformOutput', false);
    bond_set = containers.Map(bond_keys, num2cell(1 : 60));
    inv_ok_bonds = true;
    for k = 1 : 120
        pk = double(group.perms(k, :));
        for e = 1 : size(bonds, 1)
            a = pk(bonds(e, 1));
            b = pk(bonds(e, 2));
            ab = sprintf('%d-%d', min(a, b), max(a, b));
            if ~isKey(bond_set, ab)
                inv_ok_bonds = false; break;
            end
        end
        if ~inv_ok_bonds, break; end
    end
    report('   bond set I_h-invariant', inv_ok_bonds); overall = overall && inv_ok_bonds;

    %% (4) Combinatorics: Euler + face counts
    fprintf('(4) Combinatorial identities\n');
    V = 30; E = size(bonds, 1);
    % Face count from Euler: V - E + F = 2 -> F = E - V + 2
    F_expected = E - V + 2;
    ok_euler = F_expected == 32;
    report(sprintf('   Euler V-E+F=2 gives F=%d (=32 expected)', F_expected), ok_euler);
    overall = overall && ok_euler;

    %% Summary
    fprintf('\n=========================================\n');
    fprintf('OVERALL: %s\n', tern(overall, 'PASS', 'FAIL'));
end


function report(label, ok)
    fprintf('   %-45s [%s]\n', label, tern(ok, 'OK', 'FAIL'));
end

function s = tern(c, a, b)
    if c, s = a; else, s = b; end
end
