function test_min_image_fast(sample_cap)
%TEST_MIN_IMAGE_FAST  Correctness of the vectorised-digit MIN_IMAGE_IH.
%
%   TEST_MIN_IMAGE_FAST()
%   TEST_MIN_IMAGE_FAST(SAMPLE_CAP)
%
%   Verifies the power-of-two-d_loc fast digit path in MIN_IMAGE_IH against
%   an INDEPENDENT brute-force reference (apply every one of the |G|=120
%   group elements with APPLY_PERM_TO_STATE and take the element-wise min).
%   Two properties are checked on real-geometry states (super-reps plus
%   deliberately off-orbit-min states, which exercise the g_min != identity
%   path):
%     (1) reps  == brute-force orbit minimum (the basis-defining value);
%     (2) g_min is a VALID minimiser: applying group element g_min to each
%         state reproduces its reps.
%
%   The icosidodecahedron super-rep set is sub-sampled to SAMPLE_CAP
%   (default 50000) so the 120x brute force stays quick.
%
%   See also MIN_IMAGE_IH, APPLY_PERM_TO_STATE, BENCH_MIN_IMAGE_BREAKDOWN,
%            TEST_LOOKUP_SCHNACK_VS_BITMAP.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    if nargin < 1 || isempty(sample_cap), sample_cap = 50000; end
    s_val = 0.5; d_loc = 2; M = 0;
    rng(20260531);

    fprintf('=== min_image_Ih correctness (vectorised digits vs brute force) ===\n');
    geoms = {'icosahedron', 'icosidodecahedron'};
    overall = true;

    for gi = 1 : numel(geoms)
        switch geoms{gi}
            case 'icosahedron',       group = icosahedron_Ih_full();
            case 'icosidodecahedron', group = icosidodecahedron_Ih_full();
        end
        N = double(group.N);

        cache = enumerate_M_orbits_Ih_gpu(s_val, M, group);
        sr = cache.super_reps;
        if numel(sr) > sample_cap
            sr = sr(randperm(numel(sr), sample_cap));
        end

        % Off-orbit-min states: apply a fixed non-identity element. Their
        % orbit minimum is still the original super-rep -> g_min ~= identity.
        gtest = min(7, double(group.order));
        off   = apply_perm_to_state(group.perms(gtest, :), sr, d_loc, N);
        states = [sr(:); off(:)];

        [reps, g_min] = min_image_Ih(states, group, s_val);

        % (1) brute-force orbit minimum over all |G| elements.
        reps_bf = states;
        for g = 1 : double(group.order)
            img = apply_perm_to_state(group.perms(g, :), states, d_loc, N);
            reps_bf = min(reps_bf, img);
        end
        ok_reps = isequal(int64(reps(:)), int64(reps_bf(:)));

        % (2) g_min is a valid minimiser (per-state, grouped by g value).
        ok_g = true;
        ug = unique(g_min(:)).';
        for gv = ug
            m = (g_min(:) == gv);
            appl = apply_perm_to_state(group.perms(double(gv), :), states(m), d_loc, N);
            if ~isequal(int64(appl(:)), int64(reps(m)))
                ok_g = false; break;
            end
        end

        fprintf('  %-18s N=%2d  states=%7d : reps==bruteforce=%d, g_min valid=%d\n', ...
                geoms{gi}, N, numel(states), ok_reps, ok_g);
        overall = overall && ok_reps && ok_g;
    end

    fprintf('\nmin_image_fast correctness: %s\n', tern(overall, 'PASS', 'FAIL'));
    assert(overall, 'min_image_Ih correctness check FAILED');
end


function s = tern(c, a, b)
    if c, s = a; else, s = b; end
end
