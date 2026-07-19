function test_enumerate_M_orbits_Ih_gpu()
%TEST_ENUMERATE_M_ORBITS_IH_GPU  Regression test: GPU vs CPU enumerator.
%
%   Compares ENUMERATE_M_ORBITS_IH_GPU against the reference
%   ENUMERATE_M_ORBITS_IH on a few (s, M) pairs. Checks that
%       (a) super_reps are identical
%       (b) orbit_lens are identical
%       (c) stab_lists are elementwise identical
%   and reports wall times so the speedup is visible.
%
%   Designed to stay fast: s = 1/2 (M = 0, 1) and s = 1 (M = 0). The
%   user can extend with s = 2 M = 0 manually for a full benchmark, but
%   on a fresh MATLAB session that single case is what eats minutes in
%   the production pipeline.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    fprintf('=== Regression: enumerate_M_orbits_Ih_gpu vs CPU ===\n\n');

    group = icosahedron_Ih_full();

    cases = {
        struct('s', 0.5, 'M', 0)
        struct('s', 0.5, 'M', 1)
        struct('s', 1.0, 'M', 0)
    };

    overall = true;
    for k = 1 : numel(cases)
        c = cases{k};
        fprintf('--- s = %g, M = %d ---\n', c.s, c.M);

        t0 = tic;
        ref = enumerate_M_orbits_Ih(c.s, c.M, group);
        t_cpu = toc(t0);

        t0 = tic;
        new = enumerate_M_orbits_Ih_gpu(c.s, c.M, group);
        t_gpu = toc(t0);

        ok_reps = isequal(ref.super_reps, new.super_reps);
        ok_ol   = isequal(ref.orbit_lens, new.orbit_lens);
        ok_stab = same_stab_csr(ref, new);

        ok = ok_reps && ok_ol && ok_stab;
        overall = overall && ok;

        fprintf('  n_reps       : %d\n', numel(ref.super_reps));
        fprintf('  super_reps   : %s\n', tern(ok_reps, 'OK', 'FAIL'));
        fprintf('  orbit_lens   : %s\n', tern(ok_ol,   'OK', 'FAIL'));
        fprintf('  stab_lists   : %s\n', tern(ok_stab, 'OK', 'FAIL'));
        fprintf('  CPU time     : %.3f s\n', t_cpu);
        fprintf('  GPU/new time : %.3f s   (speedup x %.2f)\n\n', ...
                t_gpu, t_cpu / max(t_gpu, 1e-9));
    end

    fprintf('=====================================\n');
    fprintf('OVERALL: %s\n', tern(overall, 'PASS', 'FAIL'));
end


% ----------------------------------------------------------------
function ok = same_stab_csr(a, b)
% Compare the CSR stabilisers (stab_flat + stab_ptr) of two caches,
% order-insensitive within each rep's block.
    ok = isequal(a.stab_ptr, b.stab_ptr);
    if ~ok, return; end
    n = numel(a.stab_ptr) - 1;
    for i = 1 : n
        ai = sort(double(a.stab_flat(a.stab_ptr(i):a.stab_ptr(i+1)-1)));
        bi = sort(double(b.stab_flat(b.stab_ptr(i):b.stab_ptr(i+1)-1)));
        if ~isequal(ai(:), bi(:))
            ok = false; return;
        end
    end
end


function s = tern(c, a, b)
    if c, s = a; else, s = b; end
end
