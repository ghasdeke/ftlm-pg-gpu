function test_pipeline_opts()
%TEST_PIPELINE_OPTS  One-shot regression for the 2026-06-02 optimisation
%   campaign on the I_h FTLM pipeline. Runs every correctness gate and prints
%   a PASS/FAIL summary. Use this after touching the enumerate / collect /
%   skeleton / CUDA-kernel code to confirm nothing regressed.
%
%   Covers:
%     enumerate  R1 (GPU min_image), R2 (early-reject super-rep test),
%                R3 (D_cum direct M-sector unranking), R5 (D_cum digit)
%                -> test_enumerate_M_orbits_Ih_gpu (vs the CPU scan reference)
%     collect    R4 (GPU min_image rep+g_min)
%                -> test_lookup_schnack_vs_bitmap (entries vs independent bitmap)
%     skeleton   A1 (full-rep index), D2 (bulk-fill V), A1ph2 (shared eskel),
%     + kernel   G1 (uint8 g_idx), G2 (src|g pack), c-index
%                -> test_c_index (all 10 irreps, Lanczos exact),
%                   test_entries_storage_gpu_vs_host (host vs gpu skeleton),
%                   check_eskel_consistency (shared eskel == inline eskel)
%
%   See [[n36-memory-scaling]] for the per-lever detail.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    fprintf('===== test_pipeline_opts : 2026-06-02 optimisation campaign =====\n\n');

    gates = { ...
        'enumerate R1/R2/R3/R5  (vs CPU scan reference)',   @test_enumerate_M_orbits_Ih_gpu; ...
        'collect   R4           (schnack vs bitmap entries)', @test_lookup_schnack_vs_bitmap; ...
        'skeleton+kernel        (c-index/G1/G2/A1/D2 Lanczos)', @test_c_index; ...
        'A1 host-vs-gpu entries  (both skeleton paths)',     @test_entries_storage_gpu_vs_host; ...
        'A1ph2 shared==inline eskel (E+w bit-identical)',    @check_eskel_consistency };

    n = size(gates, 1);
    stats = cell(n, 1);
    for i = 1 : n
        fprintf('--- %s ---\n', gates{i, 1});
        try
            gates{i, 2}();
            stats{i} = 'PASS';
        catch e
            stats{i} = ['FAIL: ' e.message];
        end
        fprintf('\n');
    end

    fprintf('===== SUMMARY =====\n');
    allok = true;
    for i = 1 : n
        ok = strcmp(stats{i}, 'PASS');
        fprintf('  [%-4s] %s\n', tf(ok), gates{i, 1});
        if ~ok
            fprintf('          %s\n', stats{i});
            allok = false;
        end
    end
    fprintf('\nOVERALL: %s\n', tf(allok));
    assert(allok, 'test_pipeline_opts: one or more gates FAILED');
end


% ----------------------------------------------------------------
function check_eskel_consistency()
%CHECK_ESKEL_CONSISTENCY  A1ph2 + G2 gate: the shared-eskel (build-once) and
%   inline-eskel skeleton paths must give byte-identical Lanczos E/w. Quick
%   s=3/2 check over a 1-D, 3-D and 5-D irrep.
    if gpuDeviceCount == 0
        fprintf('   (no CUDA device -- skipped)\n'); return;
    end
    s_val = 1.5; J = 1.0; M = 0; R = 2; M_lz = 20; seed = 4242;
    group = icosahedron_Ih_full();
    bonds = adjacency_icosahedron_Ih();
    gpu_h = gpuDevice;
    cache   = enumerate_M_orbits_Ih_gpu(s_val, M, group);
    entries = collect_clt_entries_Ih(cache.super_reps, bonds, s_val, J, group, 'schnack', 'host');
    eskel   = build_entry_skeleton_Ih(entries);
    fprintf('   eskel.is_packed = %d\n', eskel.is_packed);
    irr = {'Ag', 1; 'T1g', 3; 'Hg', 5};
    ok = true;
    for i = 1 : size(irr, 1)
        d = irr{i, 2}; data = group.(irr{i, 1});
        [reps, V, eg, npr, ~, triv] = apply_irrep_to_orbits(cache, data, d, group);
        clt_s = build_clt_skeleton_from_entries_Ih(entries, reps, V, eg, npr, data, d, group, triv, eskel);
        [Es, ws] = run_ftlm_pg_sector_gpu_Ih(clt_s, R, M_lz, 0, seed, gpu_h);
        clt_s = []; wait(gpu_h); %#ok<NASGU>
        clt_i = build_clt_skeleton_from_entries_Ih(entries, reps, V, eg, npr, data, d, group, triv);
        [Ei, wi] = run_ftlm_pg_sector_gpu_Ih(clt_i, R, M_lz, 0, seed, gpu_h);
        clt_i = []; wait(gpu_h); %#ok<NASGU>
        ex = isequal(Es, Ei) && isequal(ws, wi);
        fprintf('   %-4s shared==inline: %d\n', irr{i, 1}, ex);
        ok = ok && ex;
    end
    assert(ok, 'shared-eskel and inline-eskel skeletons give different E/w');
end


% ----------------------------------------------------------------
function s = tf(b)
    if b, s = 'PASS'; else, s = 'FAIL'; end
end
