function test_compact_v()
%TEST_COMPACT_V  D2 compact-V kernel path == full-V path (bit-for-bit).
%
%   For a d=4 irrep of the 4x4 C_4v space group (which has trivial AND
%   non-trivial stabiliser reps), build the skeleton CLT both with the full
%   per-rep V tensor and with the compact-V storage (shared trivial slot +
%   per-non-trivial-rep slots, indexed by skel.v_slot), run the GPU
%   block-Lanczos on each with the SAME seed, and require identical Ritz
%   values + FTLM weights. Compact-V stores the SAME float values at remapped
%   slots, so the kernel must read the same numbers -> results identical.
%
%   See also BUILD_CLT_SKELETON_FROM_ENTRIES_IH, RUN_FTLM_PG_SECTOR_GPU_IH.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    assert(gpuDeviceCount > 0, 'need a GPU');
    gpu_h = gpuDevice;
    s_val = 0.5; J = 1.0; M = 0; R = 4; M_lz = 50; B = 0; seed = 12345;

    group = square_lattice_spacegroup(4, 4);
    group.irreps = irreps_from_group(group);
    bonds = adjacency_square_lattice(4, 4);

    cache   = enumerate_M_orbits_Ih_gpu(s_val, M, group);
    entries = collect_clt_entries_Ih(cache.super_reps, bonds, s_val, J, group, 'bitmap', 'host');
    eskel   = build_entry_skeleton_Ih(entries);

    % pick the first d=4 irrep
    p = find(arrayfun(@(s) s.d, group.irreps) == 4, 1);
    ir_data = group.irreps(p).mats;  d = group.irreps(p).d;
    fprintf('Using irrep %s (d=%d)\n', group.irreps(p).name, d);

    [reps, V, eg, npr, ~, triv] = apply_irrep_to_orbits(cache, ir_data, d, group);
    n_nt = sum(~triv & (npr(:) > 0));
    fprintf('n_reps=%d, trivial active=%d, non-trivial active=%d\n', ...
        numel(reps), sum(triv & (npr(:)>0)), n_nt);

    clt_full = build_clt_skeleton_from_entries_Ih(entries, reps, V, eg, npr, ir_data, d, group, triv, eskel, false);
    [Ef, wf] = run_ftlm_pg_sector_gpu_Ih(clt_full, R, M_lz, B, seed, gpu_h);
    clt_full = []; wait(gpu_h); %#ok<NASGU>

    clt_comp = build_clt_skeleton_from_entries_Ih(entries, reps, V, eg, npr, ir_data, d, group, triv, eskel, true);
    assert(numel(clt_comp.v_slot) > 0, 'compact-V did not populate v_slot');
    n_reps_full = numel(entries.super_reps);
    n_slots     = numel(clt_comp.V_re) / (d*d);
    fprintf('full V slots=%d, compact V slots=%d (%.1fx smaller)\n', ...
        n_reps_full, n_slots, n_reps_full / n_slots);
    [Ec, wc] = run_ftlm_pg_sector_gpu_Ih(clt_comp, R, M_lz, B, seed, gpu_h);
    clt_comp = []; wait(gpu_h); %#ok<NASGU>

    dE = max(abs(Ef - Ec));
    dW = max(abs(wf - wc));
    fprintf('max|dE| = %.3e, max|dW| = %.3e\n', dE, dW);
    assert(dE < 1e-4 && dW < 1e-3, 'compact-V differs from full-V (dE=%.2e dW=%.2e)', dE, dW);
    fprintf('PASS: compact-V == full-V.\n');
end
