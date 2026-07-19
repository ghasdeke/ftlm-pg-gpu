function test_stream_mmap()
%TEST_STREAM_MMAP  Out-of-core (memory-mapped) entry streaming == in-RAM
%   streaming, bit-identical. The entry table is spilled to a disk file and
%   memory-mapped; the SpMV reads its rep-tiles from the mapped pages (NVMe)
%   instead of a resident host array. This changes only WHERE the bytes live,
%   not their values, so the Lanczos E/w must match the in-RAM streaming path
%   exactly (dE=dW=0). Covers BOTH layouts:
%     - UNPACKED src+g : 4x4 C_4v d=4  (|G|=128 -> never packs; N=36 path).
%     - PACKED srcg    : icosahedron s=1/2 T1g d=3 (|G|=120 -> packs uint32).
%
%   See also TEST_STREAM, SPILL_ENTRIES_MMAP, MMAP_FILE, RUN_FTLM_PG_SECTOR_GPU_IH.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    assert(exist('mmap_file', 'file') == 3, 'mmap_file MEX not built.');
    gpu_h = gpuDevice;
    ok = true;

    % --- UNPACKED src+g: 4x4 C_4v d=4 ---
    group = square_lattice_spacegroup(4, 4); group.irreps = irreps_from_group(group);
    bonds = adjacency_square_lattice(4, 4);
    cache = enumerate_M_orbits_Ih_gpu(0.5, 0, group);
    entries = collect_clt_entries_Ih(cache.super_reps, bonds, 0.5, 1, group, 'bitmap', 'host');
    p = find(arrayfun(@(z) z.d, group.irreps) == 4, 1);
    ok = ok && check_mmap('4x4-c4v d=4 (unpacked)', entries, cache, group.irreps(p).mats, 4, group, gpu_h);

    % --- PACKED srcg: icosahedron s=1/2 T1g d=3 ---
    g2 = icosahedron_Ih_full(); b2 = adjacency_icosahedron_Ih();
    cache2 = enumerate_M_orbits_Ih_gpu(0.5, 0, g2);
    entries2 = collect_clt_entries_Ih(cache2.super_reps, b2, 0.5, 1, g2, 'bitmap', 'host');
    ok = ok && check_mmap('icosa s=1/2 T1g d=3 (packed)', entries2, cache2, g2.T1g, 3, g2, gpu_h);

    % --- INDEXED c (s>=1): icosahedron s=1 T1g d=3 -- the spilled file gains a
    %     trailing uint8 c-index block ([srcg][c_idx]); clt.mmap_cidx streams it.
    cache3 = enumerate_M_orbits_Ih_gpu(1.0, 0, g2);
    entries3 = collect_clt_entries_Ih(cache3.super_reps, b2, 1.0, 1.0, g2, 'bitmap', 'host');
    assert(entries3.c_is_indexed, 'test_stream_mmap: expected indexed c for s=1');
    ok = ok && check_mmap('icosa s=1 T1g d=3 (c_idx mmap)', entries3, cache3, g2.T1g, 3, g2, gpu_h);

    assert(ok, 'mmap-streamed differs from in-RAM streamed (or only 1 tile)');
    fprintf('\nPASS: out-of-core mmap streaming == in-RAM streaming (bit-identical).\n');
end

% ----------------------------------------------------------------
function ok = check_mmap(label, entries, cache, ir, d, group, gpu_h)
    R = 4; Mlz = 50; B = 2; seed = 999;
    [reps, V, eg, npr, ~, triv] = apply_irrep_to_orbits(cache, ir, d, group);
    n_coll = numel(entries.src_sorted);
    tiny   = ceil(n_coll / 7);          % ~7 rep-tiles

    % Path A: in-RAM streaming (reference).
    eskelA = build_entry_skeleton_Ih(entries, true, tiny);
    cltA = build_clt_skeleton_from_entries_Ih(entries, reps, V, eg, npr, ir, d, group, triv, eskelA, true);
    cltA.force_stream = true;
    cltA.prefix_budget = 0;   % explizit AUS: mmap-Tail-Coverage
    [E0, w0] = run_ftlm_pg_sector_gpu_Ih(cltA, R, Mlz, B, seed, gpu_h);
    cltA = []; wait(gpu_h); %#ok<NASGU>

    % Path B: spill to disk + memory-map, then stream from the mapping.
    eskelB = build_entry_skeleton_Ih(entries, true, tiny);
    cltB = build_clt_skeleton_from_entries_Ih(entries, reps, V, eg, npr, ir, d, group, triv, eskelB, true);
    cltB.force_stream = true;
    cltB.prefix_budget = 0;   % explizit AUS: mmap-Tail-Coverage
    fpath = [tempname, '.entries.bin'];
    [cltB, mh] = spill_entries_mmap(cltB, fpath);
    nt = numel(cltB.tile_e_count);
    [E, w] = run_ftlm_pg_sector_gpu_Ih(cltB, R, Mlz, B, seed, gpu_h);
    cltB = []; wait(gpu_h); %#ok<NASGU>
    cuda_lanczos_clut_block_pg_Ih('cleanup');      % ensure no borrowed ptr before unmap
    mmap_file('close', mh);
    if exist(fpath, 'file'), delete(fpath); end

    dE = max(abs(E - E0)); dW = max(abs(w - w0));
    fprintf('%-32s : mmap vs in-RAM stream  n_tiles=%d  max|dE|=%.3e max|dW|=%.3e\n', ...
            label, nt, dE, dW);
    ok = (dE == 0) && (dW == 0) && (nt >= 2);
end
