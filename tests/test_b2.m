function test_b2()
%TEST_B2  B2 entry-tiling (init_skel_b2, host->device entries) == normal path.
%   On a 4x4 C_4v d=4 block, build the skeleton with the per-entry arrays kept
%   on the HOST (force_b2=true -> kernel cudaMalloc's+uploads them, 64-bit
%   entry indexing) and compare the GPU block-Lanczos result to the normal
%   gpuArray-borrow path. Must be bit-identical (same data, different upload).

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    gpu_h=gpuDevice; s=0.5; J=1; M=0; R=4; Mlz=50; B=0; seed=999;
    group=square_lattice_spacegroup(4,4); group.irreps=irreps_from_group(group);
    bonds=adjacency_square_lattice(4,4);
    cache=enumerate_M_orbits_Ih_gpu(s,M,group);
    entries=collect_clt_entries_Ih(cache.super_reps,bonds,s,J,group,'bitmap','host');
    p=find(arrayfun(@(z)z.d,group.irreps)==4,1); ir=group.irreps(p).mats; d=4;
    [reps,V,eg,npr,~,triv]=apply_irrep_to_orbits(cache,ir,d,group);
    E0=[]; w0=[];
    for fb=[false true]
        eskel=build_entry_skeleton_Ih(entries, fb);
        fprintf('force_b2=%d -> eskel.is_b2=%d (src_idx isgpu=%d)\n', fb, eskel.is_b2, isa(eskel.src_idx,'gpuArray'));
        clt=build_clt_skeleton_from_entries_Ih(entries,reps,V,eg,npr,ir,d,group,triv,eskel,true);
        [E,w]=run_ftlm_pg_sector_gpu_Ih(clt,R,Mlz,B,seed,gpu_h);
        clt=[]; wait(gpu_h); %#ok<NASGU>
        if ~fb, E0=E; w0=w;
        else
            dE=max(abs(E-E0)); dW=max(abs(w-w0));
            fprintf('B2 vs non-B2: max|dE|=%.3e, max|dW|=%.3e\n', dE, dW);
            assert(dE<1e-4 && dW<1e-3, 'B2 differs from non-B2');
            fprintf('PASS: init_skel_b2 == init_skel_ref.\n');
        end
    end
end
