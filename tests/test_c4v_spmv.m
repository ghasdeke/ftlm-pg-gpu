function test_c4v_spmv()
%TEST_C4V_SPMV  GPU OTF SpMV vs CPU sparse H for a c4v space-group d=4 block.
%   Lanczos-free: isolates the kernel matrix elements from any Lanczos/ghost
%   effects. If GPU == CPU sparse here, any FTLM-vs-ED gap is convergence, not
%   a kernel bug.
    gpu_h = gpuDevice; s=0.5; J=1; M=0; rng(11);
    group = square_lattice_spacegroup(4,4); group.irreps = irreps_from_group(group);
    bonds = adjacency_square_lattice(4,4);
    cache   = enumerate_M_orbits_Ih_gpu(s,M,group);
    entries = collect_clt_entries_Ih(cache.super_reps,bonds,s,J,group,'bitmap','host');
    eskel   = build_entry_skeleton_Ih(entries);
    p = find(arrayfun(@(z)z.d,group.irreps)==4,1); ir = group.irreps(p).mats; d=4;
    [reps,V,eg,npr,~,triv] = apply_irrep_to_orbits(cache,ir,d,group);
    nb = double(sum(npr)); fprintf('d4 block n_basis=%d\n',nb);

    H = build_heisenberg_sparse_Ih_gamma2(reps,V,eg,npr,bonds,s,J,ir,d,group);
    H = 0.5*(H+H');

    for use_compact = [false true]
        clt = build_clt_skeleton_from_entries_Ih(entries,reps,V,eg,npr,ir,d,group,triv,eskel,use_compact);
        B = 4; X = randn(nb,B)+1i*randn(nb,B);
        Yg = gpu_otf_spmv(clt, single(real(X)), single(imag(X)), gpu_h);
        Ys = H*X;
        relerr = norm(Ys-Yg,'fro')/norm(Ys,'fro');
        fprintf('compact_v=%d : ||H*X (sparse) - GPU OTF SpMV|| rel = %.3e\n', use_compact, relerr);
    end
end

function Y = gpu_otf_spmv(clt, Xre, Xim, gpu_h)
    assert_kernel_abi();   % direct-MEX caller: bypasses the driver's handshake
    nb = clt.n_basis; nr = clt.n_reps; d = clt.d_irrep; ne = double(sum(clt.entries_per_rep)); B = size(Xre,2);
    g0 = @(t) gpuArray(zeros(0,1,t));
    if isfield(clt,'c_idx'),   ci=clt.c_idx;   else, ci=g0('uint8');  end
    if isfield(clt,'c_table'), ct=clt.c_table; else, ct=g0('single'); end
    if isfield(clt,'srcg'),    sg=clt.srcg;    else, sg=g0('uint32'); end
    if isfield(clt,'triv'),    tv=clt.triv;    else, tv=g0('uint8');  end
    if isfield(clt,'Qbar_re'), qr=clt.Qbar_re; qi=clt.Qbar_im; else, qr=g0('single'); qi=g0('single'); end
    if isfield(clt,'v_slot'),  vs=clt.v_slot;  else, vs=g0('int32');  end
    cuda_lanczos_clut_block_pg_Ih('init_skel_ref', clt.diag_vals, clt.rep_offsets, clt.n_per_rep, ...
        clt.entries_per_rep, clt.entry_offsets, clt.src_idx, clt.g_idx, clt.c_a, ...
        clt.V_re, clt.V_im, clt.rho_re, clt.rho_im, clt.sqrt_eig, ...
        nb, nr, ne, d, B, clt.c_a_const, ci, ct, sg, tv, qr, qi, vs);
    wait(gpu_h);
    [Yr,Yi] = cuda_lanczos_clut_block_pg_Ih('spmv', gpuArray(Xre), gpuArray(Xim));
    cuda_lanczos_clut_block_pg_Ih('cleanup');
    Y = double(Yr) + 1i*double(Yi);
end
