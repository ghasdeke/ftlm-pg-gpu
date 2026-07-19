function validate_spinflip12()
%VALIDATE_SPINFLIP12  Spin-flip Z2 (M=0) pipeline vs direct M=0 ED.
%   Validates the G -> G x Z2 extension (ADD_SPIN_FLIP_Z2 + the flip-aware
%   min_image/enumerate chain) end-to-end on the CPU CLT path, with three
%   independent checks per system:
%     (1) SUM RULE: sum over all Gamma+- blocks of d_Gamma * n_basis equals
%         dim(M=0) exactly (the G x Z2 decomposition tiles the M=0 sector);
%     (2) SPECTRUM: the aggregated M=0 spectrum (each block's eigenvalues
%         replicated d_Gamma-fold) equals the directly assembled M=0-sector
%         Hamiltonian's spectrum to ~1e-10;
%     (3) PAIRING: for each base irrep Gamma, the union of the Gamma+ and
%         Gamma- spectra equals the spectrum of the SAME Gamma block built
%         WITHOUT the flip (the +- subspaces split each Gamma block).
%   Systems:
%     A: kagome N=12 s=1/2  (C_6v space group, |G| 48 -> 96; constant c)
%     B: square 3x3 s=1     (C_4v space group, |G| 72 -> 144; indexed c,
%        d_loc=3 exercises the non-power-of-two digit path of the flip)
%
%   See also ADD_SPIN_FLIP_Z2, MIN_IMAGE_IH, ENUMERATE_M_ORBITS_IH_GPU.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    % --- System A: kagome N=12, s=1/2 ---
    [gA, bondsA] = kagome_spacegroup(2, 0);
    gA.irreps = irreps_from_group(gA);
    run_system('kagome N=12 s=1/2', gA, bondsA, 12, 0.5, 1.0);

    % gpu_native collect parity for the FLIP group (deprecated path, but
    % flip-safe via MIN_IMAGE_IH_GPU): host and gpu_native collects must
    % produce identical sorted entry tables under G x Z2.
    if gpuDeviceCount > 0
        gA2 = add_spin_flip_z2(gA);
        cA  = enumerate_M_orbits_Ih_gpu(0.5, 0, gA2);
        eh  = collect_clt_entries_Ih(cA.super_reps, bondsA, 0.5, 1.0, gA2, ...
                                     'schnack', 'host');
        eg  = collect_clt_entries_Ih_gpu(cA.super_reps, bondsA, 0.5, 1.0, gA2);
        ok_gn = isequal(eh.src_sorted, gather(eg.src_sorted)) && ...
                isequal(eh.tgt_sorted, gather(eg.tgt_sorted)) && ...
                isequal(eh.g_sorted,   gather(eg.g_sorted));
        fprintf('  gpu_native collect (flip group): host == gpu entries: %d\n', ok_gn);
        assert(ok_gn, 'gpu_native collect differs from host under spin-flip');
    end

    % --- System B: square 3x3, s=1 ---
    gB = square_lattice_spacegroup(3, 3);
    gB.irreps = irreps_from_group(gB);
    bondsB = adjacency_square_lattice(3, 3);
    run_system('square 3x3 s=1', gB, bondsB, 9, 1.0, 1.0);

    fprintf('\nPASS: spin-flip Z2 (M=0) == direct M=0 ED on both systems.\n');
end


% ----------------------------------------------------------------
function run_system(label, group, bonds, N, s_val, J)
    fprintf('\n=== %s ===\n', label);
    group2 = add_spin_flip_z2(group);

    % Reference: the M=0 sector Hamiltonian assembled directly in the
    % product basis (independent of all pipeline machinery).
    [E_ref, dimM0] = ed_M0_sector(bonds, N, s_val, J);
    fprintf('  dim(M=0) = %d, |G| = %d -> |G x Z2| = %d\n', ...
            dimM0, group.order, group2.order);

    % Flip pipeline: enumerate -> collect -> per-(Gamma+-) dense blocks.
    [E_blocks2, names2, sumdim2] = all_blocks(group2, bonds, s_val, J);
    % Non-flip pipeline (for the pairing check).
    [E_blocks1, names1, sumdim1] = all_blocks(group,  bonds, s_val, J);

    % (1) Sum rule (both decompositions must tile the sector).
    assert(sumdim2 == dimM0, '%s: flip sum rule %d != dim(M=0) %d', ...
           label, sumdim2, dimM0);
    assert(sumdim1 == dimM0, '%s: non-flip sum rule %d != dim(M=0) %d', ...
           label, sumdim1, dimM0);
    fprintf('  (1) sum rule: %d == dim(M=0)  OK (flip and non-flip)\n', sumdim2);

    % (2) Aggregated flip spectrum vs direct ED.
    E_all = sort(vertcat(E_blocks2{:}));
    assert(numel(E_all) == dimM0, '%s: aggregated count %d != %d', ...
           label, numel(E_all), dimM0);
    err2 = max(abs(E_all - E_ref)) / max(1, max(abs(E_ref)));
    fprintf('  (2) aggregated spectrum vs M=0 ED: max rel.err = %.3e\n', err2);
    assert(err2 < 1e-9, '%s: flip spectrum != M=0 ED (%.2e)', label, err2);

    % (3) Gamma+ u Gamma- == Gamma (non-flip), per base irrep.
    err3 = 0;
    for k = 1 : numel(names1)
        Ep = pick_block(E_blocks2, names2, [names1{k} '+']);
        Em = pick_block(E_blocks2, names2, [names1{k} '-']);
        Eu = sort([Ep; Em]);
        E1 = sort(E_blocks1{k});
        assert(numel(Eu) == numel(E1), '%s: %s pairing size %d != %d', ...
               label, names1{k}, numel(Eu), numel(E1));
        if ~isempty(E1)
            err3 = max(err3, max(abs(Eu - E1)) / max(1, max(abs(E1))));
        end
    end
    fprintf('  (3) Gamma+ u Gamma- == Gamma: max rel.err = %.3e\n', err3);
    assert(err3 < 1e-9, '%s: +- pairing broken (%.2e)', label, err3);
end


% ----------------------------------------------------------------
function [E_blocks, names, sumdim] = all_blocks(group, bonds, s_val, J)
%ALL_BLOCKS  Dense per-irrep block spectra via the CPU CLT chain. Each
%   block's eigenvalues are replicated d_Gamma-fold (partner rows).
    cache   = enumerate_M_orbits_Ih_gpu(s_val, 0, group);
    entries = collect_clt_entries_Ih(cache.super_reps, bonds, s_val, J, ...
                                     group, 'bitmap', 'host');
    nirr     = numel(group.irreps);
    E_blocks = cell(nirr, 1);
    names    = cell(nirr, 1);
    sumdim   = 0;
    for k = 1 : nirr
        ir = group.irreps(k);
        names{k} = ir.name;
        [reps, V, eg, npr] = apply_irrep_to_orbits(cache, ir.mats, ir.d, group);
        nb = double(sum(npr));
        if nb == 0
            E_blocks{k} = zeros(0, 1);
            continue;
        end
        clt = build_clt_from_entries_Ih(entries, reps, V, eg, npr, ...
                                        ir.mats, ir.d, group);
        Hd  = spmv_pg_Ih_clt_matlab(clt, eye(nb));
        Hd  = (Hd + Hd') / 2;
        Eb  = sort(real(eig(Hd)));
        E_blocks{k} = repmat(Eb, ir.d, 1);     % d_Gamma partner-row copies
        sumdim      = sumdim + ir.d * nb;
    end
end


% ----------------------------------------------------------------
function Eb = pick_block(E_blocks, names, name)
    k = find(strcmp(names, name), 1);
    assert(~isempty(k), 'block %s not found', name);
    Eb = E_blocks{k};
end


% ----------------------------------------------------------------
function [E, dimM0] = ed_M0_sector(bonds, N, s_val, J)
%ED_M0_SECTOR  Assemble + diagonalise H restricted to the M=0 product basis.
%   Mirrors the Heisenberg conventions of COLLECT_CLT_ENTRIES_IH (diagonal
%   J*Sz_i*Sz_j per bond + 0.5*J*sqrt(..)*sqrt(..) hop both directions) but
%   shares NO code with the pipeline under test.
    d_loc = round(2*s_val + 1);
    n_tot = d_loc^N;
    A_tot = round(N * s_val);                  % digit sum of the M=0 sector

    states = (0 : n_tot - 1)';
    dig    = zeros(n_tot, N);
    tmp    = states;
    for k = 1 : N
        dig(:, k) = mod(tmp, d_loc);
        tmp = floor(tmp / d_loc);
    end
    keep   = (sum(dig, 2) == A_tot);
    bas    = states(keep);
    m      = dig(keep, :) - s_val;             % [dim x N] m quantum numbers
    dimM0  = numel(bas);
    lut          = zeros(n_tot, 1);
    lut(bas + 1) = (1 : dimM0)';

    dvals = zeros(dimM0, 1);
    for b = 1 : size(bonds, 1)
        dvals = dvals + J * m(:, bonds(b, 1)) .* m(:, bonds(b, 2));
    end

    powers = d_loc .^ (0 : N - 1)';
    rows = {}; cols = {}; vals = {};
    for b = 1 : size(bonds, 1)
        si = bonds(b, 1); sj = bonds(b, 2);
        for dir = 1 : 2
            if dir == 1     % S+_i S-_j
                can = (m(:, si) < s_val - 1e-10) & (m(:, sj) > -s_val + 1e-10);
                if ~any(can), continue; end
                msi = m(can, si); msj = m(can, sj);
                c   = 0.5 * J ...
                    .* sqrt(s_val*(s_val+1) - msi .* (msi + 1)) ...
                    .* sqrt(s_val*(s_val+1) - msj .* (msj - 1));
                tgt = bas(can) + powers(si) - powers(sj);
            else            % S-_i S+_j
                can = (m(:, si) > -s_val + 1e-10) & (m(:, sj) < s_val - 1e-10);
                if ~any(can), continue; end
                msi = m(can, si); msj = m(can, sj);
                c   = 0.5 * J ...
                    .* sqrt(s_val*(s_val+1) - msi .* (msi - 1)) ...
                    .* sqrt(s_val*(s_val+1) - msj .* (msj + 1));
                tgt = bas(can) - powers(si) + powers(sj);
            end
            ti = lut(tgt + 1);
            assert(all(ti > 0), 'ed_M0_sector: hop left the M=0 sector');
            rows{end+1} = ti;        %#ok<AGROW>
            cols{end+1} = find(can); %#ok<AGROW>
            vals{end+1} = c;         %#ok<AGROW>
        end
    end
    H = sparse(vertcat(rows{:}), vertcat(cols{:}), vertcat(vals{:}), ...
               dimM0, dimM0) + spdiags(dvals, 0, dimM0, dimM0);
    H = full((H + H') / 2);
    E = sort(real(eig(H)));
end
