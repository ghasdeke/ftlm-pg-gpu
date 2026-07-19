function [super_reps, V_per_rep, eig_per_rep, n_per_rep, L_per_rep, triv_active] = ...
            apply_irrep_to_orbits(cache, irrep_data, d_irrep, group)
%APPLY_IRREP_TO_ORBITS  Per-(M, Gamma) finalisation of the gamma.2 basis.
%
%   [SUPER_REPS, V_PER_REP, EIG_PER_REP, N_PER_REP, L_PER_REP] = ...
%       APPLY_IRREP_TO_ORBITS(CACHE, IRREP_DATA, D_IRREP, GROUP)
%
%   takes the irrep-INDEPENDENT cache from ENUMERATE_M_ORBITS_IH and an
%   irreducible representation Gamma of I_h, and returns the per-rep
%   column-basis matrices V_r and eigenvalues lambda_r,k for the (M,
%   Gamma) block. Only the reps with n_Gamma(r) > 0 are returned.
%
%   For each cached super-rep r and irrep Gamma:
%       A^(r) = sum_{h in Stab(r)} rho_Gamma(h)*
%   is summed using the precomputed stabiliser list (no
%   apply_perm_to_state calls here), Hermitised, and eigendecomposed.
%   The non-zero-eigenvalue columns of the eigenvector matrix form V_r;
%   their count is n_Gamma(r) and equals (1/|Stab|) * sum_{h in Stab}
%   chi_Gamma(h).
%
%   Inputs:
%       cache        struct from ENUMERATE_M_ORBITS_IH
%       irrep_data   1D characters [120 x 1] or [d x d x 120] matrices
%       d_irrep      irrep dimension (1..5 for I_h)
%       group        struct from ICOSAHEDRON_IH_FULL
%
%   Outputs (length n_reps_active, the reps with n_Gamma > 0):
%       super_reps    sorted int64 column of orbit minima
%       V_per_rep     {n_reps_active} cell of d x n_Gamma(i) eigenvector
%                     matrices
%       eig_per_rep   {n_reps_active} cell of n_Gamma(i) non-zero
%                     eigenvalues
%       n_per_rep     int32, n_Gamma per rep
%       L_per_rep     int32, I_h orbit length per rep
%
%   See also ENUMERATE_M_ORBITS_IH, ENUMERATE_SECTOR_WITH_IH_GAMMA2.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    n_cand = numel(cache.super_reps);
    if n_cand == 0
        super_reps = int64([]); V_per_rep = {}; eig_per_rep = {};
        n_per_rep = int32([]); L_per_rep = int32([]); triv_active = false(0,1);
        return;
    end

    %% Precompute conj(rho_Gamma(g)) for all 120 group elements once.
    %  This is the dominant per-Gamma setup cost when d_irrep > 1.
    rho_star = cell(group.order, 1);
    for g = 1 : group.order
        rho_star{g} = conj(irrep_matrix(irrep_data, g, d_irrep));
    end

    %% PASS 1: keep decisions (no per-cand cell arrays -- wave-2 [1] step 1:
    %  the old flow populated TWO n_cand cell arrays and then made a SECOND
    %  filtered copy while both were alive; at ~110-150 B of mxArray header
    %  per cell that was a ~17-40 GB host transient at dodec/N=36 scale.
    %  Now: decide keep_mask first, then fill the FINAL n_active outputs
    %  directly. Values and ordering are byte-identical.)
    %
    % FAST PATH: reps with a TRIVIAL stabiliser (Stab = {e}, equivalently
    % orbit length == |G|) ALL share the SAME projector A_r =
    % conj(rho_Gamma(e)) -> ONE eigendecomposition, broadcast on fill.
    % This is exactly what the per-rep loop produced for these reps, hence
    % byte-identical -- including the d = 5 H irreps, whose stored
    % rho_Gamma(e) carries tiny numerical noise (so it is NOT exactly the
    % identity; assuming V = I_d, eig = 1 would NOT match the eig() output).
    tol  = 1e-8;
    triv = (cache.orbit_lens(:) == group.order);
    nt   = sum(triv);
    keep_mask = false(n_cand, 1);
    Ve_keep = [];  eig_e_keep = [];  n_e_keep = int32(0);
    if nt > 0
        A_e = 0.5 * (rho_star{group.identity} + rho_star{group.identity}');
        [Ve, De] = eig(A_e);
        eigs_e = real(diag(De));
        [eigs_e, ord_e] = sort(eigs_e, 'descend');
        Ve = Ve(:, ord_e);
        keep_e = abs(eigs_e) > tol;
        if any(keep_e)
            Ve_keep    = Ve(:, keep_e);
            eig_e_keep = eigs_e(keep_e);
            n_e_keep   = int32(sum(keep_e));
            keep_mask(triv) = true;
        end
    end

    % SLOW PATH: only the (few) reps with a non-trivial stabiliser need the
    % per-rep projector sum A_r = sum_{h in Stab} conj(rho_Gamma(h)) and its
    % eigendecomposition. Results parked in SMALL per-nontrivial cells.
    nontriv_idx = find(~triv).';
    n_nt   = numel(nontriv_idx);
    V_nt   = cell(n_nt, 1);
    eig_nt = cell(n_nt, 1);
    n_nt_a = zeros(n_nt, 1, 'int32');
    for k = 1 : n_nt
        i = nontriv_idx(k);
        % CSR stabilisers: rep i's fixing group elements (uint16 indices into
        % rho_star) are stab_flat(stab_ptr(i):stab_ptr(i+1)-1).
        stab = cache.stab_flat(cache.stab_ptr(i) : cache.stab_ptr(i+1) - 1);
        A_r = zeros(d_irrep, d_irrep);
        for kk = 1 : numel(stab)
            A_r = A_r + rho_star{stab(kk)};
        end
        A_r = 0.5 * (A_r + A_r');     % suppress FP noise

        [V, D] = eig(A_r);
        eigs_r = real(diag(D));
        [eigs_r, ord] = sort(eigs_r, 'descend');
        V = V(:, ord);
        keep = abs(eigs_r) > tol;
        if any(keep)
            V_nt{k}   = V(:, keep);
            eig_nt{k} = eigs_r(keep);
            n_nt_a(k) = int32(sum(keep));
            keep_mask(i) = true;
        end
    end

    %% PASS 2: fill the FINAL outputs once, at n_active.
    super_reps  = cache.super_reps(keep_mask);
    L_per_rep   = cache.orbit_lens(keep_mask);
    triv_active = triv(keep_mask);
    n_active    = sum(keep_mask);
    V_per_rep   = cell(n_active, 1);
    eig_per_rep = cell(n_active, 1);
    n_per_rep   = zeros(n_active, 1, 'int32');
    pos = zeros(n_cand, 1);  pos(keep_mask) = 1 : n_active;
    if any(keep_mask & triv)
        ta = pos(keep_mask & triv);
        V_per_rep(ta)   = {Ve_keep};      % shared-data references, as before
        eig_per_rep(ta) = {eig_e_keep};
        n_per_rep(ta)   = n_e_keep;
    end
    for k = 1 : n_nt
        i = nontriv_idx(k);
        if keep_mask(i)
            j = pos(i);
            V_per_rep{j}   = V_nt{k};
            eig_per_rep{j} = eig_nt{k};
            n_per_rep(j)   = n_nt_a(k);
        end
    end
end


% ----------------------------------------------------------------
function M = irrep_matrix(irrep_data, g, d)
    % Preserve the REAL type of realified irreps (FS=+1, see REALIFY_IRREPS):
    % a real irrep_data gives a real projector A_r, hence real eigenvectors V,
    % hence real H / CLT downstream. Complex irreps (I_h T1g..Hu, momentum k)
    % keep the complex path exactly as before.
    if d == 1
        v = irrep_data(g);
    else
        v = irrep_data(:, :, g);
    end
    if isreal(irrep_data)
        M = v;
    else
        M = complex(v);
    end
end
