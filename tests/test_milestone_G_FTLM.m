function test_milestone_G_FTLM()
%TEST_MILESTONE_G_FTLM  Verify the full I_h-FTLM pipeline against full ED.
%
%   For the s=1/2 Heisenberg icosahedron the test runs three paths and
%   compares thermodynamic observables on a common T grid:
%
%   (A) Full-ED reference: per-M dense ED via the unsymmetrized sparse
%       builder + aggregation in compute_observables_pg.
%
%   (B) I_h-ED-aggregate: ftlm_observables_pg_Ih with ed_thresh = inf,
%       i.e., dense ED on every (M, Gamma) block via the gamma.2
%       column-picking construction, then aggregation with the
%       mult_M * d_Gamma weights.
%
%   (C) I_h-FTLM: ftlm_observables_pg_Ih with finite R, M_lz, FTLM on
%       every block. Must converge to (A) up to a tolerable Monte-Carlo
%       error for the chosen R, M_lz.
%
%   Path (B) must agree with (A) to machine precision; this validates:
%       - gamma.2 block construction for all 10 irreps (non-Abelian
%         included),
%       - aggregation with mult_M * d_Gamma weights,
%       - the sum-rule sum_i w_i = full Hilbert dimension.
%
%   Path (C) is a softer check on the FTLM kernel. The s=1/2
%   icosahedron is a small system: the largest (M, Gamma) block has
%   dimension 46 (H_g, M = 0). Inside RUN_FTLM_PG_SECTOR the effective
%   number of random vectors is internally capped at R_eff = min(R, D),
%   so the statistical weight noise per eigenvalue cannot drop below
%   ~ 1 / sqrt(D) ~ 15% on this system, no matter how large R is set.
%   At very low T (kT << gap to first excited state), C(T) is dominated
%   by a few eigenvalues and inherits this noise; chi(T) is less
%   sensitive because the M^2 weights average over many states.
%   On publication-relevant systems (s >= 1, dim_M in the thousands)
%   D is much larger and the noise floor drops accordingly. The
%   tolerances below (20% on C, 5% on chi) reflect the s=1/2
%   icosahedron noise floor; Path (C) confirms the kernel runs and
%   converges to ED-aggregate within statistical expectation.
%
%   Run from MATLAB inside the mit_pg directory:
%       >> test_milestone_G_FTLM
%
%   No GPU, no MEX. Pure MATLAB. The test creates two .mat output files
%   in a temporary directory; they are NOT cleaned up automatically so
%   one can inspect the per-sector diagnostics.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    fprintf('=== Milestone G FTLM: I_h-FTLM observables vs full ED ===\n\n');

    s_val   = 0.5;
    J       = 1.0;
    T_range = [0.1, 0.3, 1.0, 3.0, 10.0];

    %% Path A: full-ED reference (no point-group symmetry, only M)
    [C_full, chi_full, Z_full, n_total] = full_ED_reference(s_val, J, T_range);
    fprintf('Path A (full ED, M-sector aggregation):\n');
    fprintf('  n_total = %d, C(T) and chi(T) computed at %d temperatures.\n\n', ...
            n_total, numel(T_range));

    %% Common output dir for paths B and C
    out_dir = tempname();
    [ok, msg] = mkdir(out_dir);
    if ~ok
        error('test_milestone_G_FTLM:mkdir', 'mkdir %s: %s', out_dir, msg);
    end

    %% Path B: I_h-ED-aggregate via ftlm_observables_pg_Ih(ed_thresh=inf)
    input_B = fullfile(out_dir, 'input_B.m');
    write_input(input_B, s_val, J, 1, 1, T_range, inf);
    fprintf('Path B (I_h-ED-aggregate via ftlm_observables_pg_Ih, ed_thresh=inf):\n');
    ftlm_observables_pg_Ih(input_B);
    matB = load(fullfile(pwd, get_outname(s_val)));
    [pass_B, err_B] = compare_split(C_full, chi_full, matB.C_T, matB.chi_T, ...
                                     1e-10, 1e-10);
    fprintf('  max relative err vs Path A: C = %.2e, chi = %.2e  [%s]\n\n', ...
            err_B.C, err_B.chi, ternary(pass_B, 'OK', 'FAIL'));

    %% Path C: I_h-FTLM with finite R, M_lz, ed_thresh = 5
    %  ed_thresh = 5 lets blocks with n_basis <= 5 be diagonalised
    %  exactly (they are deterministic anyway when M_lz_actual = D),
    %  isolating the FTLM kernel to the larger blocks.
    R       = 30;
    M_lz    = 80;
    ed_C    = 5;
    tol_C_C   = 0.20;     % 20% on C(T): low-T floor on 46-dim block
    tol_C_chi = 0.05;     %  5% on chi(T)
    input_C = fullfile(out_dir, 'input_C.m');
    write_input(input_C, s_val, J, R, M_lz, T_range, ed_C);
    fprintf('Path C (I_h-FTLM with R=%d, M_lz=%d, ed_thresh=%d):\n', R, M_lz, ed_C);
    ftlm_observables_pg_Ih(input_C);
    matC = load(fullfile(pwd, get_outname(s_val)));
    [pass_C, err_C] = compare_split(C_full, chi_full, matC.C_T, matC.chi_T, ...
                                     tol_C_C, tol_C_chi);
    fprintf('  max relative err vs Path A: C = %.2e (tol %.0f%%), chi = %.2e (tol %.0f%%)  [%s]\n\n', ...
            err_C.C, 100*tol_C_C, err_C.chi, 100*tol_C_chi, ...
            ternary(pass_C, 'OK', 'FAIL'));

    %% Per-T printout for the FTLM run
    fprintf('Per-T comparison (Path C vs Path A):\n');
    fprintf('   T        C_full       C_FTLM    relErr_C      chi_full    chi_FTLM   relErr_chi\n');
    for iT = 1 : numel(T_range)
        T = T_range(iT);
        relC = abs(matC.C_T(iT) - C_full(iT)) / max(abs(C_full(iT)), 1e-12);
        relX = abs(matC.chi_T(iT) - chi_full(iT)) / max(abs(chi_full(iT)), 1e-12);
        fprintf('  %5.2f  %10.6f  %10.6f  %9.2e   %10.6f  %10.6f  %9.2e\n', ...
            T, C_full(iT), matC.C_T(iT), relC, ...
            chi_full(iT), matC.chi_T(iT), relX);
    end
    fprintf('\n');

    %% Verdict
    overall = pass_B && pass_C;
    fprintf('=========================================================\n');
    fprintf('OVERALL Milestone G FTLM: %s\n', ternary(overall, 'PASS', 'FAIL'));
end


% ----------------------------------------------------------------
function name = get_outname(s_val)
    two_s = round(2 * s_val);
    if mod(two_s, 2) == 0, s_str = sprintf('%d', two_s/2);
    else,                  s_str = sprintf('%do2', two_s);
    end
    name = sprintf('ftlm_pg_Ih_icos_s%s.mat', s_str);
end

% ----------------------------------------------------------------
function write_input(path, s_val, J, R, M_lz, T_range, ed_thresh)
    fid = fopen(path, 'w');
    fprintf(fid, '%% Auto-generated input for test_milestone_G_FTLM\n');
    fprintf(fid, 's_val = %g;\n', s_val);
    fprintf(fid, 'J     = %g;\n', J);
    fprintf(fid, 'R     = %d;\n', R);
    fprintf(fid, 'M_lz  = %d;\n', M_lz);
    fprintf(fid, 'T_range = [%s];\n', strjoin(arrayfun(@(x) sprintf('%.10g', x), T_range, 'uni', 0), ', '));
    if isinf(ed_thresh)
        fprintf(fid, 'ed_thresh = inf;\n');
    else
        fprintf(fid, 'ed_thresh = %g;\n', ed_thresh);
    end
    fclose(fid);
end


% ----------------------------------------------------------------
function [C_T, chi_T, Z_eff, n_total] = full_ED_reference(s_val, J, T_range)
% Full-ED reference: per-M dense diagonalization with no point-group
% symmetry. Aggregates via compute_observables_pg.
    N_sites = 12;
    d_loc   = round(2*s_val + 1);
    n_total = double(d_loc)^N_sites;
    M_max   = round(N_sites * s_val);
    bonds   = adjacency_icosahedron_Ih();

    all_E = []; all_w = []; all_M = [];
    for M = 0 : M_max
        mult_M = 1 + (M > 0);
        [basis_M, dim_M] = enumerate_M_basis(N_sites, s_val, d_loc, M);
        if dim_M == 0, continue; end
        H = build_H_M(basis_M, bonds, s_val, J, N_sites, n_total);
        E = sort(real(eig(full(H))));
        w = mult_M * ones(numel(E), 1);
        all_E = [all_E; E];                         %#ok<AGROW>
        all_w = [all_w; w];                         %#ok<AGROW>
        all_M = [all_M; M * ones(numel(E), 1)];     %#ok<AGROW>
    end
    [C_T, chi_T, Z_eff] = compute_observables_pg(all_E, all_w, all_M, T_range);
end


% ----------------------------------------------------------------
function [pass, err] = compare_split(C_ref, chi_ref, C_test, chi_test, tol_C, tol_chi)
    rel_C   = abs(C_test(:)   - C_ref(:))   ./ max(abs(C_ref(:)),   1e-12);
    rel_chi = abs(chi_test(:) - chi_ref(:)) ./ max(abs(chi_ref(:)), 1e-12);
    err.C   = max(rel_C);
    err.chi = max(rel_chi);
    pass    = (err.C < tol_C) && (err.chi < tol_chi);
end


% ----------------------------------------------------------------
function [basis, dim] = enumerate_M_basis(N_sites, s_val, d_loc, M_target)
    n_total = int64(d_loc)^int64(N_sites);
    if s_val == 0.5
        k_up = N_sites/2 + M_target;
        if k_up < 0 || k_up > N_sites || abs(k_up - round(k_up)) > 0.01
            basis = int64([]); dim = 0; return;
        end
        k_up = round(k_up);
        all_states = int64(0 : double(n_total) - 1)';
        pop = zeros(numel(all_states), 1);
        for b = 0 : N_sites - 1
            pop = pop + double(bitand(bitshift(all_states, -b), int64(1)));
        end
        basis = all_states(pop == k_up);
    else
        all_states = int64(0 : double(n_total) - 1)';
        Mv = zeros(numel(all_states), 1);
        tmp = all_states;
        for kk = 1 : N_sites
            dg = double(mod(tmp, int64(d_loc)));
            Mv = Mv + dg - s_val;
            tmp = (tmp - int64(dg)) / int64(d_loc);
        end
        Mv = round(Mv * 2) / 2;
        basis = all_states(Mv == M_target);
    end
    dim = numel(basis);
end


% ----------------------------------------------------------------
function H = build_H_M(basis, bonds, s_val, J, N_sites, n_total)
% Sparse Heisenberg H on the unsymmetrized M-sector. Identical to the
% helper used in test_milestone_G_gamma2.
    d_loc = round(2*s_val + 1);
    dim = numel(basis);
    n_b = size(bonds, 1);
    powers = int64(d_loc).^int64((0 : N_sites - 1)');
    lookup = zeros(n_total, 1, 'int32');
    lookup(double(basis) + 1) = int32(1 : dim);

    mi = zeros(dim, N_sites);
    tmp = basis;
    for site = 1 : N_sites
        dg = double(mod(tmp, int64(d_loc)));
        mi(:, site) = dg - s_val;
        tmp = (tmp - int64(dg)) / int64(d_loc);
    end

    diag_vals = zeros(dim, 1);
    for b = 1 : n_b
        diag_vals = diag_vals + J * mi(:, bonds(b,1)) .* mi(:, bonds(b,2));
    end

    cap = dim * (1 + 2*n_b);
    rows = zeros(cap, 1); cols = zeros(cap, 1); vals = zeros(cap, 1);
    rows(1:dim) = (1:dim)'; cols(1:dim) = (1:dim)'; vals(1:dim) = diag_vals;
    nz = dim;
    for b = 1 : n_b
        si = bonds(b,1); sj = bonds(b,2);
        can = (mi(:,si) < s_val - 1e-10) & (mi(:,sj) > -s_val + 1e-10);
        idx_from = find(can);
        if ~isempty(idx_from)
            m_i = mi(idx_from, si); m_j = mi(idx_from, sj);
            c = 0.5*J * sqrt(s_val*(s_val+1) - m_i.*(m_i+1)) .* ...
                          sqrt(s_val*(s_val+1) - m_j.*(m_j-1));
            new_states = basis(idx_from) + powers(si) - powers(sj);
            ns1 = double(new_states) + 1;
            ok = ns1 >= 1 & ns1 <= n_total;
            ni = zeros(numel(idx_from), 1, 'int32');
            ni(ok) = lookup(ns1(ok));
            v = ni > 0; nv = sum(v);
            rows(nz+1:nz+nv) = double(ni(v));
            cols(nz+1:nz+nv) = double(idx_from(v));
            vals(nz+1:nz+nv) = c(v);
            nz = nz + nv;
        end
        can = (mi(:,si) > -s_val + 1e-10) & (mi(:,sj) < s_val - 1e-10);
        idx_from = find(can);
        if ~isempty(idx_from)
            m_i = mi(idx_from, si); m_j = mi(idx_from, sj);
            c = 0.5*J * sqrt(s_val*(s_val+1) - m_i.*(m_i-1)) .* ...
                          sqrt(s_val*(s_val+1) - m_j.*(m_j+1));
            new_states = basis(idx_from) - powers(si) + powers(sj);
            ns1 = double(new_states) + 1;
            ok = ns1 >= 1 & ns1 <= n_total;
            ni = zeros(numel(idx_from), 1, 'int32');
            ni(ok) = lookup(ns1(ok));
            v = ni > 0; nv = sum(v);
            rows(nz+1:nz+nv) = double(ni(v));
            cols(nz+1:nz+nv) = double(idx_from(v));
            vals(nz+1:nz+nv) = c(v);
            nz = nz + nv;
        end
    end
    H = sparse(rows(1:nz), cols(1:nz), vals(1:nz), dim, dim);
    H = 0.5 * (H + H');
end


function s = ternary(cond, a, b)
    if cond, s = a; else, s = b; end
end
