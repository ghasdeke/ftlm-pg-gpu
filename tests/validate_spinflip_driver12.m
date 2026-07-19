function validate_spinflip_driver12()
%VALIDATE_SPINFLIP_DRIVER12  use_spin_flip in the GPU driver vs full ED (N=12).
%   Two end-to-end runs of FTLM_OBSERVABLES_PG_GPU_IH on the kagome N=12
%   torus with use_spin_flip = true:
%     A: full M sweep, ed_thresh = inf (exact ED of every block; M=0 runs
%        with G x Z2 / Gamma+-, all other M with the base group):
%        sum rule == 2^12 exact and C(T) == full-ED C(T) to ~1e-8;
%     B: only_M0, ed_thresh = 0 (FTLM through the GPU FP32 kernel -> the
%        2|G|-sliced rho/Qbar tables and the doubled-stabiliser V blocks
%        are consumed by the CUDA path): FTLM sum rule == dim(M=0) = 924
%        (exact bookkeeping, R/M_lz-independent) and E0(M=0) matches ED to
%        FP32 accuracy. The blocks are tiny, so M_lz spans the full Krylov
%        space and the Ritz values are exact up to FP32 SpMV rounding.
%
%   See also ADD_SPIN_FLIP_Z2, VALIDATE_SPINFLIP12, VALIDATE_KAGOME12.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    outd = fullfile(tempdir, 'validate_spinflip_driver12');
    if ~exist(outd, 'dir'), mkdir(outd); end
    cleanup = onCleanup(@() rmdir(outd, 's'));
    mat_path = fullfile(outd, 'ftlm_pg_gpu_Ih_kag2_0_s1o2.mat');

    N = 12; s_val = 0.5; J = 1.0;
    [~, bonds] = kagome_spacegroup(2, 0);

    % --- Run A: exact ED of every block, full M sweep, flip at M=0 ---
    in_a = fullfile(outd, 'input_sf_ed.m');
    write_input(in_a, outd, 'ed_thresh = inf; only_M0 = false;');
    fprintf('\n>>> Run A: GPU driver, use_spin_flip, ED all blocks, full M sweep\n');
    ftlm_observables_pg_gpu_Ih(in_a);
    ra = load(mat_path);

    sumw_err = abs(sum(ra.all_w) - 2^N) / 2^N;
    E_full = ed_full_heisenberg(bonds, N, s_val, J);
    C_ref  = compute_observables_pg(E_full, ones(numel(E_full), 1), ...
                                    zeros(numel(E_full), 1), ra.T_range(:)');
    nz   = C_ref > 1e-9 * max(C_ref);
    errC = max(abs(ra.C_T(nz) - C_ref(nz)) ./ C_ref(nz));
    fprintf('\n=== Run A checks ===\n');
    fprintf('  sum rule: sum_w = %.6g (expect %d), rel.err = %.2e\n', ...
            sum(ra.all_w), 2^N, sumw_err);
    fprintf('  C(T) vs full ED: max rel.err = %.3e\n', errC);
    assert(sumw_err < 1e-10, 'Run A sum rule violated: %.2e', sumw_err);
    assert(errC < 1e-8, 'Run A C(T) != full ED: %.2e', errC);

    % --- Run B: FTLM via the GPU kernel, M=0 only, flip ---
    in_b = fullfile(outd, 'input_sf_gpu.m');
    write_input(in_b, outd, 'ed_thresh = 0; only_M0 = true;');
    fprintf('\n>>> Run B: GPU driver, use_spin_flip, FP32 kernel FTLM, M=0 only\n');
    ftlm_observables_pg_gpu_Ih(in_b);
    rb = load(mat_path);

    dimM0 = nchoosek(N, N/2);
    sumw_err_b = abs(sum(rb.all_w) - dimM0) / dimM0;
    % E0(M=0) reference from the exact run A (its M=0 part).
    E0_ref = min(ra.all_E(ra.all_M == 0));
    dE0    = abs(min(rb.all_E) - E0_ref) / max(1, abs(E0_ref));
    fprintf('\n=== Run B checks ===\n');
    fprintf('  FTLM sum rule: sum_w = %.6g (expect %d), rel.err = %.2e\n', ...
            sum(rb.all_w), dimM0, sumw_err_b);
    fprintf('  E0(M=0) GPU FTLM vs ED: rel.err = %.3e\n', dE0);
    assert(sumw_err_b < 1e-8, 'Run B FTLM sum rule violated: %.2e', sumw_err_b);
    assert(dE0 < 1e-3, 'Run B E0 mismatch (FP32 kernel): %.2e', dE0);

    % --- Run C: CPU driver (ftlm_observables_pg_Ih), ED all blocks, flip ---
    %  Exercises the CPU enumerate's flip-aware stabiliser pass
    %  (enumerate_M_orbits_Ih applies only the permutation half; flip
    %  elements stabilise via g*r == C - r) and the CPU driver wiring.
    in_c = fullfile(outd, 'input_sf_cpu.m');
    write_input(in_c, outd, 'ed_thresh = inf; only_M0 = false;');
    fprintf('\n>>> Run C: CPU driver, use_spin_flip, ED all blocks, full M sweep\n');
    rc = ftlm_observables_pg_Ih(in_c);
    sumw_err_c = abs(rc.sum_w - 2^N) / 2^N;
    errC_c     = max(abs(rc.C_T(nz) - C_ref(nz)) ./ C_ref(nz));
    fprintf('\n=== Run C checks ===\n');
    fprintf('  sum rule: sum_w = %.6g (expect %d), rel.err = %.2e\n', ...
            rc.sum_w, 2^N, sumw_err_c);
    fprintf('  C(T) vs full ED: max rel.err = %.3e\n', errC_c);
    assert(sumw_err_c < 1e-10, 'Run C sum rule violated: %.2e', sumw_err_c);
    assert(errC_c < 1e-8, 'Run C C(T) != full ED: %.2e', errC_c);

    fprintf('\nPASS: use_spin_flip driver paths (GPU ED + GPU kernel + CPU) validated vs full ED.\n');
end


% ----------------------------------------------------------------
function write_input(path, outd, extra)
    fid = fopen(path, 'w');
    fprintf(fid, [ ...
        'geometry = ''kagome''; kag_a = 2; kag_b = 0;\n', ...
        's_val = 0.5; J = 1.0; R = 8; M_lz = 60; B_gpu = 0;\n', ...
        'use_spin_flip = true; checkpoint = false;\n', ...
        'lookup_method = ''bitmap''; entries_storage = ''host'';\n', ...
        'T_range = logspace(-1, 1, 40);\n', ...
        'output_dir = ''%s'';\n%s\n'], strrep(outd, '''', ''''''), extra);
    fclose(fid);
end
