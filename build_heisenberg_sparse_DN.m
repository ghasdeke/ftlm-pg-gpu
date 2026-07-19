function H = build_heisenberg_sparse_DN(super_reps, orbit_lens, type_arr, ...
                                         sigma_R_idx_in_C, lam_real, ...
                                         reps_C, L_C, bonds, ...
                                         s_val, J, N, p_irrep, sigma_par, n_total)
%BUILD_HEISENBERG_SPARSE_DN  Sparse H on the sigma-super-rep basis.
%
%   H = BUILD_HEISENBERG_SPARSE_DN(SUPER_REPS, ORBIT_LENS, TYPE_ARR, ...
%       SIGMA_R_IDX_IN_C, LAM_REAL, REPS_C, L_C, BONDS, S_VAL, J, N,
%       P_IRREP, SIGMA_PAR, N_TOTAL)
%
%   builds the Heisenberg Hamiltonian on the symmetry-adapted basis of
%   the (M, p, sigma_par) sector under D_N, for p in {0, N/2}, real H.
%
%   The construction goes via the C_N rep basis: first build the
%   underlying H_C (Hermitian on REPS_C), then form the dense matrix in
%   the sigma-super-rep basis using the explicit symmetrization
%   coefficients
%       |I, sigma_par> = n_I (|reps_C(i_I)> + sigma_par * lam_I |reps_C(partner_I)>)
%   for cross-paired I (n_I = 1/sqrt(2)) and |I, sigma_par> = |reps_C(i_I)>
%   for self-stable I (n_I = 1). The result is sparsified and returned.
%
%   Used for verification and for the ED branch in the driver. Per-sector
%   construction is O(dim_super_rep^2) in matrix density but the
%   underlying H_C is sparse so the total cost is manageable for small
%   sectors; large sectors use the matrix-free GPU kernel instead.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    dim_sup = length(super_reps);
    if dim_sup == 0
        H = sparse([], [], [], 0, 0); return;
    end

    % Underlying H_C on the C_N rep basis at (M, p_irrep).
    H_C = build_heisenberg_sparse_pg(reps_C, L_C, bonds, ...
                                      s_val, J, N, p_irrep, n_total);
    % Hermitize against round-off and densify for the linear combination.
    H_C = 0.5 * (H_C + H_C');
    Hd_C = full(H_C);

    % Index in reps_C of each super-rep (same order; super_reps is a
    % subset of reps_C with preserved order).
    reps_C_to_idx = containers.Map(num2cell(double(reps_C)), num2cell(1:length(reps_C)));
    i_in_C = zeros(dim_sup, 1, 'int32');
    for I = 1 : dim_sup
        i_in_C(I) = int32(reps_C_to_idx(double(super_reps(I))));
    end

    % Build the dense H_DN by summing the 1-, 2-, or 4-term coefficient
    % products (depending on cross/self type of I and J).
    H_DN = zeros(dim_sup, dim_sup);

    for I = 1 : dim_sup
        if type_arr(I) == 1
            j_I = i_in_C(I);
            c_I = 1.0;
            partner_I_idx = -1;
            n_I = 1.0;
        else
            partner_I_idx = sigma_R_idx_in_C(I);
            j_I = i_in_C(I);
            c_I = 1.0;
            % Coefficient of the partner term: sigma_par * lambda_sigma
            c_I_partner = sigma_par * lam_real(I);
            n_I = 1.0 / sqrt(2.0);
        end
        for J = 1 : dim_sup
            if type_arr(J) == 1
                j_J = i_in_C(J);
                c_J = 1.0;
                partner_J_idx = -1;
                n_J = 1.0;
            else
                partner_J_idx = sigma_R_idx_in_C(J);
                j_J = i_in_C(J);
                c_J = 1.0;
                c_J_partner = sigma_par * lam_real(J);
                n_J = 1.0 / sqrt(2.0);
            end

            acc = 0.0;
            % Direct-direct term
            acc = acc + c_I * c_J * Hd_C(j_I, j_J);
            % Cross terms
            if partner_J_idx > 0
                acc = acc + c_I * c_J_partner * Hd_C(j_I, partner_J_idx);
            end
            if partner_I_idx > 0
                acc = acc + c_I * c_J * Hd_C(partner_I_idx, j_J);   % first c_I is from bra; reuse symbol
                % Above used c_I but bra term uses conj(c_I) which equals c_I (real)
                % Redo more carefully:
                acc = -c_I * c_J * Hd_C(j_I, j_J);   % undo the wrong line
                acc = acc + c_I * c_J * Hd_C(j_I, j_J);  % redo correctly
            end

            % Re-derive properly: c_I[1]=1, c_I[2]=sigma_par*lam_I (if cp);
            % c_J similar. Accumulator: sum over (a, b) in {direct, partner}
            % of conj(c_I[a]) * c_J[b] * Hd_C(idx_I[a], idx_J[b]).
            % Since all c's are real and Hd_C is symmetric, conj(c) = c.
            % Rewrite cleanly:
            acc = 0.0;
            if type_arr(I) == 1
                I_list = j_I;
                cI_list = 1.0;
            else
                I_list = [j_I; int32(partner_I_idx)];
                cI_list = [1.0; sigma_par * lam_real(I)];
            end
            if type_arr(J) == 1
                J_list = j_J;
                cJ_list = 1.0;
            else
                J_list = [j_J; int32(partner_J_idx)];
                cJ_list = [1.0; sigma_par * lam_real(J)];
            end
            for ii = 1 : numel(I_list)
                for jj = 1 : numel(J_list)
                    acc = acc + cI_list(ii) * cJ_list(jj) * Hd_C(I_list(ii), J_list(jj));
                end
            end

            H_DN(I, J) = n_I * n_J * acc;
        end
    end

    % Hermitize again as safety, then sparsify.
    H_DN = 0.5 * (H_DN + H_DN');
    H = sparse(H_DN);
end
