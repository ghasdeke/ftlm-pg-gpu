function Y = spmv_pg_Ih_clt_matlab(clt, X)
%SPMV_PG_IH_CLT_MATLAB  Gather-style SpMV on the I_h (M, Gamma) basis.
%
%   Y = SPMV_PG_IH_CLT_MATLAB(CLT, X)
%
%   computes Y = H_block * X using the precomputed compressed lookup
%   table CLT from BUILD_CLT_PG_IH. This is the matrix-element formula
%   in gather form: each "thread" (loop iteration over the output rep t)
%   reads its precomputed entry list and accumulates contributions
%   without any runtime min-image search or sparse-matrix storage.
%
%   This MATLAB routine is the byte-for-byte CPU reference for the
%   CUDA kernel CUDA_LANCZOS_CLUT_BLOCK_PG_IH.CU; the GPU port will
%   replace this MATLAB scalar/cell-array loop with one CUDA thread per
%   output rep, reading the same flat arrays from device memory and
%   doing the small n_t x n_src GEMM in registers.
%
%   Block-mode: X may be supplied as [n_basis x B] for block-Lanczos.
%
%   Inputs:
%       clt    struct from BUILD_CLT_PG_IH
%       X      [n_basis x B] complex (or real) input block
%
%   Output:
%       Y      same shape as X
%
%   The result must be bit-identical (modulo FP summation order) to
%   SPMV_PG_IH_MATLAB(super_reps, V_per_rep, ..., X). The two-loop
%   ordering differs, so the relative error vs the matrix-free
%   reference is typically at the 1e-13 level for complex Hermitian
%   blocks of a few hundred dimensions in FP64.
%
%   See also BUILD_CLT_PG_IH, SPMV_PG_IH_MATLAB.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    if size(X, 1) ~= clt.n_basis
        error('spmv_pg_Ih_clt_matlab:DimMismatch', ...
            'X has %d rows, expected n_basis = %d.', size(X, 1), clt.n_basis);
    end

    % Promote Y to complex only if the matrix-element tensor is actually
    % complex. BUILD_CLT_* stores a real clt.M for realified (FS=+1) irreps,
    % including realified d>1 space-group irreps, so the gather stays real for
    % real X. Do NOT key this off clt.d_irrep > 1.
    irrep_is_complex = ~isreal(clt.M);
    if irrep_is_complex
        Y = complex(zeros(size(X)));
    else
        Y = zeros(size(X), 'like', X);
    end

    n_reps = clt.n_reps;
    d      = clt.d_irrep;
    for t = 1 : n_reps
        off_t = double(clt.rep_offsets(t));
        n_t   = double(clt.n_per_rep(t));
        idx_t = off_t + 1 : off_t + n_t;

        %% Diagonal S^z * S^z (rep-constant)
        Y(idx_t, :) = Y(idx_t, :) + clt.diag_vals(t) * X(idx_t, :);

        %% Off-diagonal: gather from CLT entries.
        %  clt.M is now a [d x d x n_entries] padded tensor; we slice the
        %  active n_t x n_s top-left block per entry. The unused rows /
        %  cols are zero by construction so a full d x d multiplication
        %  would also work, but slicing saves a few flops.
        e_off = double(clt.entry_offsets(t));
        n_e   = double(clt.entries_per_rep(t));
        for k = 1 : n_e
            e = e_off + k;
            src = double(clt.src_idx(e));
            off_s = double(clt.rep_offsets(src));
            n_s   = double(clt.n_per_rep(src));
            idx_s = off_s + 1 : off_s + n_s;
            M_e   = clt.M(1:n_t, 1:n_s, e);     % active sub-block
            Y(idx_t, :) = Y(idx_t, :) + M_e * X(idx_s, :);
        end
    end
end
