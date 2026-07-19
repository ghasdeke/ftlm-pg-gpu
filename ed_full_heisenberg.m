function E_all = ed_full_heisenberg(bonds, N, s_val, J)
%ED_FULL_HEISENBERG  All eigenvalues of the Heisenberg H, no symmetry (small N).
%
%   E_ALL = ED_FULL_HEISENBERG(BONDS, N, S_VAL, J) returns EVERY eigenvalue
%   (with multiplicity) of the isotropic Heisenberg Hamiltonian
%
%       H = J * sum_{<i,j> in BONDS} S_i . S_j
%         = J * sum_{<i,j>} [ Sz_i Sz_j + 1/2 (S+_i S-_j + S-_i S+_j) ]
%
%   on N spin-S_VAL sites, with NO symmetry reduction beyond the total-Sz
%   block structure that the Hamiltonian conserves automatically. This is an
%   INDEPENDENT reference for validating the symmetry-adapted FTLM pipeline
%   (the Hamiltonian convention matches BUILD_HEISENBERG_SPARSE_IH_GAMMA2:
%   diagonal J * Sz_i Sz_j, off-diagonal coefficient 1/2 * J for the s=1/2
%   flip, general-S ladder coefficients otherwise).
%
%   E_ALL has length d_loc^N with d_loc = 2*S_VAL+1, so a flat Boltzmann sum
%   sum_k exp(-beta*E_ALL(k)) is the exact partition function.
%
%   Each total-Sz sector is diagonalised densely. The global spin-flip
%   symmetry (sector A <-> sector Amax-A of the isotropic Heisenberg model)
%   is used to diagonalise only A <= Amax/2 and replicate the spectrum, which
%   halves the work on the largest sectors. Intended for N up to ~ 18.
%
%   See also BUILD_HEISENBERG_SPARSE_IH_GAMMA2, VALIDATE_SQUARE_4X4,
%            FTLM_OBSERVABLES_PG_IH.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    d_loc   = round(2 * s_val + 1);
    n_total = d_loc ^ N;
    assert(n_total <= 2^22, ...
        'ed_full_heisenberg: n_total = %g too large for full ED (small-N reference only).', n_total);

    %% Digit (local quantum number) decomposition of every state.
    states = (0 : n_total - 1)';
    digits = zeros(n_total, N);                 % digits(:,k) in 0..d_loc-1
    tmp = states;
    for k = 1 : N
        digits(:, k) = mod(tmp, d_loc);
        tmp = floor(tmp / d_loc);
    end
    A_state = sum(digits, 2);                    % digit sum (0..Amax)
    Amax    = N * (d_loc - 1);
    mid     = Amax / 2;

    %% Per-Sz-sector dense ED, replicating mirror sectors A <-> Amax-A.
    pw   = d_loc .^ (0 : N - 1);                 % site weights for state index
    E_all = zeros(n_total, 1);
    pos   = 0;
    for A = 0 : floor(mid)
        sec_mask = (A_state == A);
        sec      = states(sec_mask);
        ndim     = numel(sec);
        if ndim == 0, continue; end

        E_sec = sector_spectrum(sec, digits(sec_mask, :), n_total, d_loc, s_val, J, bonds, pw);

        is_mirror = (A < mid - 1e-9);            % A and Amax-A share the spectrum
        reps = 1 + is_mirror;                    % count this spectrum once or twice
        for rr = 1 : reps
            E_all(pos + 1 : pos + ndim) = E_sec;
            pos = pos + ndim;
        end
    end
    assert(pos == n_total, ...
        'ed_full_heisenberg: collected %d of %d eigenvalues', pos, n_total);
    E_all = sort(E_all);
end

% ----------------------------------------------------------------
function E = sector_spectrum(sec, dig, n_total, d_loc, s_val, J, bonds, pw)
%SECTOR_SPECTRUM  Dense eigenvalues of one fixed-Sz Heisenberg block.
    ndim = numel(sec);

    % state -> local index within this sector, as a direct lookup array
    % (sec values are in 0..n_total-1). Vectorised, no per-element loop.
    pos_full          = zeros(n_total, 1);
    pos_full(sec + 1) = 1 : ndim;

    rows = (1 : ndim)';  cols = (1 : ndim)';     % start with diagonal slots

    % Diagonal: J * sum_bonds (d_i - s)(d_j - s).
    diag_acc = zeros(ndim, 1);
    nb = size(bonds, 1);
    for b = 1 : nb
        i = bonds(b, 1);  j = bonds(b, 2);
        diag_acc = diag_acc + J * (dig(:, i) - s_val) .* (dig(:, j) - s_val);
    end
    vals = diag_acc;

    % Off-diagonal: per bond, the "raise i / lower j" move enumerates each
    % connected pair once; add the Hermitian-symmetric pair with value
    % 1/2 * J * <ladder coeffs>.
    sps = s_val * (s_val + 1);
    for b = 1 : nb
        i = bonds(b, 1);  j = bonds(b, 2);
        di = dig(:, i);   dj = dig(:, j);
        mi = di - s_val;  mj = dj - s_val;
        can = (di <= d_loc - 2) & (dj >= 1);     % can raise i, lower j
        if ~any(can), continue; end
        src    = sec(can);
        new_st = src + pw(i) - pw(j);            % i:+1, j:-1
        coeff  = 0.5 * J .* sqrt(sps - mi(can) .* (mi(can) + 1)) ...
                          .* sqrt(sps - mj(can) .* (mj(can) - 1));
        cidx = pos_full(src + 1);                % column = source state
        ridx = pos_full(new_st + 1);             % row    = flipped state
        rows = [rows; ridx; cidx];               %#ok<AGROW>
        cols = [cols; cidx; ridx];               %#ok<AGROW>
        vals = [vals; coeff; coeff];             %#ok<AGROW>
    end

    H = sparse(rows, cols, vals, ndim, ndim);
    H = 0.5 * (H + H');                          % symmetrise away FP noise
    E = sort(real(eig(full(H))));
end
