function [rirreps, info] = realify_irreps(irreps, G, varargin)
%REALIFY_IRREPS  Frobenius-Schur classification and realification of irreps.
%
%   [RIRREPS, INFO] = REALIFY_IRREPS(IRREPS, G) takes the unitary irreps from
%   DIXON_IRREPS for ANY finite group G (rings, polyhedra, lattices, ...) and,
%   for every irrep that is of real type, returns an equivalent set of matrices
%   that are EXACTLY real (orthogonal).  Irreps that cannot be made real are
%   returned unchanged.
%
%   [RIRREPS, INFO] = REALIFY_IRREPS(IRREPS, G, TOL, SEED) sets the tolerance
%   for declaring residual imaginary parts negligible (default 1e-9) and the
%   RNG seed for reproducibility (default 0).
%
%   OUTPUT
%     RIRREPS : cell array like IRREPS; real-type irreps replaced by real ones.
%     INFO    : struct with fields (1 x nirr)
%                 fs        Frobenius-Schur indicator nu in {+1, 0, -1}
%                 types     'real' (+1) | 'complex' (0) | 'pseudoreal' (-1)
%                 realified logical, true where matrices were made real
%                 maximag   max imaginary residual removed (real type only)
%                 dims      irrep dimensions
%
%   THEORY
%     nu_a = (1/|G|) sum_g chi_a(g^2)  is +1 (real / orthogonal type, can be
%     made real), 0 (complex type, chi not real, cannot be made real), or
%     -1 (pseudoreal / quaternionic, chi real but cannot be made real).
%
%   REALIFICATION (for nu = +1)
%     The conjugate representation is equivalent to D via an intertwiner
%         S = sum_g conj(D(g)) X D(g)'      (random X),   D(g)* = S D(g) S^{-1},
%     which for real type is (proportional to) a symmetric unitary matrix.  A
%     Takagi factorisation  conj(S) = U U^T  (U unitary), obtained from the
%     simultaneously-diagonalisable real/imaginary parts of the symmetric
%     conj(S), yields  B(g) = U' D(g) U  which is real orthogonal.  Residual
%     imaginary parts (~machine epsilon) are then removed with REAL().

    tol = 1e-9;  seed = 0;
    if numel(varargin) >= 1 && ~isempty(varargin{1}), tol  = varargin{1}; end
    if numel(varargin) >= 2 && ~isempty(varargin{2}), seed = varargin{2}; end
    rng(seed, 'twister');

    n    = G.n;
    nirr = numel(irreps);
    g2   = zeros(1, n);
    for g = 1:n, g2(g) = G.multtab(g, g); end

    rirreps   = irreps;
    fs        = zeros(1, nirr);
    types     = cell(1, nirr);
    realified = false(1, nirr);
    maximag   = nan(1, nirr);
    dims      = zeros(1, nirr);

    for a = 1:nirr
        D   = irreps{a};
        d   = size(D{1}, 1);  dims(a) = d;
        chi = cellfun(@trace, D);
        nu  = round(real(sum(chi(g2))) / n);
        fs(a) = nu;
        switch nu
            case  1, types{a} = 'real';
            case -1, types{a} = 'pseudoreal';
            otherwise, types{a} = 'complex';  fs(a) = 0;
        end

        if fs(a) == 1
            [B, mi, ok] = realify_one(D, n, d, tol);
            maximag(a) = mi;
            if ok
                rirreps{a}   = B;
                realified(a) = true;
            else
                warning('realify_irreps:fail', ...
                    'Irrep %d is real type but realification left residual %.2e.', a, mi);
            end
        end
    end

    info = struct('fs', fs, 'types', {types}, 'realified', realified, ...
                  'maximag', maximag, 'dims', dims, 'tol', tol, 'seed', seed);
end

% ============================ local function ============================

function [B, maximag, ok] = realify_one(D, n, d, tol)
%REALIFY_ONE  Make one real-type unitary irrep D exactly real (orthogonal).
    B = D;  maximag = inf;  ok = false;

    for attempt = 1:10
        % intertwiner with the complex-conjugate representation
        X = randn(d) + 1i*randn(d);
        S = zeros(d);
        for g = 1:n
            S = S + conj(D{g}) * X * D{g}';
        end
        if norm(S, 'fro') < 1e-8, continue; end

        % symmetric matrix to Takagi-factor: conj(S) = U U^T
        A   = conj(S);  A = (A + A.') / 2;
        A_R = real(A);  A_I = imag(A);

        % A_R and A_I commute (A is proportional to unitary) -> common real
        % orthonormal eigenbasis from a generic real combination
        c   = 0.37 + 0.11*attempt;
        Msym = A_R + c*A_I;  Msym = (Msym + Msym.') / 2;
        [Q, ~] = eig(Msym);
        Q = real(Q);
        Lam = Q' * A * Q;
        offd = max(abs(Lam(:) - reshape(diag(diag(Lam)), [], 1)));
        if offd > 1e-6 * max(abs(diag(Lam))) + 1e-12
            continue;                       % not simultaneously diagonalised
        end

        ph = diag(Lam);  ph = ph ./ abs(ph);    % unit-modulus phases
        U  = Q * diag(sqrt(ph));                 % unitary, U U^T = conj(S)/scale

        % realify
        Bc = cell(1, n);  mi = 0;
        for g = 1:n
            M = U' * D{g} * U;
            mi = max(mi, max(abs(imag(M(:)))));
            Bc{g} = M;
        end
        maximag = mi;
        if mi < tol
            for g = 1:n, Bc{g} = real(Bc{g}); end   % eliminate machine-eps imag parts
            B = Bc;  ok = true;  return;
        end
    end
end
