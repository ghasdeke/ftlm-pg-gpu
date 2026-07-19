function S = salc_projectors(irreps, G, N, varargin)
%SALC_PROJECTORS  Symmetry-adapted linear combinations of spin-1/2 states.
%
%   S = SALC_PROJECTORS(IRREPS, G, N) builds, for the permutation
%   representation of the group G on the 2^N-dimensional spin-1/2 product
%   basis, the isotypic projectors, the transfer (shift) operators and an
%   orthonormal SALC basis organised by irrep and partner index.
%
%   S = SALC_PROJECTORS(IRREPS, G, N, TOL) sets the numerical tolerance
%   (default 1e-10; used only for reporting / stored in S.tol).
%
%   INPUT
%     IRREPS : cell array of unitary irreps from DIXON_IRREPS.
%     G      : group struct from GROUP_CLOSURE.
%     N      : number of sites; the basis dimension is M = 2^N.
%
%   OUTPUT (struct S)
%     S.M         basis dimension 2^N
%     S.Dspin     1 x n cell of sparse M x M permutation matrices D(g)
%     S.dims      1 x nirr irrep dimensions d_a
%     S.mult      1 x nirr multiplicities m_a in the spin representation
%     S.sectordim 1 x nirr isotypic sector dimensions d_a * m_a
%     S.Piso      1 x nirr cell of sparse isotypic projectors P_a
%     S.U         M x M matrix whose columns are the orthonormal SALCs,
%                 ordered by (irrep a, partner i, multiplicity copy k)
%     S.blocks    struct array (one per (a,i) block) with fields
%                 alpha, i, mult, cols
%     S.blockId   1 x M block index of each SALC column
%
%   METHOD
%     A site permutation g acts on a product state |s_1...s_N> by
%         (g.s)(i) = s(g^{-1} i),
%     i.e. the spin on site i moves to site g(i); this gives a sparse
%     permutation matrix D(g).  Then
%         isotypic projector   P_a      = (d_a/|G|) sum_g conj(chi_a(g)) D(g)
%         transfer operator    P^a_{ij} = (d_a/|G|) sum_g conj(D^a_{ij}(g)) D(g)
%     The range of the Hermitian projector P^a_{11} provides the m_a
%     first-partner SALCs; the other partners follow from
%         |a,i,k> = P^a_{i1} |a,1,k> ,
%     which is automatically orthonormal (P^a_{ij} P^a_{kl} = d_jk P^a_{il}).
%     In this basis every operator commuting with all D(g) is block diagonal
%     in (a,i).  Complex matrix elements / SALCs are fully supported.

    tol = 1e-10;
    if ~isempty(varargin) && ~isempty(varargin{1}), tol = varargin{1}; end

    n    = G.n;
    M    = 2^N;
    nirr = numel(irreps);
    dims = cellfun(@(c) size(c{1}, 1), irreps);

    % --- characters chi_a(g) = trace D^a(g) ---
    chars = zeros(nirr, n);
    for a = 1:nirr
        for g = 1:n
            chars(a, g) = trace(irreps{a}{g});
        end
    end

    % --- spin-1/2 permutation representation D(g), sparse ---
    %     bit i (value 2^(i-1)) holds the spin on site i; site i -> site g(i)
    Dspin  = cell(1, n);
    states = 0:M-1;
    for gi = 1:n
        g      = G.elements(gi, :);
        target = zeros(1, M);
        for i = 1:N
            bit_i  = bitget(states, i);
            target = target + bit_i * 2^(g(i) - 1);
        end
        Dspin{gi} = sparse(target + 1, 1:M, 1, M, M);
    end

    % --- isotypic projectors P_a ---
    Piso = cell(1, nirr);
    for a = 1:nirr
        P = sparse(M, M);
        for g = 1:n
            c = conj(chars(a, g));
            if c ~= 0
                P = P + c * Dspin{g};
            end
        end
        Piso{a} = (dims(a) / n) * P;
    end

    % --- transfer operators and SALC basis ---
    parts     = {};
    blocks    = struct('alpha', {}, 'i', {}, 'mult', {}, 'cols', {});
    mult      = zeros(1, nirr);
    sectordim = zeros(1, nirr);
    colcount  = 0;
    bcount    = 0;

    for a = 1:nirr
        da  = dims(a);
        P11 = transferOp(a, 1, 1);
        P11 = (P11 + P11') / 2;               % enforce Hermitian
        [V, D] = eig(full(P11));
        ev  = real(diag(D));
        Q   = V(:, ev > 0.5);                 % projector eigenvalues are 0 or 1
        ma  = size(Q, 2);
        mult(a)      = ma;
        sectordim(a) = da * ma;
        if ma == 0, continue; end
        for i = 1:da
            if i == 1
                part = Q;
            else
                part = transferOp(a, i, 1) * Q;
            end
            bcount        = bcount + 1;
            cols          = colcount + (1:ma);
            colcount      = colcount + ma;
            parts{end+1}  = part; %#ok<AGROW>
            blocks(bcount) = struct('alpha', a, 'i', i, 'mult', ma, 'cols', cols);
        end
    end

    U       = [parts{:}];
    blockId = zeros(1, size(U, 2));
    for b = 1:numel(blocks)
        blockId(blocks(b).cols) = b;
    end

    S = struct();
    S.M = M;  S.Dspin = Dspin;  S.dims = dims;  S.mult = mult;
    S.sectordim = sectordim;  S.Piso = Piso;  S.U = U;
    S.blocks = blocks;  S.blockId = blockId;  S.tol = tol;

    % ======================= nested helper =======================
    function T = transferOp(a, i, j)
        T = sparse(M, M);
        for g = 1:n
            c = conj(irreps{a}{g}(i, j));
            if c ~= 0
                T = T + c * Dspin{g};
            end
        end
        T = (dims(a) / n) * T;
    end
end
