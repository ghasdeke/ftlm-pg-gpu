function [irreps, info] = dixon_irreps(G, varargin)
%DIXON_IRREPS  Numerical irreducible representations via Dixon's method.
%
%   [IRREPS, INFO] = DIXON_IRREPS(G) computes all inequivalent irreducible
%   *unitary* representations of the finite group G (struct from
%   GROUP_CLOSURE) by decomposing the regular representation.
%
%   [IRREPS, INFO] = DIXON_IRREPS(G, SEED, TOL) sets the RNG seed
%   (default 42, for reproducibility) and the reporting tolerance
%   (default 1e-10).
%
%   OUTPUT
%     IRREPS : 1 x nirr cell.  IRREPS{a} is a 1 x n cell and IRREPS{a}{g}
%              is the d_a x d_a unitary matrix of the g-th group element.
%     INFO   : struct with fields
%                dims       1 x nirr irrep dimensions d_a
%                nirr       number of inequivalent irreps
%                chars      nirr x n characters  chi_a(g) = trace(IRREPS{a}{g})
%                chartable  nirr x nc character table over conjugacy classes
%                classReps  1 x nc class representative indices
%                classSizes 1 x nc class sizes
%                seed, tol  the settings used
%
%   METHOD (Dixon)
%     The regular representation D_reg (|G| x |G|) contains every irrep a with
%     multiplicity d_a.  Averaging a random Hermitian matrix X over the group,
%         H = (1/|G|) * sum_g D(g) * X * D(g)' ,
%     yields a Hermitian element of the commutant.  Its eigenspaces are
%     invariant subspaces; generically each distinct eigenvalue isolates one
%     irreducible copy.  Blocks are split recursively -- resampling X if an
%     accidental eigenvalue collision leaves a reducible block -- until each
%     block satisfies the irreducibility test (1/|G|) sum_g |chi(g)|^2 = 1.
%     Equivalent copies are grouped by character and one unitary representative
%     is returned per irrep.  Complex irreps are handled (X is complex
%     Hermitian, all arithmetic complex).
%
%   Internal Dixon tolerances (eigenvalue clustering / character matching) are
%   looser than TOL (1e-6 relative) because they act on eigensolver output;
%   TOL is the tolerance reported back for the verification suite.

    seed = 42;  tol = 1e-10;
    if numel(varargin) >= 1 && ~isempty(varargin{1}), seed = varargin{1}; end
    if numel(varargin) >= 2 && ~isempty(varargin{2}), tol  = varargin{2}; end

    n = G.n;
    rng(seed, 'twister');

    redtol      = 1.5;     % reducible iff (1/n) sum|chi|^2 >= 1.5  (value is integer)
    chartol     = 1e-6;    % character matching tolerance
    maxattempts = 50;      % resamples before giving up on a block

    % --- left regular representation: D(g) e_h = e_{g*h} ---
    Dreg = cell(1, n);
    for gi = 1:n
        Dreg{gi} = sparse(G.multtab(gi, :), 1:n, 1, n, n);
    end

    % --- recursively split the regular representation ---
    blocks = splitSpace(full(speye(n)));     % cell of n x d_a orthonormal bases

    % --- character of every irreducible block ---
    nb    = numel(blocks);
    bchar = zeros(nb, n);
    for b = 1:nb
        Ub = blocks{b};
        for gi = 1:n
            bchar(b, gi) = trace(Ub' * (Dreg{gi} * Ub));
        end
    end

    % --- collapse equivalent copies: one representative per character class ---
    used      = false(1, nb);
    repBlocks = {};
    for b = 1:nb
        if ~used(b)
            repBlocks{end+1} = blocks{b}; %#ok<AGROW>
            for c = b:nb
                if ~used(c) && norm(bchar(b, :) - bchar(c, :), inf) < chartol
                    used(c) = true;
                end
            end
        end
    end
    nirr = numel(repBlocks);

    % --- unitary irrep matrices from the representatives ---
    irreps = cell(1, nirr);
    dims   = zeros(1, nirr);
    chars  = zeros(nirr, n);
    for a = 1:nirr
        Ua = repBlocks{a};
        da = size(Ua, 2);
        dims(a) = da;
        Da = cell(1, n);
        for gi = 1:n
            Mg = Ua' * (Dreg{gi} * Ua);
            Da{gi}     = Mg;
            chars(a,gi) = trace(Mg);
        end
        irreps{a} = Da;
    end

    % --- canonical order: trivial first, then by dimension, then by character ---
    key       = [dims(:), -real(sum(chars, 2)), real(chars), imag(chars)];
    [~, ord]  = sortrows(key);
    irreps    = irreps(ord);
    dims      = dims(ord);
    chars     = chars(ord, :);

    % --- character table over conjugacy classes ---
    nc        = numel(G.classes);
    chartable = chars(:, G.classReps);   % nirr x nc

    info = struct('dims', dims, 'nirr', nirr, 'chars', chars, ...
                  'chartable', chartable, 'classReps', G.classReps, ...
                  'classSizes', G.classSizes, 'seed', seed, 'tol', tol);

    % ======================= nested helpers =======================
    function blk = splitSpace(U)
        % Split the invariant subspace spanned by the orthonormal columns of U
        % into irreducible pieces; returns a cell of orthonormal n x d_a bases.
        d   = size(U, 2);
        Dr  = cell(1, n);
        chi = zeros(n, 1);
        for g = 1:n
            Dg     = U' * (Dreg{g} * U);
            Dr{g}  = Dg;
            chi(g) = trace(Dg);
        end
        if sum(abs(chi).^2) / n < redtol
            blk = {U};                       % already irreducible
            return;
        end
        for attempt = 1:maxattempts          %#ok<NASGU>
            X = randHerm(d);
            H = zeros(d);
            for g = 1:n
                H = H + Dr{g} * X * Dr{g}';
            end
            H = (H + H') / (2 * n);          % Hermitian element of the commutant
            [V, Dd]   = eig(H);
            [evs, oo] = sort(real(diag(Dd)));
            V         = V(:, oo);
            grps      = clusterEig(evs);
            if numel(grps) > 1
                blk = {};
                for c = 1:numel(grps)
                    Uc  = U * V(:, grps{c});         % back to the n-dim space
                    blk = [blk, splitSpace(Uc)];     %#ok<AGROW>
                end
                return;
            end
        end
        error('dixon_irreps:splitFailed', ...
              'Could not split a reducible block of dimension %d.', d);
    end

    function grps = clusterEig(evs)
        % Group sorted eigenvalues into clusters of (numerically) equal values.
        m      = numel(evs);
        ctol   = max(1e-9, 1e-6 * (evs(end) - evs(1)));
        grps   = {};  st = 1;
        for k = 2:m
            if evs(k) - evs(k-1) > ctol
                grps{end+1} = st:(k-1); %#ok<AGROW>
                st = k;
            end
        end
        grps{end+1} = st:m; %#ok<AGROW>
    end

    function X = randHerm(d)
        A = randn(d) + 1i * randn(d);
        X = (A + A') / 2;
    end
end
