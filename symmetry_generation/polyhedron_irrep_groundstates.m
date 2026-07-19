function out = polyhedron_irrep_groundstates(name, opts)
%POLYHEDRON_IRREP_GROUNDSTATES  M=0 ground energy per I_h irrep, s=1/2 Heisenberg.
%
%   OUT = POLYHEDRON_IRREP_GROUNDSTATES(NAME) builds the spin-1/2 Heisenberg
%   antiferromagnet H = sum_<i,j> S_i.S_j on the 'icosahedron' (12 sites) or
%   'dodecahedron' (20 sites), restricts to the total-S_z = 0 sector, and
%   returns the ground-state energy in that sector projected onto each of the
%   10 irreducible representations of the full icosahedral group I_h.
%
%   OUT = POLYHEDRON_IRREP_GROUNDSTATES(NAME, OPTS) accepts options:
%     OPTS.seed   RNG seed for Dixon       (default 1)
%     OPTS.k      eigenpairs per sector    (default 1)
%     OPTS.tol    Dixon/report tolerance   (default 1e-10)
%     OPTS.verbose print a table           (default true)
%
%   OUTPUT struct OUT
%     OUT.name, OUT.N, OUT.nM (= C(N,N/2))
%     OUT.G, OUT.info        group struct and Dixon info (char table etc.)
%     OUT.R                  1 x 10 struct array per irrep with fields
%        label   physical label (A_g, T1g, ..., H_u)
%        dim     irrep dimension
%        parity  +1 (gerade) / -1 (ungerade)
%        c5char  character on a C5 rotation (distinguishes T1/T2)
%        secdim  dimension of the irrep's isotypic component in the M=0 sector
%        E0      ground energy in that sector (via eigs)
%        E0_exact dense cross-check (only for small sectors, else NaN)
%     OUT.global_ground  min over irreps (= overall M=0 ground energy)
%
%   METHOD
%     The total S_z and every I_h site permutation commute with H, so the M=0
%     sector is invariant.  For irrep a the isotypic projector
%        P_a = (d_a/|G|) sum_g chi_a(g) D_{M0}(g)
%     is built as a sparse operator on the M=0 sector.  The sector ground
%     energy is the smallest eigenvalue of H on range(P_a).  To make this an
%     EIGS problem that converges robustly, the dominant eigenpair of the
%     shifted, projected operator
%        M_a v = P_a (c I - H) P_a v ,   c > ||H||
%     is computed with 'largestabs': its largest eigenvalue is  c - E_ground,
%     so  E_ground = c - lambda_max.  The huge P_a null space maps to 0 and
%     never dominates the Krylov space.  No explicit symmetry-adapted basis of
%     the (up to 184756-dimensional) sector is ever formed.

    if nargin < 2, opts = struct(); end
    def = struct('seed', 1, 'k', 1, 'tol', 1e-10, 'verbose', true);
    f = fieldnames(def);
    for i = 1:numel(f)
        if ~isfield(opts, f{i}), opts.(f{i}) = def.(f{i}); end
    end
    phi = (1 + sqrt(5)) / 2;

    % ---- geometry, symmetry group, irreps ----
    model = polyhedron_model(name);
    N = model.N;  bonds = model.bonds;

    [perms, Om] = polyhedron_symmetries(model.V);
    gens = {};
    for k = 1:numel(perms)
        if ~isequal(perms{k}, 1:N), gens = {perms{k}}; break; end
    end
    G = group_closure(gens, N);
    for k = 1:numel(perms)
        if G.n == numel(perms), break; end
        if ~ismember(perms{k}, G.elements, 'rows')
            gens{end+1} = perms{k}; %#ok<AGROW>
            G = group_closure(gens, N);
        end
    end
    [irreps, info] = dixon_irreps(G, opts.seed, opts.tol); %#ok<ASGLU>
    n = G.n;  nirr = info.nirr;  dims = info.dims;

    % ---- identify inversion (O=-I) and a true C5 (det +1, trace=phi) ----
    keymap = containers.Map('KeyType', 'char', 'ValueType', 'double');
    for m = 1:numel(perms), keymap(sprintf('%d,', perms{m})) = m; end
    Oel = cell(1, n);
    for e = 1:n, Oel{e} = Om{ keymap(sprintf('%d,', G.elements(e, :))) }; end
    invIdx = find(cellfun(@(O) norm(O + eye(3)) < 1e-6, Oel), 1);
    c5Idx  = find(cellfun(@(O) abs(det(O) - 1) < 1e-6 && ...
                                abs(trace(O) - phi) < 1e-4, Oel), 1);

    % ---- M=0 sector ----
    [states, pos, nM] = build_m0(N);
    H = heisenberg_m0(bonds, states, pos);

    % spin-permutation row indices for all g (used to assemble projectors)
    rows = zeros(n * nM, 1);
    cols = repmat((1:nM)', n, 1);
    for g = 1:n
        rows((g-1)*nM + (1:nM)) = m0_perm(G.elements(g, :), states, pos, N);
    end

    doExact = nM <= 3000;            % dense cross-check only for the small sector
    cShift  = 0.75 * size(bonds, 1) + 1;   % c > ||H|| (each bond: |S.S| <= 3/4)

    R = struct('label', {}, 'dim', {}, 'parity', {}, 'c5char', {}, ...
               'secdim', {}, 'E0', {}, 'E0_exact', {});
    for a = 1:nirr
        chi  = real(info.chars(a, :));
        vals = zeros(n * nM, 1);
        for g = 1:n, vals((g-1)*nM + (1:nM)) = chi(g); end
        Pa = (dims(a) / n) * sparse(rows, cols, vals, nM, nM);
        secdim = round(full(sum(diag(Pa))));

        par = sign(real(info.chars(a, invIdx)));
        c5c = real(info.chars(a, c5Idx));
        lab = irrep_label(dims(a), par, c5c);

        E0 = NaN;  E0e = NaN;
        if secdim > 0
            % dominant eigenpair of M_a = P_a (cI - H) P_a  ->  E0 = c - lambda_max
            Mfun = @(v) cShift * (Pa * v) - Pa * (H * (Pa * v));
            kk   = min(opts.k, secdim);
            lam  = eigs(Mfun, nM, kk, 'largestabs', ...
                        'IsFunctionSymmetric', true, ...
                        'Tolerance', 1e-11, 'MaxIterations', 3000);
            E0   = cShift - max(real(lam));
            if doExact
                % orthonormal basis of range(P_a); (eig of a only-almost-
                % Hermitian sparse Pa uses the non-symmetric solver and would
                % return non-orthonormal vectors -> use orth instead)
                B   = orth(full(Pa));
                blk = B' * H * B;  blk = (blk + blk') / 2;
                E0e = min(real(eig(blk)));
            end
        end
        R(a) = struct('label', lab, 'dim', dims(a), 'parity', par, ...
                      'c5char', c5c, 'secdim', secdim, 'E0', E0, 'E0_exact', E0e);
    end

    % independent check: overall M=0 ground state without any symmetry
    Hfun  = @(v) cShift * v - H * v;             % dominant eig -> c - E_ground
    lamG  = eigs(Hfun, nM, 1, 'largestabs', 'IsFunctionSymmetric', true, ...
                 'Tolerance', 1e-11, 'MaxIterations', 3000);
    Edir  = cShift - max(real(lamG));

    out = struct('name', lower(name), 'N', N, 'nM', nM, 'G', G, ...
                 'info', info, 'R', R, 'global_ground', min([R.E0]), ...
                 'global_direct', Edir);
    if opts.verbose, print_results(out); end
end

% ============================ local functions ============================

function [states, pos, nM] = build_m0(N)
%BUILD_M0  Integer labels of all S_z=0 product states (N/2 up spins) + lookup.
    combs  = nchoosek(1:N, N/2);
    states = sort(sum(2.^(combs - 1), 2));
    nM     = numel(states);
    pos    = zeros(2^N, 1);
    pos(states + 1) = (1:nM)';
end

function H = heisenberg_m0(bonds, states, pos)
%HEISENBERG_M0  Sparse Heisenberg Hamiltonian restricted to the M=0 sector.
    nM = numel(states);
    diagH = zeros(nM, 1);
    for b = 1:size(bonds, 1)
        i = bonds(b, 1);  j = bonds(b, 2);
        diagH = diagH + 0.25 * ((2*bitget(states, i) - 1) .* (2*bitget(states, j) - 1));
    end
    Ic = {(1:nM)'};  Jc = {(1:nM)'};  Vc = {diagH};
    for b = 1:size(bonds, 1)
        i = bonds(b, 1);  j = bonds(b, 2);
        anti = bitget(states, i) ~= bitget(states, j);
        src  = states(anti);
        flp  = bitxor(src, 2^(i-1) + 2^(j-1));
        Ic{end+1} = pos(flp + 1);  Jc{end+1} = pos(src + 1); %#ok<AGROW>
        Vc{end+1} = 0.5 * ones(numel(src), 1);               %#ok<AGROW>
    end
    H = sparse(cell2mat(Ic'), cell2mat(Jc'), cell2mat(Vc'), nM, nM);
    H = (H + H') / 2;
end

function p = m0_perm(g, states, pos, N)
%M0_PERM  Index map of the site permutation g acting on the M=0 states.
%   Applying D(g) to a vector v is then  w(p) = v.
    target = zeros(numel(states), 1);
    for i = 1:N
        target = target + bitget(states, i) * 2^(g(i) - 1);
    end
    p = pos(target + 1);
end

function lab = irrep_label(d, par, c5c)
%IRREP_LABEL  Physical I_h label from dimension, parity and C5 character.
    if par > 0, suf = 'g'; else, suf = 'u'; end
    switch d
        case 1, base = 'A';
        case 3, if c5c > 0.5, base = 'T1'; else, base = 'T2'; end
        case 4, base = 'G';
        case 5, base = 'H';
        otherwise, base = sprintf('d%d', d);
    end
    lab = [base suf];
end

function print_results(out)
%PRINT_RESULTS  Tabulate the per-irrep M=0 ground energies.
    hasExact = ~isnan(out.R(1).E0_exact);
    fprintf('\n===== %s : I_h, |G|=%d, M=0 sector dim = %d =====\n', ...
            upper(out.name), out.G.n, out.nM);
    if hasExact
        fprintf('%-5s %3s %4s %7s %16s %16s\n', ...
                'irrep','dim','par','secdim','E0 (eigs)','E0 (exact)');
    else
        fprintf('%-5s %3s %4s %7s %16s\n', 'irrep','dim','par','secdim','E0 (eigs)');
    end
    for a = 1:numel(out.R)
        r = out.R(a);
        if r.parity > 0, ps = 'g'; else, ps = 'u'; end
        if isnan(r.E0)
            e0s = '        ---     ';
        else
            e0s = sprintf('%16.8f', r.E0);
        end
        fprintf('%-5s %3d %4s %7d %s', r.label, r.dim, ps, r.secdim, e0s);
        if hasExact
            if isnan(r.E0_exact), fprintf(' %16s', '---');
            else, fprintf(' %16.8f', r.E0_exact); end
        end
        fprintf('\n');
    end
    fprintf('-----------------------------------------------------------\n');
    [~, im] = min([out.R.E0]);
    fprintf('Global M=0 ground (min over irreps): %.8f   in irrep %s\n', ...
            out.global_ground, out.R(im).label);
    fprintf('Global M=0 ground (direct eigs, no symmetry): %.8f\n', out.global_direct);
    fprintf('  (agreement: %.2e)\n', abs(out.global_ground - out.global_direct));
    fprintf('Sum of sector dims = %d  (should be C(N,N/2) = %d)\n', ...
            sum([out.R.secdim]), out.nM);
end
