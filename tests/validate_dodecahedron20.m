function validate_dodecahedron20()
%VALIDATE_DODECAHEDRON20  Verify the dodecahedron I_h provider + pipeline (N=20).
%
%   The dodecahedron (20 vertices, 30 edges, 3-regular) is the dual of the
%   icosahedron and carries the same I_h symmetry (|G|=120, 10 irreps). A
%   brute-force full ED is infeasible at N=20 (the Sz=0 sector alone is
%   C(20,10)=184756, a 1.8e5^2 dense matrix), so this validates in three
%   layers:
%
%     (1) PROVIDER (pure group theory, no physics): the 120 site
%         permutations form a faithful action that leaves the bond set
%         invariant, is closed consistently with the inherited I_h
%         multiplication table, and the inherited irreps satisfy
%         sum(d^2)=120 and character orthogonality.
%     (2) COMPLETENESS: the symmetry-adapted EXACT run (driver,
%         ed_thresh=inf) over the Sz=0 sector has sum_w = C(20,10), i.e. the
%         (M=0,Gamma) blocks tile the Sz=0 subspace with no states
%         lost/double-counted.
%     (3) PHYSICS: the low-lying Sz=0 spectrum from the symmetry-adapted
%         decomposition matches an INDEPENDENT, symmetry-free sparse-Lanczos
%         diagonalisation of the Sz=0 Heisenberg matrix to ~1e-6. The s=1/2
%         AFM ground state is a singlet (in Sz=0), so this also pins the
%         global ground-state energy E0.
%
%   See also DODECAHEDRON_IH_FULL, ADJACENCY_DODECAHEDRON_IH,
%            ICOSAHEDRON_TRIANGLES, VALIDATE_KAGOME12, VALIDATE_TRIANGULAR12.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    s_val = 0.5;  J = 1.0;
    group = dodecahedron_Ih_full();
    bonds = adjacency_dodecahedron_Ih();
    N = double(group.N);  assert(N == 20);

    %% ---- (1) PROVIDER: group action + irrep checks ------------------------
    fprintf('\n=== (1) provider / group-theory checks ===\n');
    P = double(group.perms);
    % every row is a permutation of 1..20
    for k = 1:group.order
        assert(isequal(sort(P(k,:)), 1:N), 'perms row %d is not a permutation', k);
    end
    % identity acts trivially
    assert(isequal(P(group.identity,:), 1:N), 'identity does not act trivially');
    % bond set invariant under EVERY group element
    bk = @(a,b) min(a,b)*100 + max(a,b);
    bondset = sort(arrayfun(@(e) bk(bonds(e,1),bonds(e,2)), 1:size(bonds,1)));
    for k = 1:group.order
        p = P(k,:);
        img = sort(arrayfun(@(e) bk(p(bonds(e,1)), p(bonds(e,2))), 1:size(bonds,1)));
        assert(isequal(img, bondset), 'bond set not invariant under element %d', k);
    end
    % closure consistent with the inherited multiplication table
    rng(7);
    for t = 1:20
        a = randi(group.order); b = randi(group.order);
        assert(isequal(P(group.mul(a,b),:), P(a,P(b,:))), 'closure broken (%d,%d)', a, b);
    end
    % irreps: sum d^2 = 120 and character orthogonality X' X = 120 I
    irr = {group.Ag, group.Au, group.T1g, group.T1u, group.T2g, ...
           group.T2u, group.Fg, group.Fu, group.Hg, group.Hu};
    ds  = [1 1 3 3 3 3 4 4 5 5];
    assert(sum(ds.^2) == 120, 'sum d^2 != 120');
    chars = zeros(group.order, numel(irr));
    for m = 1:numel(irr)
        A = irr{m};
        if ds(m) == 1
            chars(:,m) = A(:);
        else
            for k = 1:group.order, chars(k,m) = trace(A(:,:,k)); end
        end
    end
    G = chars' * conj(chars);                 % should be 120 * I
    orth_err = max(abs(G(:) - 120*reshape(eye(numel(irr)),[],1)));
    fprintf('  perms OK | bond set invariant under all 120 | closure OK\n');
    fprintf('  sum d^2 = %d | character orthogonality err = %.2e\n', sum(ds.^2), orth_err);
    assert(orth_err < 1e-8, 'character orthogonality failed: %.2e', orth_err);

    %% ---- (2) COMPLETENESS: symmetry-adapted exact run, Sz=0 sum rule ------
    fprintf('\n=== (2) symmetry-adapted exact run (Sz=0, ed_thresh=inf) ===\n');
    r = ftlm_observables_pg_Ih('input_dodecahedron20_ED.m');
    dimM0 = nchoosek(N, N/2);
    sumw_err = abs(r.sum_w - dimM0) / dimM0;
    fprintf('  sum_w = %.8g, expected C(20,10) = %d, rel.err = %.2e\n', ...
        r.sum_w, dimM0, sumw_err);
    assert(sumw_err < 1e-10, 'Sz=0 sum-rule violated: %.2e', sumw_err);

    %% ---- (3) PHYSICS: low spectrum vs independent sparse Lanczos ----------
    fprintf('\n=== (3) independent symmetry-free sparse Lanczos (Sz=0) ===\n');
    k_low = 16;              % enough eigenvalues that >= 5 DISTINCT clusters
                             % survive even when eigs drops degenerate copies
    E_ref = lowest_sz0_sparse(bonds, N, J, k_low);
    assert(~isempty(E_ref), 'eigs returned an empty reference spectrum');
    % Compare the lowest DISTINCT level values rather than position-by-
    % position: eigs('smallestreal') may drop copies inside degenerate
    % clusters (release-dependent ARPACK behavior -- R2024b under-converges
    % cases that R2025b resolves), which misaligns a sorted elementwise
    % comparison even though every returned value is a genuine eigenvalue.
    % The distinct VALUES are robust against missing multiplicities; the
    % multiplicity bookkeeping itself is already pinned exactly by the
    % layer-(2) sum rule.
    [Es, ord] = sort(r.all_E(:));
    ws   = round(r.all_w(ord));
    phys = repelem(Es, ws);
    uq    = @(v) v([true; diff(v(:)) > 1e-6]);   % distinct values of a sorted list
    u_sym = uq(phys(:));
    u_ref = uq(sort(E_ref(:)));
    L = min([5, numel(u_sym), numel(u_ref)]);
    fprintf('  lowest %d distinct levels (sym-adapted vs sparse Lanczos):\n', L);
    for q = 1:L
        fprintf('     %2d   %12.8f   %12.8f   |d|=%.2e\n', ...
            q, u_sym(q), u_ref(q), abs(u_sym(q)-u_ref(q)));
    end
    dE = max(abs(u_sym(1:L) - u_ref(1:L)));
    fprintf('  max |dE| over lowest %d distinct = %.2e\n', L, dE);
    assert(dE < 1e-6, 'low spectrum mismatch: %.2e', dE);

    E0 = u_ref(1);
    fprintf('\n  ground state  E0 = %.8f J   (E0/N = %.6f,  E0/bond = %.6f)\n', ...
        E0, E0/N, E0/size(bonds,1));
    fprintf('  [compare with literature for the s=1/2 dodecahedron Heisenberg AFM]\n');
    fprintf('\n  PASS: dodecahedron I_h provider + pipeline validated (N=20).\n');
end

% ----------------------------------------------------------------
function E = lowest_sz0_sparse(bonds, N, J, k)
% Lowest k eigenvalues of the s=1/2 Heisenberg AFM in the Sz=0 sector,
% built WITHOUT point-group symmetry (independent reference).
%   H = J * sum_<ij> [ Sz_i Sz_j + 1/2 (S+_i S-_j + S-_i S+_j) ].
    nup = N / 2;
    combs = nchoosek(1:N, nup);                  % up-spin sites per basis state
    dim = size(combs, 1);
    states = zeros(dim, 1, 'uint32');
    for c = 1:nup
        states = bitset(states, combs(:, c));    % set bit at site (1-based)
    end
    states = sort(states);
    lut = zeros(2^N, 1, 'int32');
    lut(double(states) + 1) = int32(1:dim);

    diagv = zeros(dim, 1);
    rows = []; cols = []; vals = [];
    for e = 1:size(bonds, 1)
        i = bonds(e, 1);  j = bonds(e, 2);
        bi = double(bitget(states, i));
        bj = double(bitget(states, j));
        diagv = diagv + J * (bi - 0.5) .* (bj - 0.5);    % Sz_i Sz_j
        fl = bi ~= bj;                                    % flippable on this bond
        if any(fl)
            mask = bitset(bitset(uint32(0), i), j);
            sf  = bitxor(states(fl), mask);              % swapped state
            to  = double(lut(double(sf) + 1));
            from = find(fl);
            rows = [rows; from];                 %#ok<AGROW>
            cols = [cols; to];                   %#ok<AGROW>
            vals = [vals; (J/2) * ones(numel(from), 1)]; %#ok<AGROW>
        end
    end
    % both (from->to) and (to->from) are generated by the loop -> already symmetric
    H = sparse(rows, cols, vals, dim, dim) + spdiags(diagv, 0, dim, dim);
    assert(issymmetric(H), 'sparse Sz=0 Hamiltonian is not symmetric');
    % Request a few extra eigenvalues with a large Krylov subspace + tight
    % tolerance: 'smallestreal' under-converges degenerate low clusters at
    % defaults, so resolve k+4 and return the well-converged lowest k.
    E = sort(eigs(H, k + 4, 'smallestreal', ...
        'Tolerance', 1e-12, 'MaxIterations', 3000, 'SubspaceDimension', 150));
    E = E(1:k);
end
