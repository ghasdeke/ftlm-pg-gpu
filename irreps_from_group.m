function irreps = irreps_from_group(group, max_tries)
%IRREPS_FROM_GROUP  All irreducible representations of a finite group, as
%                   explicit unitary matrices, from its multiplication table.
%
%   IRREPS = IRREPS_FROM_GROUP(GROUP) returns a 1 x n_irrep struct array
%   (the generic interface the FTLM pipeline consumes), each with fields
%       .name  char  e.g. 'irr3' (d=3, the 3rd found)
%       .d     irrep dimension
%       .mats  [d x d x order] complex unitary matrices rho(g), satisfying
%              rho(a)*rho(b) = rho(mul(a,b)) exactly (to numerical precision).
%
%   Method (decomposition of the regular representation via a random
%   commutant element). The left regular representation L(g) (L(g)e_h=e_{gh})
%   decomposes as (+)_Gamma d_Gamma copies of each irrep. A generic Hermitian
%   element C = sum_h a_h R(h) of the RIGHT regular algebra (which commutes
%   with every L(g)) acts within each isotypic block as I_{d} (x) H, with H a
%   generic d x d Hermitian whose d distinct eigenvalues split the d copies.
%   Hence each eigenspace of C is exactly one IRREDUCIBLE L-invariant
%   subspace: with U its orthonormal basis, rho(g) = U' * L(g) * U is a
%   unitary irrep (homomorphism guaranteed because L is and U spans an
%   invariant subspace). Equivalent copies are removed by character. A
%   complex C separates complex-conjugate irrep pairs (e.g. C_4 momenta).
%
%   GROUP needs fields: order, mul [order x order], inv [order x 1],
%   identity. Optional group.n_class is used as a completeness check.
%
%   See also SQUARE_LATTICE_SPACEGROUP, ICOSAHEDRON_IH_FULL,
%            FTLM_OBSERVABLES_PG_GPU_IH (build_full_irrep_table).

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    if nargin < 2, max_tries = 12; end
    n   = double(group.order);
    mul = double(group.mul);
    inv = double(group.inv);
    if isfield(group, 'n_class'), n_class = group.n_class; else, n_class = []; end

    for attempt = 1 : max_tries
        rng(1000 + attempt);                 % deterministic, varied per retry

        %% Random Hermitian element C = sum_h a_h R(h) of the right-regular
        %  algebra. R(h)[mul(x,h), x] = 1. Hermitian <=> a_h = conj(a_{h^-1}).
        a = randn(n, 1) + 1i * randn(n, 1);
        a = 0.5 * (a + conj(a(inv)));
        xs = (1:n)';
        rows = zeros(n*n, 1); cols = zeros(n*n, 1); val = complex(zeros(n*n, 1));
        p = 0;
        for h = 1 : n
            rows(p+1:p+n) = mul(xs, h);
            cols(p+1:p+n) = xs;
            val(p+1:p+n)  = a(h);
            p = p + n;
        end
        C = full(sparse(rows, cols, val, n, n));
        C = 0.5 * (C + C');                   % kill FP non-Hermiticity

        [V, D] = eig(C);
        ev = real(diag(D));
        [ev, ord] = sort(ev);
        V = V(:, ord);

        %% Cluster eigenvalues: within-irrep copies are exactly degenerate
        %  (~eps), distinct clusters are O(range/#distinct) apart.
        rng_ev = max(ev) - min(ev) + eps;
        gaps   = diff(ev);
        breaks = [0; find(gaps > 1e-6 * rng_ev); n];

        reps  = {};      % rho tensors
        chis  = {};      % character vectors (for dedup)
        ok    = true;
        for c = 1 : numel(breaks) - 1
            cols_c = breaks(c)+1 : breaks(c+1);
            U = V(:, cols_c);
            d = numel(cols_c);
            % character first (cheap), to dedup before building full rho
            chi = zeros(n, 1);
            for g = 1 : n
                rp = mul(inv(g), xs);
                chi(g) = sum(sum(conj(U) .* U(rp, :)));
            end
            % irreducible? sum_g |chi(g)|^2 == |G|
            if abs(sum(abs(chi).^2) - n) > 1e-3 * n
                ok = false; break;           % bad cluster -> re-roll
            end
            % new irrep?
            isnew = true;
            for e = 1 : numel(chis)
                if norm(chi - chis{e}) < 1e-5 * sqrt(n), isnew = false; break; end
            end
            if ~isnew, continue; end
            rho = complex(zeros(d, d, n));
            for g = 1 : n
                rp = mul(inv(g), xs);
                rho(:, :, g) = U' * U(rp, :);
            end
            reps{end+1} = rho;  chis{end+1} = chi;  %#ok<AGROW>
        end
        if ~ok, continue; end

        %% Completeness checks.
        dims = cellfun(@(r) size(r, 1), reps);
        if sum(dims.^2) ~= n, continue; end
        if ~isempty(n_class) && numel(reps) ~= n_class, continue; end

        %% Pack, sorted by dimension then discovery order.
        [~, sidx] = sortrows([dims(:), (1:numel(reps))']);
        irreps = struct('name', {}, 'd', {}, 'mats', {});
        for j = 1 : numel(sidx)
            r = reps{sidx(j)};
            irreps(j).name = sprintf('d%d_%d', size(r,1), j);
            irreps(j).d    = size(r, 1);
            irreps(j).mats = r;
        end
        return;
    end
    error('irreps_from_group:failed', ...
        'Could not extract a complete irrep set in %d tries (sum d^2 / count mismatch).', max_tries);
end
