function [perms, Omats] = polyhedron_symmetries(V)
%POLYHEDRON_SYMMETRIES  Point-group symmetries of a vertex set as permutations.
%
%   [PERMS, OMATS] = POLYHEDRON_SYMMETRIES(V) takes the N x 3 vertex
%   coordinates of a (centro)symmetric polyhedron and returns every
%   orthogonal map that permutes the vertex set onto itself:
%     PERMS{k} : 1 x N permutation vector, PERMS{k}(i) = image of vertex i
%     OMATS{k} : the corresponding 3 x 3 orthogonal matrix (det +-1)
%
%   METHOD
%     A symmetry is fixed by the image of a base frame of three linearly
%     independent vertices (vertex 1 and two of its neighbours).  Every image
%     frame with matching mutual distances gives a candidate orthogonal map
%     O = Img / Base; those O that are orthogonal and map the whole vertex set
%     onto itself are kept (deduplicated by the induced permutation).  This
%     returns both proper (rotations) and improper (reflections, inversion)
%     operations, i.e. the full point group.

    n   = size(V, 1);
    tol = 1e-6 * max(abs(V(:)));

    D   = sqrt(sum((reshape(V, n, 1, 3) - reshape(V, 1, n, 3)).^2, 3));
    dd  = D + diag(inf(n, 1));
    el  = min(dd(:));
    adj = abs(D - el) < tol;

    % base frame: vertex 1 and two independent neighbours
    a  = 1;
    nb = find(adj(a, :));
    b  = nb(1);
    c  = [];
    for cc = nb
        if cc ~= b && abs(det([V(a,:); V(b,:); V(cc,:)])) > tol
            c = cc; break;
        end
    end
    if isempty(c)
        for cc = 1:n
            if cc ~= a && cc ~= b && abs(det([V(a,:); V(b,:); V(cc,:)])) > tol
                c = cc; break;
            end
        end
    end
    dbc  = D(b, c);
    Base = [V(a,:); V(b,:); V(c,:)].';

    perms = {};  Omats = {};
    seen  = containers.Map('KeyType', 'char', 'ValueType', 'logical');
    for ap = 1:n
        nbrs = find(adj(ap, :));
        for bp = nbrs
            for cp = nbrs
                if cp == bp || abs(D(bp, cp) - dbc) > tol, continue; end
                Img = [V(ap,:); V(bp,:); V(cp,:)].';
                O   = Img / Base;
                if norm(O.' * O - eye(3)) > 1e-6, continue; end
                W = (O * V.').';
                p = matchPerm(W, V, tol);
                if isempty(p), continue; end
                key = sprintf('%d,', p);
                if ~isKey(seen, key)
                    seen(key)     = true;
                    perms{end+1}  = p;   %#ok<AGROW>
                    Omats{end+1}  = O;   %#ok<AGROW>
                end
            end
        end
    end
end

function p = matchPerm(W, V, tol)
%MATCHPERM  For each transformed point W(k,:) find the matching vertex of V.
    n = size(V, 1);  p = zeros(1, n);  used = false(1, n);
    for k = 1:n
        d = sqrt(sum((V - W(k, :)).^2, 2));
        [mn, idx] = min(d);
        if mn > tol || used(idx), p = []; return; end
        p(k) = idx;  used(idx) = true;
    end
end
