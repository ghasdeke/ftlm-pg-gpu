function [G, info] = square_lattice_group(A, B)
%SQUARE_LATTICE_GROUP  Space group of an A x B square lattice (torus) as permutations.
%
%   [G, INFO] = SQUARE_LATTICE_GROUP(A, B) builds the full site-permutation
%   symmetry group of the periodic (toroidal) A x B square lattice and returns
%   it via GROUP_CLOSURE.  Sites are (x,y), x = 0..A-1, y = 0..B-1, flattened to
%   index  idx = x + A*y + 1  (N = A*B sites).
%
%   The "general procedure" only proposes candidate generators and keeps the
%   ones that are genuine site permutations preserving the nearest-neighbour
%   bond set:
%       translations  Tx, Ty
%       reflections   sx (x->-x), sy (y->-y)
%       rotation      r4 (90 degrees)        [valid only if A == B]
%       diagonal      sd (x<->y)             [valid only if A == B]
%   This automatically yields the point group D4 for A == B (|G| = 8*A^2) and
%   D2 for A ~= B (|G| = 4*A*B), and also copes with small-lattice coincidences
%   (e.g. reflections that collapse to the identity for A = 2).
%
%   OUTPUT
%     G    : struct from GROUP_CLOSURE, with extra fields
%              G.A, G.B          lattice dimensions
%              G.bonds           P x 2 nearest-neighbour bond list
%              G.gen_labels      labels of the surviving generators
%              G.point_order     |G| / (A*B)  (realised point-group order)
%     INFO : struct with A, B, nbonds, gens (labels), order, point_order
%
%   The returned group plugs directly into DIXON_IRREPS to obtain all
%   (in general complex) irreducible representation matrices.

    N = A * B;
    bonds = build_bonds(A, B);

    cand = { make_perm(A, B, @(x,y) [mod(x+1,A), y]), ...   % Tx
             make_perm(A, B, @(x,y) [x, mod(y+1,B)]), ...   % Ty
             make_perm(A, B, @(x,y) [mod(-x,A), y]),  ...   % sx
             make_perm(A, B, @(x,y) [x, mod(-y,B)]),  ...   % sy
             make_perm(A, B, @(x,y) [mod(-y,A), x]),  ...   % r4 (90 deg)
             make_perm(A, B, @(x,y) [y, x]) };              % sd (diagonal)
    labels = {'Tx','Ty','sx','sy','r4','sd'};

    gens = {};  glab = {};
    for c = 1:numel(cand)
        p = cand{c};
        if is_perm(p, N) && ~isequal(p, 1:N) && preserves_bonds(p, bonds)
            gens{end+1}  = p;          %#ok<AGROW>
            glab{end+1}  = labels{c};  %#ok<AGROW>
        end
    end

    G = group_closure(gens, N);
    G.A = A;  G.B = B;  G.bonds = bonds;
    G.gen_labels = glab;  G.point_order = G.n / (A*B);

    info = struct('A', A, 'B', B, 'nbonds', size(bonds,1), ...
                  'gens', {glab}, 'order', G.n, 'point_order', G.point_order);
end

% ============================ local functions ============================

function p = make_perm(A, B, f)
%MAKE_PERM  Turn an affine site map f:(x,y)->[x' y'] into a permutation vector.
%   Out-of-range images are left as invalid entries; IS_PERM rejects them.
    N = A*B;  p = zeros(1, N);
    for y = 0:B-1
        for x = 0:A-1
            v = f(x, y);
            p(x + A*y + 1) = v(1) + A*v(2) + 1;
        end
    end
end

function bonds = build_bonds(A, B)
%BUILD_BONDS  Nearest-neighbour bonds of the periodic A x B square lattice.
    rows = zeros(0, 2);
    for y = 0:B-1
        for x = 0:A-1
            s  = x + A*y + 1;
            sr = mod(x+1, A) + A*y + 1;        % right neighbour
            su = x + A*mod(y+1, B) + 1;        % up neighbour
            if s ~= sr, rows(end+1, :) = sort([s sr]); end %#ok<AGROW>
            if s ~= su, rows(end+1, :) = sort([s su]); end %#ok<AGROW>
        end
    end
    bonds = unique(rows, 'rows');
end

function ok = is_perm(p, N)
%IS_PERM  True if p is a permutation of 1..N.
    ok = numel(p) == N && all(p >= 1 & p <= N) && numel(unique(p)) == N;
end

function ok = preserves_bonds(p, bonds)
%PRESERVES_BONDS  True if the permutation p maps the bond set onto itself.
    pp = p(:);
    im = sort([pp(bonds(:,1)), pp(bonds(:,2))], 2);
    ok = isequal(sortrows(im), sortrows(bonds));
end
