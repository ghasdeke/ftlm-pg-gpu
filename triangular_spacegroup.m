function [group, bonds, pos_cart] = triangular_spacegroup(a, b)
%TRIANGULAR_SPACEGROUP  Space group of a C_6-symmetric triangular (hexagonal
%                       Bravais) lattice torus, as a site-permutation group,
%                       with nearest-neighbour bonds.
%
%   [GROUP, BONDS, POS_CART] = TRIANGULAR_SPACEGROUP(A, B) builds the
%   triangular lattice (one site per unit cell, sites at the Bravais points)
%   on a hexagonal (C_6-symmetric) supercell of the triangular Bravais lattice
%   spanned by
%       T1 = A*a1 + B*a2 ,   T2 = R6 * T1
%   (R6 = 60 deg rotation). The supercell holds |det| = A^2+A*B+B^2 cells,
%   each with ONE site -> N = A^2+A*B+B^2 sites. The full space group is the
%   translations (order = #cells) semidirect the site point group C_6v
%   (order 12, about a LATTICE SITE at the origin = the C_6 rotation centre),
%   total order 12*#cells. Examples: (A,B)=(2,2) -> N=12, |G|=144;
%   (4,0) -> N=16, |G|=192; (6,0) -> N=36, |G|=432.
%
%   This is the 1-site-per-cell sibling of KAGOME_SPACEGROUP (which puts 3
%   sites per cell at lower-symmetry Wyckoff positions). Because the single
%   site sits AT the C_6v Wyckoff position, the site-permutation action can be
%   UNFAITHFUL for very small supercells (e.g. (2,0) -> N=4 would need |G|=48
%   distinct permutations on 4 sites, impossible since 48 > 4!). When that
%   happens the BFS closure returns the FAITHFUL QUOTIENT of the space group;
%   that quotient is still a genuine symmetry group of the Heisenberg
%   Hamiltonian (every returned permutation is a bond automorphism), so the
%   FTLM block decomposition and the sum rule stay exact -- only the amount of
%   symmetry reduction is smaller. The order assert is therefore SOFT (a
%   warning, not an error) so small/degenerate cases still run correctly.
%
%   Returned as the same struct contract as KAGOME_SPACEGROUP /
%   SQUARE_LATTICE_SPACEGROUP / ICOSAHEDRON_IH_FULL (perms, mul, inv,
%   identity, class_idx/size), so IRREPS_FROM_GROUP + the FTLM pipeline
%   consume it unchanged. BONDS is the [3N x 2] NN bond list (triangular
%   coordination z=6 -> 3N bonds). POS_CART is [N x 2] Cartesian positions.
%
%   Convention: work in (a1,a2)-COEFFICIENT coordinates scaled by 2 (so the
%   shared KAGOME helpers carry over verbatim; the single basis site is the
%   origin w=[0 0]). Point-group elements are integer matrices in this basis:
%   R6 = [0 -1; 1 1], sigma = [1 1; 0 -1].
%
%   See also KAGOME_SPACEGROUP, SQUARE_LATTICE_SPACEGROUP, IRREPS_FROM_GROUP,
%            MIN_IMAGE_IH, TEST_TRIANGULAR_SPACEGROUP, VALIDATE_TRIANGULAR12.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    assert(isscalar(a) && isscalar(b) && a==round(a) && b==round(b), 'a,b integer');
    R6  = [0 -1; 1 1];          % C_6 in (a1,a2)-coefficient basis
    sig = [1 1; 0 -1];          % a mirror in (a1,a2)-coefficient basis
    Tm  = [ [a;b], R6*[a;b] ];  % supercell columns T1, T2 (coeff coords)
    ncells = abs(round(det(Tm)));
    N = ncells;                 % ONE site per cell (triangular)
    assert(ncells == a^2 + a*b + b^2, 'cell count mismatch');

    %% Cartesian basis + supercell (for bonds / positions).
    a1 = [1, 0];  a2 = [0.5, sqrt(3)/2];
    A  = [a1.', a2.'];                 % columns; coeff -> Cartesian
    Tcart = A * Tm;                    % Cartesian supercell columns

    %% 1 triangular basis site at the origin (the C_6 centre), in 2*(a1,a2).
    basis_w = [0 0];                   % rows; = 2 * basis coeff (origin)

    %% Enumerate the N sites as integer w = 2*(cell + basis), cells being
    %  representatives of Z^2 / (Tm Z^2). Collect unique sites mod 2*Tm*Z^2.
    rng_box = 2*(abs(a)+abs(b)) + 3;
    site_w = zeros(0, 2);
    for m = -rng_box : rng_box
        for n = -rng_box : rng_box
            for bk = 1 : size(basis_w, 1)
                w = [2*m, 2*n] + basis_w(bk, :);
                if ~any(arrayfun(@(i) in_sublattice(w - site_w(i,:), Tm), 1:size(site_w,1)))
                    site_w(end+1, :) = w; %#ok<AGROW>
                end
            end
        end
    end
    assert(size(site_w,1) == N, 'triangular_spacegroup: found %d sites, expected %d', size(site_w,1), N);

    %% Site Cartesian positions (p = w/2 in coeff -> Cartesian A*(w'/2)).
    pos_cart = (A * (site_w.' / 2)).';    % [N x 2]

    %% Permutation of a symmetry op (point matrix M about origin, + 2*trans tw).
    perm_of = @(M, tw) build_perm(M, tw, site_w, Tm);

    %% Generators: translations a1,a2 (always a torus symmetry) + the point-group
    %  operations (R6, sigma) that actually PRESERVE this supercell lattice. A
    %  point op M is a symmetry of the torus Z^2/(Tm Z^2) iff Tm\(M*Tm) is an
    %  integer matrix (M maps the supercell lattice to itself). R6 always passes
    %  (T2 = R6*T1 by construction) -> at least C_6. The mirror sigma passes only
    %  for NON-CHIRAL cells (b==0 or a==b) -> C_6v; chiral cells (a~=b, both ~=0)
    %  drop it -> C_6. Without this filter an invalid sigma blows the BFS closure
    %  up to a huge non-crystallographic group.
    I2 = eye(2);
    gens = { perm_of(I2,[2 0]), perm_of(I2,[0 2]) };
    for Mc = {R6, sig}
        Msuper = Tm \ (Mc{1} * Tm);
        if all(all(abs(Msuper - round(Msuper)) < 1e-9))
            gens{end+1} = perm_of(Mc{1}, [0 0]); %#ok<AGROW>
        end
    end

    %% BFS closure -> all |G| (faithful) site permutations.
    key_of = @(p) sprintf('%d,', p);
    seen = containers.Map('KeyType','char','ValueType','double');
    perms = 1:N;  seen(key_of(perms)) = 1;  frontier = 1;
    while ~isempty(frontier)
        nxt = [];
        for fi = 1:numel(frontier)
            pa = perms(frontier(fi), :);
            for gi = 1:numel(gens)
                pc = pa(gens{gi}); k = key_of(pc);
                if ~isKey(seen,k)
                    perms(end+1,:) = pc; seen(k)=size(perms,1); nxt(end+1)=size(perms,1); %#ok<AGROW>
                end
            end
        end
        frontier = nxt;
    end
    order = size(perms,1);

    %% SOFT order check: the translation subgroup is always faithful (free), so
    %  order = ncells * (faithful point-group order). For C_6-symmetric cells
    %  the point group closes to C_6v (order 12); small/degenerate supercells
    %  give a faithful quotient (still a valid symmetry group). Warn, do not error.
    assert(mod(order, ncells) == 0, ...
        'triangular_spacegroup: closure order %d not a multiple of ncells %d', order, ncells);
    pg_order = order / ncells;
    switch pg_order
        case 12, point_group = 'C_6v';
        case 6,  point_group = 'C_6';
        case 4,  point_group = 'C_2v';
        case 3,  point_group = 'C_3';
        case 2,  point_group = 'C_2';
        case 1,  point_group = 'C_1';
        otherwise, point_group = sprintf('PG%d', pg_order);
    end
    if pg_order < 12
        warning('triangular_spacegroup:partialPG', ...
            ['site-permutation action faithful only for a %s point group ', ...
             '(|G|=%d, full C_6v would be 12*%d=%d). Still a valid symmetry ', ...
             'group -> sum rule / ED stay exact, reduction is just smaller.'], ...
            point_group, order, ncells, 12*ncells);
    end

    %% mul / inv / classes / identity.
    mul = zeros(order,order,'int32');
    for x=1:order, px=perms(x,:); for y=1:order, mul(x,y)=int32(seen(key_of(px(perms(y,:))))); end, end
    identity = seen(key_of(1:N));
    inv_idx = zeros(order,1,'int32');
    for k=1:order, pk=perms(k,:); pinv=zeros(1,N); pinv(pk)=1:N; inv_idx(k)=int32(seen(key_of(pinv))); end
    assert(all(arrayfun(@(k) mul(k,inv_idx(k))==identity,1:order)), 'inverse sanity');
    assigned=zeros(order,1,'int32'); classes={};
    for i=1:order
        if assigned(i)>0, continue; end
        cls=false(order,1); for h=1:order, cls(mul(h,mul(i,inv_idx(h))))=true; end
        classes{end+1,1}=find(cls); assigned(classes{end})=numel(classes); %#ok<AGROW>
    end

    %% Permutation matrices.
    perm_mats = zeros(N,N,order);
    for k=1:order, for c=1:N, perm_mats(perms(k,c),c,k)=1; end, end

    %% Nearest-neighbour bonds (triangular NN Cartesian distance = 1 under PBC).
    bonds = build_bonds_tri(pos_cart, Tcart, N);

    %% Pack.
    group.N=N; group.order=order; group.a=a; group.b=b; group.ncells=ncells;
    group.point_group=point_group; group.perms=perms; group.perm_mats=perm_mats;
    group.inv=inv_idx; group.mul=mul; group.identity=int32(identity);
    group.class_idx=assigned; group.class_size=int32(cellfun(@numel,classes));
    group.n_class=numel(classes); group.Tcart=Tcart;
end

% ----------------------------------------------------------------
function tf = in_sublattice(dw, Tm)
%IN_SUBLATTICE  is integer dw in 2*Tm*Z^2 ?
    c = (2*Tm) \ dw(:);
    tf = all(abs(c - round(c)) < 1e-6);
end

function p = build_perm(M, tw, site_w, Tm)
    N = size(site_w,1);  p = zeros(1,N);
    for i=1:N
        w = (M * site_w(i,:).').' + tw;      % image (integer)
        j = 0;
        for q=1:N
            if in_sublattice(w - site_w(q,:), Tm), j=q; break; end
        end
        assert(j>0, 'build_perm: image of site %d not matched', i);
        p(i)=j;
    end
    assert(isequal(sort(p),1:N), 'build_perm: not a permutation');
end

function bonds = build_bonds_tri(pos, Tcart, N)
%BUILD_BONDS_TRI  triangular NN bonds: minimum-image Cartesian distance = 1.
    d0 = 1.0;  tol = 1e-6;  bonds = zeros(0,2);
    shifts = [];
    for s1=-1:1, for s2=-1:1, shifts(end+1,:) = (Tcart*[s1;s2]).'; end, end %#ok<AGROW>
    for i=1:N
        for j=i+1:N
            dmin = inf;
            for s=1:size(shifts,1)
                dmin = min(dmin, norm(pos(i,:)-pos(j,:)-shifts(s,:)));
            end
            if abs(dmin - d0) < tol, bonds(end+1,:)=[i,j]; end %#ok<AGROW>
        end
    end
    bonds = sortrows(bonds);
    assert(size(bonds,1) == 3*N, 'build_bonds_tri: %d bonds != 3N=%d (small/degenerate torus?)', ...
        size(bonds,1), 3*N);
end
