function H = heisenberg_ring(N, varargin)
%HEISENBERG_RING  Spin-1/2 Heisenberg Hamiltonian on a ring (sparse).
%
%   H = HEISENBERG_RING(N) returns the sparse 2^N x 2^N matrix of
%       H = sum_<i,j> S_i . S_j
%   for the periodic chain with nearest-neighbour bonds (i,i+1) and (N,1)
%   and coupling J = 1, with S the spin-1/2 operators.
%
%   H = HEISENBERG_RING(N, BONDS, J) uses an explicit P x 2 bond list and
%   coupling J instead.
%
%   Basis convention (matching SALC_PROJECTORS): state index b (0-based)
%   stores the spin on site i in bit i (value 2^(i-1)); bit = 1 is up
%   (Sz = +1/2), bit = 0 is down (Sz = -1/2).
%
%   With  S_i.S_j = Sz_i Sz_j + (1/2)(S+_i S-_j + S-_i S+_j), the diagonal
%   part is (J/4) sigma_i sigma_j and each antiparallel bond contributes an
%   off-diagonal J/2 connecting the two spin-flipped states.

    bonds = [(1:N).', [2:N, 1].'];
    J     = 1;
    if numel(varargin) >= 1 && ~isempty(varargin{1}), bonds = varargin{1}; end
    if numel(varargin) >= 2 && ~isempty(varargin{2}), J     = varargin{2}; end

    M      = 2^N;
    states = (0:M-1).';

    % --- diagonal:  (J/4) sum_<ij> sigma_i sigma_j ---
    diagH = zeros(M, 1);
    for b = 1:size(bonds, 1)
        i = bonds(b, 1);  j = bonds(b, 2);
        si = 2 * bitget(states, i) - 1;       % +/-1
        sj = 2 * bitget(states, j) - 1;
        diagH = diagH + J * 0.25 * (si .* sj);
    end
    Ii = (1:M).';  Jj = (1:M).';  Vv = diagH;

    % --- off-diagonal flips on antiparallel bonds:  J/2 ---
    for b = 1:size(bonds, 1)
        i = bonds(b, 1);  j = bonds(b, 2);
        anti    = bitget(states, i) ~= bitget(states, j);
        src     = states(anti);
        mask    = 2^(i-1) + 2^(j-1);
        flipped = bitxor(src, mask);
        Ii = [Ii; flipped + 1];               %#ok<AGROW>
        Jj = [Jj; src + 1];                   %#ok<AGROW>
        Vv = [Vv; J * 0.5 * ones(numel(src), 1)]; %#ok<AGROW>
    end

    H = sparse(Ii, Jj, Vv, M, M);
    H = (H + H') / 2;                          % symmetrise (cleans rounding)
end
