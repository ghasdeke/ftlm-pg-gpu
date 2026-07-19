%% demo_square.m -- space group and irreps of the A x B square lattice (torus)
%
% Uses the general, generator-based procedure: square_lattice_group proposes
% candidate symmetry operations, keeps the bond-preserving ones, and closes
% them with group_closure; dixon_irreps then produces all irreducible
% representation matrices (complex in general).

clear; clc;

examples = {[3 3], [4 4], [6 6], [3 4], [4 6]};

fprintf('Square-lattice space groups (periodic A x B), generators only\n\n');
fprintf('%-6s %-15s %4s %4s %5s %7s  %s\n', ...
        'AxB','generators','|G|','|P|','#cls','sum d^2','irrep dims');
store = cell(1, numel(examples));
for e = 1:numel(examples)
    A = examples{e}(1);  B = examples{e}(2);
    [G, info] = square_lattice_group(A, B);
    [ir, di]  = dixon_irreps(G, 1, 1e-10);
    store{e}  = struct('A',A,'B',B,'G',G,'ir',{ir},'di',di);
    fprintf('%-6s %-15s %4d %4d %5d %7d  %s\n', ...
            sprintf('%dx%d',A,B), strjoin(info.gens,','), G.n, G.point_order, ...
            numel(G.classes), sum(di.dims.^2), mat2str(di.dims));
end

%% showcase actual irrep matrices for the 4x4 lattice
s = store{2};  G = s.G;  ir = s.ir;  di = s.di;
fprintf('\n--- Sample irrep matrices, 4x4 lattice (|G| = %d) ---\n', G.n);

txp = G.gens{strcmp(G.gen_labels, 'Tx')};
sxp = G.gens{strcmp(G.gen_labels, 'sx')};
txIdx = find(ismember(G.elements, txp, 'rows'));
sxIdx = find(ismember(G.elements, sxp, 'rows'));

[~, a] = max(di.dims);                      % a maximal-dimensional irrep
fprintf('Irrep #%d, dimension %d\n', a, di.dims(a));
fprintf('D(Tx) =\n');  disp(round(ir{a}{txIdx} * 1e4) / 1e4);
lam = eig(ir{a}{txIdx});
fprintf('eig(D(Tx))  (|.|=1; phases are the lattice momenta k_x):\n');
disp([lam, abs(lam)]);
fprintf('D(sx) =\n');  disp(round(ir{a}{sxIdx} * 1e4) / 1e4);

u = norm(ir{a}{txIdx}' * ir{a}{txIdx} - eye(di.dims(a)));
h = norm(ir{a}{G.multtab(txIdx, sxIdx)} - ir{a}{txIdx} * ir{a}{sxIdx});
fprintf('unitarity ||D(Tx)''D(Tx)-I|| = %.2e\n', u);
fprintf('homomorphism ||D(Tx)D(sx)-D(Tx*sx)|| = %.2e\n', h);
fprintf('(complex entries present: %d)\n', ~isreal(ir{a}{txIdx}));
