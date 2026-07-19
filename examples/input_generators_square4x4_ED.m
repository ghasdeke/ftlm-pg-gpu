% 4x4 square lattice, s=1/2, via the GENERIC 'generators' geometry: the full
% C_4v space group (order 128) is built from hand-written permutation GENERATORS
% instead of the dedicated square_lattice_spacegroup provider. Demonstrates how
% to run a symmetry/system that is NOT hard-coded. FULL M sweep, EXACT
% diagonalisation of every (M,Gamma) block (ed_thresh=inf). Used by
% VALIDATE_GENERATORS_SQUARE4X4 to check it reproduces the full ED (and the
% native provider) bit-for-bit.
geometry  = 'generators';

% --- hand-written generators of the 4x4 torus C_4v space group -------------
% Site (x,y), x=0..3, y=0..3, index i = 1 + x + 4*y. A permutation P has
% P(i) = image of site i (1-based) -- the project-wide convention.
gen_Lx   = 4;  gen_Ly = 4;  gen_N = gen_Lx * gen_Ly;
gen_idx  = 1 : gen_N;
gen_cx   = mod(gen_idx - 1, gen_Lx);
gen_cy   = floor((gen_idx - 1) / gen_Lx);
gen_site = @(xp, yp) 1 + mod(xp, gen_Lx) + gen_Lx * mod(yp, gen_Ly);
gen_tx   = gen_site(gen_cx + 1, gen_cy);     % translation  x -> x+1
gen_ty   = gen_site(gen_cx, gen_cy + 1);     % translation  y -> y+1
gen_c4   = gen_site(-gen_cy, gen_cx);        % C_4 rotation (x,y) -> (-y, x)
gen_mx   = gen_site(-gen_cx, gen_cy);        % mirror       x -> -x
gens     = { gen_tx, gen_ty, gen_c4, gen_mx };   % close -> order 128

% --- bonds are GEOMETRY (separate from symmetry), 1-based, group-invariant --
bonds       = adjacency_square_lattice(gen_Lx, gen_Ly);
point_group = 'C_4v (from generators)';
sys_name    = 'gen_sq4x4';

% --- physics / run knobs (match input_square_4x4_c4v_ED.m) ------------------
s_val     = 0.5;
J         = 1.0;
R         = 8;            % unused for ED
M_lz      = 80;          % unused for ED
ed_thresh = inf;         % exact ED for every block
T_range   = logspace(-1, 1, 40);

clear gen_*;             % drop the construction temporaries (keeps gens/bonds)
