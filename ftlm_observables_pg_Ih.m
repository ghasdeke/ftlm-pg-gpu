function out = ftlm_observables_pg_Ih(input_file)
%FTLM_OBSERVABLES_PG_IH  Sector-FTLM with a finite symmetry group.
%
%   Despite the historical name, this driver is GROUP-GENERIC: the
%   'geometry' input selects the provider (icosahedron / icosidodecahedron
%   under I_h, or an Lx x Ly periodic square lattice under the translation
%   group C_Lx x C_Ly). The square lattice needs Lx and Ly in the input
%   file. An optional output struct (T_range, C_T, chi_T, Z_eff, ...) is
%   returned when requested, so callers can compare runs without reloading
%   the saved .mat (used by VALIDATE_SQUARE_4X4).
%
%   FTLM_OBSERVABLES_PG_IH(INPUT_FILE) computes the specific heat C(T),
%   magnetic susceptibility chi(T), and effective partition function
%   Z_eff(T) of the 12-site Heisenberg icosahedron with local spin
%   S_VAL, exploiting the joint conservation of total S^z and the full
%   icosahedral point group I_h (order 120, 10 irreducible representations).
%   The Hilbert space is decomposed into simultaneous eigenspaces of S^z
%   (label M) and an I_h irrep (label Gamma), and FTLM is run independently
%   in each (M, Gamma) block.
%
%   The block construction uses the Phase gamma.2 super-rep + column-
%   picking machinery (ENUMERATE_SECTOR_WITH_IH_GAMMA2,
%   BUILD_HEISENBERG_SPARSE_IH_GAMMA2). Each block has dimension
%       n_basis(M, Gamma) = sum_r n_Gamma(r),
%   sampling only the alpha=1 partner row of Gamma. Block eigenvalues
%   are replicated d_Gamma times in the full Hilbert space (one copy
%   per partner row), and we apply this multiplicity in the FTLM
%   weights.
%
%   Aggregation rule:
%       Z(beta)   = sum_{M, Gamma, k} mult_M * d_Gamma * w_k * exp(-beta*E_k)
%       <O>(beta) = sum_{M, Gamma, k} mult_M * d_Gamma * w_k * O_k *
%                                                 exp(-beta*E_k) / Z
%   with w_k = (n_basis / R_eff) * |q_k^(1)|^2 from the per-block FTLM
%   chain, and mult_M = 1 + (M > 0) for the M <-> -M spin-inversion.
%
%   INPUT_FILE is a plain MATLAB script. Required variables:
%       s_val      local spin (0.5, 1, 1.5, ...)
%       J          uniform nearest-neighbor exchange coupling
%       R          number of FTLM random vectors per (M, Gamma) sector
%       M_lz       Lanczos iterations per random vector
%       T_range    temperature grid (positive, units of J/k_B)
%
%   Optional inputs (defaults in parentheses):
%       only_M0          (false)  restrict to M = 0 (shortcut for M_sectors = 0)
%       M_sectors        ([])     vector of non-negative |M| sectors to run, e.g.
%                                 0, 2, or [0 1 2] (the +/-M mirror is folded in).
%                                 Overrides only_M0. Default [] = full sweep
%                                 (0..M_max) -> physical C(T) and chi(T); a subset
%                                 gives that subset's partial / per-sector result.
%       irrep_list       ({...})  cell array of irrep names to include
%                                 (default: all 10). Names: 'A_g', 'A_u',
%                                 'T1g', 'T1u', 'T2g', 'T2u', 'F_g',
%                                 'F_u', 'H_g', 'H_u'.
%       ed_thresh        (0)      blocks with n_basis <= ed_thresh are
%                                 diagonalized exactly instead of FTLM.
%                                 For the s=1/2 icosahedron every block
%                                 is small (<= 46); ed_thresh = inf gives
%                                 the exact ED reference. ed_thresh = 0
%                                 forces FTLM throughout.
%       output_dir       ('.')    directory for the output .mat file
%
%   Output: a single .mat file named ftlm_pg_Ih_icos_s<s_str>.mat in
%   OUTPUT_DIR, containing T_range, C_T, chi_T, Z_eff, configuration
%   and per-sector diagnostics.
%
%   See also ENUMERATE_SECTOR_WITH_IH_GAMMA2,
%            BUILD_HEISENBERG_SPARSE_IH_GAMMA2,
%            RUN_FTLM_PG_SECTOR,
%            COMPUTE_OBSERVABLES_PG,
%            FTLM_OBSERVABLES_PG.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    if nargin < 1 || isempty(input_file)
        error('ftlm_observables_pg_Ih:NoInput', ...
              'Usage: ftlm_observables_pg_Ih(''input.m'')');
    end
    if exist(input_file, 'file') ~= 2
        error('ftlm_observables_pg_Ih:InputNotFound', ...
              'Input file not found: %s', input_file);
    end

    fprintf('=== ftlm_observables_pg_Ih: symmetry-adapted FTLM/ED (CPU FP64) ===\n');
    fprintf('Input file: %s\n\n', input_file);

    run(input_file);   %#ok<*NODEF>

    %% Required inputs
    required = {'s_val', 'J', 'R', 'M_lz', 'T_range'};
    for k = 1 : numel(required)
        if ~exist(required{k}, 'var')
            error('ftlm_observables_pg_Ih:Missing', ...
                  'Required variable missing in %s: %s', input_file, required{k});
        end
    end

    %% Optional inputs
    if ~exist('only_M0',    'var'), only_M0    = false;          end
    if ~exist('ed_thresh',  'var'), ed_thresh  = 0;              end
    if ~exist('output_dir', 'var'), output_dir = '.';            end
    if ~exist('geometry',   'var'), geometry   = 'icosahedron';  end
    if ~exist('use_spin_flip', 'var'), use_spin_flip = false;    end
    % force_complex: keep the I_h named irreps in their historical COMPLEX form
    % (skip realify_irrep_table below) -> byte-identical old baselines.
    if ~exist('force_complex', 'var'), force_complex = false;    end
    have_irrep_list = exist('irrep_list', 'var') && ~isempty(irrep_list);

    % Create output_dir up front (mirrors the GPU driver): a missing
    % directory used to fail only at the final save, AFTER the entire
    % computation.
    if ~exist(output_dir, 'dir'), mkdir(output_dir); end

    %% Validation
    assert(s_val > 0 && abs(2*s_val - round(2*s_val)) < 1e-12, ...
        's_val must be a positive half-integer.');
    assert(isfinite(J) && isnumeric(J) && isscalar(J), 'J must be a finite scalar.');
    assert(R >= 1 && R == round(R), 'R must be a positive integer.');
    assert(M_lz >= 1 && M_lz == round(M_lz), 'M_lz must be a positive integer.');
    assert(all(T_range > 0) && all(isfinite(T_range)), ...
        'T_range must contain positive finite values.');
    T_range = T_range(:).';

    %% System setup. The geometry switch selects (group, bonds) from the
    %  appropriate provider; add new geometries here. The square lattice
    %  reads Lx, Ly from the input file workspace (visible here after run()).
    switch lower(geometry)
        case 'icosahedron'
            group = icosahedron_Ih_full();
            bonds = adjacency_icosahedron_Ih();
            sys_label = '12-site icosahedron';
            sym_label = 'I_h (|G|=120)';
            geo_tag   = 'icos';
        case 'icosidodecahedron'
            group = icosidodecahedron_Ih_full();
            bonds = adjacency_icosidodecahedron_Ih();
            sys_label = '30-site icosidodecahedron';
            sym_label = 'I_h (|G|=120)';
            geo_tag   = 'icosido';
        case 'dodecahedron'
            group = dodecahedron_Ih_full();
            bonds = adjacency_dodecahedron_Ih();
            sys_label = '20-site dodecahedron';
            sym_label = 'I_h (|G|=120)';
            geo_tag   = 'dodec';
        case 'cuboctahedron'
            % Quasiregular O_h Archimedean solid (12 vertices, 24 edges,
            % 8 corner-sharing triangles -> frustrated). Irreps generic:
            % 10 irreps, max d=3, all FS=+1 -> realified.
            [group, bonds] = cuboctahedron_Oh();
            group.irreps = irreps_from_group(group);
            group.irreps = realify_irreps(group.irreps, group);   % FS=+1: real orthogonal irreps
            sys_label = '12-site cuboctahedron';
            sym_label = sprintf('O_h (|G|=%d)', group.order);
            geo_tag   = 'cubocta';
        case 'square_lattice'
            assert(exist('Lx', 'var') == 1 && exist('Ly', 'var') == 1, ...
                'square_lattice geometry requires Lx and Ly in the input file.');
            group = square_lattice_translation_group(Lx, Ly);
            bonds = adjacency_square_lattice(Lx, Ly);
            sys_label = sprintf('%dx%d periodic square lattice (N=%d)', Lx, Ly, Lx*Ly);
            sym_label = sprintf('C_%d x C_%d (|G|=%d)', Lx, Ly, Lx*Ly);
            geo_tag   = sprintf('sq%dx%d', Lx, Ly);
        case 'square_lattice_c4v'
            % Full space group: translations C_Lx x C_Ly semidirect point
            % group (C_4v if Lx==Ly, else C_2v). Irreps built generically.
            assert(exist('Lx', 'var') == 1 && exist('Ly', 'var') == 1, ...
                'square_lattice_c4v geometry requires Lx and Ly in the input file.');
            group = square_lattice_spacegroup(Lx, Ly);
            group.irreps = irreps_from_group(group);
            % The point group contains C_2 (-k is always in the star of k), so
            % every irrep is self-conjugate with Frobenius-Schur indicator +1:
            % bring them to real orthogonal form so the (M,Gamma) H blocks are
            % real-symmetric (real eig / Lanczos, half the storage). The
            % abelian translation path and the I_h path are NOT touched.
            group.irreps = realify_irreps(group.irreps, group);
            bonds = adjacency_square_lattice(Lx, Ly);
            sys_label = sprintf('%dx%d square lattice + %s point group (N=%d)', ...
                Lx, Ly, group.point_group, Lx*Ly);
            sym_label = sprintf('%s space group (|G|=%d)', group.point_group, group.order);
            geo_tag   = sprintf('sq%dx%dsg', Lx, Ly);
        case 'kagome'
            % Kagome torus on a C_6-symmetric supercell T1=(kag_a,kag_b),
            % T2=R6*T1; full space group C_6v semidirect translations.
            assert(exist('kag_a','var')==1 && exist('kag_b','var')==1, ...
                'kagome geometry requires kag_a and kag_b in the input file.');
            [group, bonds] = kagome_spacegroup(kag_a, kag_b);
            group.irreps = irreps_from_group(group);
            group.irreps = realify_irreps(group.irreps, group);   % FS=+1: real orthogonal irreps
            sys_label = sprintf('kagome torus (a,b)=(%d,%d), N=%d', kag_a, kag_b, group.N);
            sym_label = sprintf('%s space group (|G|=%d)', group.point_group, group.order);
            geo_tag   = sprintf('kag%d_%d', kag_a, kag_b);
        case 'triangular'
            % Triangular (hexagonal Bravais) torus on a C_6-symmetric supercell
            % T1=(tri_a,tri_b), T2=R6*T1; full space group C_6v semidirect
            % translations (1 site/cell, site at the C_6 centre).
            assert(exist('tri_a','var')==1 && exist('tri_b','var')==1, ...
                'triangular geometry requires tri_a and tri_b in the input file.');
            [group, bonds] = triangular_spacegroup(tri_a, tri_b);
            group.irreps = irreps_from_group(group);
            group.irreps = realify_irreps(group.irreps, group);   % FS=+1: real orthogonal irreps
            sys_label = sprintf('triangular torus (a,b)=(%d,%d), N=%d', tri_a, tri_b, group.N);
            sym_label = sprintf('%s space group (|G|=%d)', group.point_group, group.order);
            geo_tag   = sprintf('tri%d_%d', tri_a, tri_b);
        case 'generators'
            % User-supplied symmetry: the input file gives only the permutation
            % GENERATORS (cell/matrix of 1xN site perms); group_from_generators
            % closes them into the full group in the standard provider layout, so
            % systems NOT hard-coded above can be run. The nearest-neighbour bond
            % list is geometry, supplied SEPARATELY in the input file ('bonds'),
            % and MUST be invariant under every group element (correctness cond.).
            assert(exist('gens', 'var') == 1, ...
                ['generators geometry requires permutation generators ''gens'' ', ...
                 '(cell/matrix of 1xN perms) in the input file.']);
            assert(exist('bonds', 'var') == 1 && ~isempty(bonds), ...
                ['generators geometry requires a bond list ''bonds'' [n_bonds x 2] ', ...
                 'in the input file (geometry is separate from symmetry).']);
            if exist('point_group', 'var') == 1, pg_lab = point_group; else, pg_lab = 'custom'; end
            group = group_from_generators(gens, [], pg_lab);
            % HARD GUARD before the |G|^2-cost irrep build: a non-invariant
            % bond list would give silently WRONG physics (the sum rule is a
            % trace identity and does NOT catch it).
            assert_bonds_group_invariant(bonds, group, 'generators geometry');
            group.irreps = irreps_from_group(group);
            group.irreps = realify_irreps(group.irreps, group);   % FS=+1: real orthogonal irreps
            if exist('sys_name', 'var') == 1, geo_tag = sys_name; else, geo_tag = 'custom'; end
            sys_label = sprintf('user-generated group ''%s'' (N=%d, |G|=%d)', ...
                geo_tag, group.N, group.order);
            sym_label = sprintf('user permutation group (|G|=%d)', group.order);
        otherwise
            error('ftlm_observables_pg_Ih:Geometry', ...
                ['Unknown geometry "%s" (icosahedron / dodecahedron / ', ...
                 'icosidodecahedron / cuboctahedron / square_lattice / ', ...
                 'square_lattice_c4v / kagome / triangular / generators).'], geometry);
    end

    N_sites = double(group.N);
    d_loc   = round(2*s_val + 1);
    n_total = double(d_loc)^N_sites;
    M_max   = round(N_sites * s_val);

    %% M-sector selection. Input M_sectors (vector of non-negative |M| in 0..M_max)
    %  restricts the run to those sectors; else only_M0 -> {0}; else the full sweep
    %  0..M_max (the physical C(T)+chi(T)). The +/-M mirror is folded via mult_M, so
    %  list only non-negative |M|. M_sectors overrides only_M0.
    if exist('M_sectors', 'var') && ~isempty(M_sectors)
        M_run = unique(round(abs(M_sectors(:)')));
        assert(all(M_run >= 0 & M_run <= M_max), ...
            'M_sectors must lie in 0..M_max=%d (got [%s]).', M_max, num2str(M_run));
    elseif only_M0
        M_run = 0;
    else
        M_run = 0 : M_max;
    end
    is_full_sweep = isequal(M_run, 0:M_max);

    two_s = round(2 * s_val);
    if mod(two_s, 2) == 0
        s_str = sprintf('%d', two_s/2);
    else
        s_str = sprintf('%do2', two_s);
    end

    %% Irreps: generic list from the provider (group.irreps) or the I_h
    %  named fields. Default to ALL irreps when the input gives no list.
    irreps_all = build_full_irrep_table(group);
    % I_h polyhedra: realify the named irreps (all FS = +1) -> real V / real-
    % symmetric H blocks (real eig/Lanczos, half storage). force_complex keeps
    % the historical complex form (byte-identical old baselines).
    if ~isfield(group, 'irreps') && ~force_complex
        irreps_all = realify_irrep_table(irreps_all, group);
    end
    if have_irrep_list
        irreps = filter_irreps(irreps_all, irrep_list);
    else
        irreps     = irreps_all;
        irrep_list = cellfun(@(c) c.name, irreps, 'uni', 0);
    end

    %% Spin-flip Z2 (opt-in `use_spin_flip`): G -> G x Z2 at M=0 only, each
    %  Gamma split into Gamma+- with rho'((g,f)) = (+-1)^f rho(g). Mirrors
    %  the GPU driver; see ADD_SPIN_FLIP_Z2.
    if use_spin_flip
        group_sf  = add_spin_flip_z2(group);
        irreps_sf = cell(1, 0);
        for kk = 1 : numel(irreps)
            c0 = irreps{kk};
            for sgn = [1, -1]
                c2 = c0;
                if c0.d == 1
                    mm      = c0.data(:).';
                    c2.data = [mm, sgn * mm];
                else
                    c2.data = cat(3, c0.data, sgn * c0.data);
                end
                if sgn > 0, c2.name = [c0.name '+']; else, c2.name = [c0.name '-']; end
                irreps_sf{end + 1} = c2; %#ok<AGROW>
            end
        end
        fprintf('Spin-flip Z2 ON for M=0: |G| %d -> %d, %d -> %d irreps (Gamma+-).\n', ...
                group.order, group_sf.order, numel(irreps), numel(irreps_sf));
    end

    fprintf('System:    %s, s=%s (d_loc=%d, n_total=%g, M_max=%d)\n', ...
        sys_label, s_str, d_loc, n_total, M_max);
    fprintf('Symmetry:  %s on %d / %d irreps\n', ...
        sym_label, numel(irreps), numel(irreps_all));
    fprintf('FTLM:      R=%d, M_lz=%d, T-grid: %d points in [%.3g, %.3g]\n', ...
        R, M_lz, numel(T_range), min(T_range), max(T_range));
    if is_full_sweep
        fprintf('Full M sweep (M = 0..%d) -> physical C(T) and chi(T)\n', M_max);
    else
        fprintf('Restricted to M sectors {%s} of 0..%d (partial / sector observables)\n', ...
            num2str(M_run), M_max);
    end
    if ed_thresh == inf
        fprintf('Exact ED for every block (ed_thresh = inf)\n');
    elseif ed_thresh > 0
        fprintf('Dense ED for blocks with n_basis <= %d, FTLM otherwise\n', ed_thresh);
    else
        fprintf('FTLM throughout (ed_thresh = 0)\n');
    end
    fprintf('\n');

    %% Main loop over (M, Gamma) sectors
    all_E       = zeros(0, 1);
    all_w       = zeros(0, 1);
    all_M       = zeros(0, 1);
    sector_M    = zeros(0, 1);
    sector_G    = zeros(0, 1);
    sector_dims = zeros(0, 1);

    t_start = tic;

    for M = 0 : M_max
        if ~ismember(M, M_run), continue; end
        mult_M = 1 + (M > 0);

        % Spin-flip Z2: P preserves only the M = 0 sector, so the extended
        % group + doubled (Gamma+-) irrep list apply there alone.
        if use_spin_flip && M == 0
            group_M = group_sf;  irreps_M = irreps_sf;
        else
            group_M = group;     irreps_M = irreps;
        end

        % Irrep-INDEPENDENT precompute: M-sector enumeration + min_image
        % + stabiliser lists. Runs once per M and is reused for all 10
        % irreps below.
        cache_M = enumerate_M_orbits_Ih(s_val, M, group_M);

        for ig = 1 : numel(irreps_M)
            ir = irreps_M{ig};

            [reps, V_per_rep, eig_per_rep, n_per_rep, ~] = ...
                apply_irrep_to_orbits(cache_M, ir.data, ir.d, group_M);
            % Sum in DOUBLE: int32 native sum saturates at 2^31-1.
            n_basis = sum(double(n_per_rep));
            if n_basis == 0
                continue;
            end

            sector_M(end+1, 1)    = M;             %#ok<AGROW>
            sector_G(end+1, 1)    = ig;            %#ok<AGROW>
            sector_dims(end+1, 1) = n_basis;       %#ok<AGROW>

            H = build_heisenberg_sparse_Ih_gamma2(reps, V_per_rep, ...
                eig_per_rep, n_per_rep, bonds, s_val, J, ir.data, ir.d, group_M);

            % Decide arithmetic regime: real if H_block is real to within
            % a relative tolerance. 1D irreps A_g and A_u give real H;
            % higher-d irreps give complex H in general.
            max_abs_imag = max(abs(imag(H(:))));
            max_abs_full = max(abs(H(:))) + 1e-30;
            is_real = (max_abs_imag / max_abs_full < 1e-10);
            if is_real
                H = real(H);
            end

            % Force Hermiticity to absorb any FP noise.
            H = 0.5 * (H + H');

            t_sec = tic;
            if n_basis <= ed_thresh
                % Dense ED branch (deterministic).
                Hd = full(H);
                E_sec = sort(real(eig(Hd)));
                w_sec = ones(numel(E_sec), 1);
                method_str = 'ED';
            else
                seed = 7e6 + M * 1e4 + ig * 100;
                [E_sec, w_sec] = run_ftlm_pg_sector(H, n_basis, R, M_lz, ...
                                                    is_real, seed);
                method_str = sprintf('FTLM R=%d, M_lz=%d', ...
                                     min(R, n_basis), min(M_lz, n_basis));
            end

            % Apply multiplicities:
            %   mult_M : M <-> -M spin-inversion (1 for M = 0)
            %   ir.d   : Gamma partner-row degeneracy in full Hilbert space
            w_sec = w_sec * (mult_M * ir.d);

            all_E = [all_E; E_sec];                       %#ok<AGROW>
            all_w = [all_w; w_sec];                       %#ok<AGROW>
            all_M = [all_M; M * ones(numel(E_sec), 1)];   %#ok<AGROW>

            arith_str = ternary(is_real, 'real', 'cplx');
            fprintf('  M=%2d %-4s (d=%d, %s) n_basis=%4d full_block=%4d %-22s t=%.2fs\n', ...
                M, ir.name, ir.d, arith_str, n_basis, n_basis * ir.d, ...
                method_str, toc(t_sec));
        end
    end
    t_wall = toc(t_start);
    fprintf('\nTotal wall time: %.2f s\n', t_wall);

    %% Observables
    [C_T, chi_T, Z_eff] = compute_observables_pg(all_E, all_w, all_M, T_range);

    %% Save results
    mat_name = sprintf('ftlm_pg_Ih_%s_s%s.mat', geo_tag, s_str);
    mat_path = fullfile(output_dir, mat_name);
    n_total_save = n_total;     %#ok<NASGU>
    save(mat_path, ...
        'T_range', 'C_T', 'chi_T', 'Z_eff', ...
        's_val', 'J', 'R', 'M_lz', 'M_max', 'n_total_save', ...
        'only_M0', 'irrep_list', 'ed_thresh', 'geometry', ...
        'sector_M', 'sector_G', 'sector_dims', ...
        't_wall', '-v7.3');
    fprintf('Results saved to: %s\n', mat_path);

    %% Sum-rule check. Compare against the part of the Hilbert space actually
    %  covered: dim(M=0) for an only_M0 run, the full n_total otherwise.
    if is_full_sweep
        Z_expected  = n_total;
        chk_label   = 'full dim';
    else
        A0 = round(N_sites * s_val);                           % digit sum at M=0
        [D_chk, ~] = build_D_table(N_sites, round(2*s_val), A0 + M_max);
        Z_expected = 0;
        for Mc = M_run
            Z_expected = Z_expected + (1 + (Mc > 0)) * ...
                double(D_chk(N_sites + 1, A0 + Mc + 1));        % mult_M * dim(M)
        end
        chk_label  = sprintf('dim(M in {%s})', num2str(M_run));
    end
    Z_inf = sum(all_w);
    fprintf('Sum-rule check: sum_i w_i = %.6g, %s = %.6g, rel.err = %.2e\n', ...
        Z_inf, chk_label, Z_expected, abs(Z_inf - Z_expected) / max(Z_expected, 1));

    %% Optional return struct (so callers can compare runs without reloading).
    if nargout > 0
        out = struct('T_range', T_range, 'C_T', C_T, 'chi_T', chi_T, ...
            'Z_eff', Z_eff, 's_val', s_val, 'J', J, 'geometry', geometry, ...
            'ed_thresh', ed_thresh, 'R', R, 'M_lz', M_lz, ...
            'all_E', all_E, 'all_w', all_w, 'all_M', all_M, ...
            'sum_w', Z_inf, 'Z_expected', Z_expected);
    end
end

% ----------------------------------------------------------------
function full = build_full_irrep_table(group)
%BUILD_FULL_IRREP_TABLE  All irreps as {name, d, data}, group-generically.
%   data is the form IRREP_MATRIX expects: an [order x 1] character vector
%   for d == 1, else a [d x d x order] matrix stack.
    if isfield(group, 'irreps')
        n    = numel(group.irreps);
        full = cell(1, n);
        for p = 1 : n
            irp = group.irreps(p);
            if irp.d == 1
                data = reshape(irp.mats, [], 1);     % [order x 1]
            else
                data = irp.mats;                     % [d x d x order]
            end
            full{p} = struct('name', irp.name, 'd', irp.d, 'data', data);
        end
    else
        full = {};
        full{end+1} = struct('name', 'A_g', 'd', 1, 'data', group.Ag);
        full{end+1} = struct('name', 'A_u', 'd', 1, 'data', group.Au);
        full{end+1} = struct('name', 'T1g', 'd', 3, 'data', group.T1g);
        full{end+1} = struct('name', 'T1u', 'd', 3, 'data', group.T1u);
        full{end+1} = struct('name', 'T2g', 'd', 3, 'data', group.T2g);
        full{end+1} = struct('name', 'T2u', 'd', 3, 'data', group.T2u);
        full{end+1} = struct('name', 'F_g', 'd', 4, 'data', group.Fg);
        full{end+1} = struct('name', 'F_u', 'd', 4, 'data', group.Fu);
        full{end+1} = struct('name', 'H_g', 'd', 5, 'data', group.Hg);
        full{end+1} = struct('name', 'H_u', 'd', 5, 'data', group.Hu);
    end
end

function irreps = filter_irreps(full, names_keep)
    known = cellfun(@(c) c.name, full, 'uni', 0);
    % A misspelled entry used to be dropped SILENTLY (the run continued with
    % the matching subset) -- error per unknown name instead.
    unknown = names_keep(~ismember(names_keep, known));
    if ~isempty(unknown)
        error('ftlm_observables_pg_Ih:no_irreps', ...
              'Unknown irrep name(s) in irrep_list: %s. Known: %s.', ...
              strjoin(cellstr(string(unknown)), ', '), strjoin(known, ', '));
    end
    irreps = full(ismember(known, names_keep));
end

function s = ternary(cond, a, b)
    if cond, s = a; else, s = b; end
end
