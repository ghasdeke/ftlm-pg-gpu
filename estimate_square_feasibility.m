function estimate_square_feasibility(Lx, Ly, s_val, R, B)
%   *** SUPERSEDED (2026-06-05) by the generic ESTIMATE_FEASIBILITY(group, bonds,
%   s, R, M_lz), which works for ANY provider (square / kagome / triangular / I_h),
%   predicts the VRAM mode + the g/d hard limits, and is wired into the GPU driver
%   as a pre-flight guard. This square-only (translation, d=1) estimator is kept
%   for back-reference. ***
%ESTIMATE_SQUARE_FEASIBILITY  Project memory/feasibility of an M=0 square-lattice
%                             FTLM run BEFORE launching it.
%
%   ESTIMATE_SQUARE_FEASIBILITY(LX, LY, S_VAL, R, B) prints the dominant
%   memory footprints (host entries table, GPU skeleton, Krylov vectors,
%   lookup) for the M=0 sector of an LX x LY periodic square lattice under
%   the translation group C_LX x C_LY (order LX*LY, all 1D irreps), and
%   compares them against the local hardware.
%
%   The entry count is estimated from the average fraction of antiparallel
%   nearest-neighbour bonds in a random M=0 configuration:
%       frac_anti = (N/2)/(N-1),  n_bonds = 2*N  -> entries/rep = frac*n_bonds.

    if nargin < 3, s_val = 0.5; end
    if nargin < 4, R = 8;       end
    if nargin < 5, B = 8;       end

    N      = Lx * Ly;
    order  = Lx * Ly;
    d_loc  = round(2*s_val + 1);
    assert(d_loc == 2, 'estimate assumes s=1/2 (d_loc=2).');

    n_total = 2^N;
    dimM0   = nchoosek(N, N/2);                 % C(N, N/2), exact for N<=~50 in double
    n_reps  = dimM0 / order;                    % ~ average (most orbits full length)
    n_bonds = 2 * N;
    frac    = (N/2) / (N - 1);
    epr     = frac * n_bonds;                    % entries per rep
    n_entry = n_reps * epr;

    GB = @(b) b / 1e9;
    host_entries = GB(n_entry * 9);             % src int32(4)+tgt int32(4)+g uint8(1)
    host_reps    = GB(n_reps * 8 * 2);          % super_reps + schnack rank array (int64)
    g2_packs     = n_reps < 2^25;               % G2 src|g uint32 packing only below 2^25
    skel_bpe     = 5 - 1*g2_packs;              % 5 B/entry (src4+g1), 4 if G2 packs
    vram_skel    = GB(n_entry * skel_bpe);
    vram_perrep  = GB(n_reps * (8 + 4));        % V_re/V_im (d=1) + sqrt_eig
    vram_krylov  = GB(double(n_reps) * B * 12 * 4);   % ~12 fp32 buffers of n_basis*B

    HOST_GB = 63.3;  VRAM_GB = 20.5;

    fprintf('\n=== %dx%d square lattice, s=1/2, M=0 only, C_%dxC_%d (order %d) ===\n', ...
        Lx, Ly, Lx, Ly, order);
    fprintf('  n_total      = 2^%d = %.3g\n', N, n_total);
    fprintf('  dim(M=0)     = C(%d,%d) = %.4g\n', N, N/2, dimM0);
    fprintf('  n_reps       ~ dim/order = %.3g   (G2 packing: %s)\n', n_reps, ternary(g2_packs,'YES','NO (n_reps>2^25)'));
    fprintf('  n_entries    ~ %.3g   (%.0f/rep, %d bonds, frac_anti=%.3f)\n', n_entry, epr, n_bonds, frac);
    fprintf('  bitmap lookup: %s (n_total %s 2^32)\n', ternary(n_total<=2^32,'OK','IMPOSSIBLE -> need schnack'), ternary(n_total<=2^32,'<=','>'));
    fprintf('  --- HOST (have %.0f GB) ---\n', HOST_GB);
    fprintf('    entries table        : %7.1f GB  %s\n', host_entries, flag(host_entries, HOST_GB));
    fprintf('    super_reps + rank    : %7.1f GB\n', host_reps);
    fprintf('    HOST TOTAL (approx)  : %7.1f GB  %s\n', host_entries+host_reps+3, flag(host_entries+host_reps+3, HOST_GB));
    fprintf('  --- VRAM (have %.0f GB) ---\n', VRAM_GB);
    fprintf('    skeleton (1 irrep)   : %7.1f GB  %s\n', vram_skel, flag(vram_skel, VRAM_GB));
    fprintf('    per-rep V + sqrt_eig : %7.1f GB\n', vram_perrep);
    fprintf('    Krylov (B=%d)         : %7.1f GB\n', B, vram_krylov);
    fprintf('    VRAM TOTAL (B=%d)     : %7.1f GB  %s\n', B, vram_skel+vram_perrep+vram_krylov, flag(vram_skel+vram_perrep+vram_krylov, VRAM_GB));
    fprintf('    Krylov (B=1)         : %7.1f GB\n', GB(double(n_reps)*1*12*4));
    fprintf('\n');
end

function s = ternary(c,a,b), if c, s=a; else, s=b; end, end
function s = flag(val, cap), if val > cap, s = sprintf('<-- EXCEEDS %.0f GB', cap); else, s = '(fits)'; end, end
