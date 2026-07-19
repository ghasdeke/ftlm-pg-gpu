function [irreps, info] = realify_irreps(irreps, group, opts)
%REALIFY_IRREPS  Bring self-conjugate (FS = +1) irreps to real orthogonal form.
%
%   IRREPS = REALIFY_IRREPS(IRREPS, GROUP) post-processes the generic irrep
%   struct array produced by IRREPS_FROM_GROUP (fields .name / .d / .mats,
%   with .mats a [d x d x |G|] stack of unitary matrices rho(g) in the random
%   basis from the regular-representation decomposition). For every irrep whose
%   Frobenius-Schur indicator
%
%       nu = (1/|G|) * sum_g chi(g^2),   chi(g) = trace(rho(g)),  g^2 = mul(g,g)
%
%   equals +1 (the irrep is self-conjugate AND of REAL type), the routine finds
%   a unitary basis change W such that
%
%       sigma(g) = W' * rho(g) * W
%
%   is real orthogonal for ALL g, then nulls the (numerically negligible)
%   imaginary part exactly. Irreps with nu != +1 (complex type, nu ~ 0; or
%   quaternionic type, nu ~ -1) are left UNCHANGED as complex matrices -- a
%   clean fallback. The returned struct array keeps the same .name / .d and the
%   same generic interface; only .mats is replaced (real for the realified
%   irreps, unchanged complex otherwise).
%
%   Why: the FTLM engine (block-Lanczos SpMV) is memory-bound. Complex
%   vectors / matrix elements cost ~2x the memory traffic and ~4x the FLOPs of
%   real arithmetic. Real irrep matrices make every symmetry-adapted H block
%   real-symmetric instead of complex-Hermitian, so the CPU driver
%   (FTLM_OBSERVABLES_PG_IH) and the CPU fallback / dense-ED paths run real
%   eig / real Lanczos at half the storage. For the space groups handled here
%   the point group always contains C_2 (-k lies in the star of every k), so
%   every irrep is self-conjugate with nu = +1, i.e. ALL of them realify; the
%   random commutant basis from IRREPS_FROM_GROUP merely hides it.
%
%   Construction of W (standard, via the symmetric unitary intertwiner S):
%     1. S0 = sum_g conj(rho(g)) * X * rho(g)'  with X = X.' a symmetric
%        complex seed (deterministically seeded). Group averaging forces S0 to
%        intertwine rho with its conjugate,  conj(rho(g)) = S0 * rho(g) * S0^-1,
%        and the symmetric seed makes S0 symmetric. By Schur the intertwiner
%        space is 1-D, so S0 = c * S with S the (essentially unique) symmetric
%        unitary intertwiner.
%     2. Normalise S0 to the symmetric unitary S (S0' * S0 = |c|^2 * I).
%     3. W = S^(-1/2)  (= conj(sqrtm(S))).  Then sigma(g) = W' * rho(g) * W
%        = S^(1/2) * rho(g) * S^(-1/2) satisfies conj(sigma) = sigma, i.e. it
%        is real. (Any residual global phase of S cancels in sigma.)
%
%   IRREPS = REALIFY_IRREPS(IRREPS, GROUP, OPTS) accepts an options struct:
%       opts.tol      reality / homomorphism / unitarity tolerance (def 1e-8)
%       opts.verbose  print a per-irrep table (def false; a one-line summary
%                     is printed regardless)
%       opts.seed     base RNG seed for the deterministic seeds (def 8675309)
%
%   [IRREPS, INFO] = ... also returns a struct array INFO(p) with .name, .d,
%   .nu (FS indicator), .realified (logical), .max_imag_before, .max_imag_after,
%   .hom_err, .uni_err, .char_err for inspection / testing.
%
%   The GROUP struct only needs .mul ([|G| x |G|] multiplication table) and
%   .order. The global RNG state is saved and restored, so the function is
%   side-effect free and fully deterministic.
%
%   See also IRREPS_FROM_GROUP, FTLM_OBSERVABLES_PG_IH,
%            FTLM_OBSERVABLES_PG_GPU_IH, APPLY_IRREP_TO_ORBITS.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    if nargin < 3, opts = struct(); end
    if ~isfield(opts, 'tol'),     opts.tol     = 1e-8;    end
    if ~isfield(opts, 'verbose'), opts.verbose = false;   end
    if ~isfield(opts, 'seed'),    opts.seed    = 8675309; end

    n   = double(group.order);
    mul = double(group.mul);
    assert(size(mul, 1) == n && size(mul, 2) == n, ...
        'realify_irreps: group.mul must be |G| x |G|.');

    % g2(g) = index of g^2 (for the Frobenius-Schur indicator).
    g2 = zeros(n, 1);
    for g = 1 : n, g2(g) = mul(g, g); end

    % Save + restore the global RNG so this routine is deterministic and has
    % no side effect on the caller's stream.
    rng_state = rng;
    restore_rng = onCleanup(@() rng(rng_state));

    n_irr = numel(irreps);
    info  = struct('name', {}, 'd', {}, 'nu', {}, 'realified', {}, ...
                   'max_imag_before', {}, 'max_imag_after', {}, ...
                   'hom_err', {}, 'uni_err', {}, 'char_err', {});
    n_realified = 0;

    for p = 1 : n_irr
        rho = irreps(p).mats;
        d   = double(irreps(p).d);

        % --- Frobenius-Schur indicator nu = (1/|G|) sum_g chi(g^2) ----------
        chi = zeros(n, 1);
        for g = 1 : n, chi(g) = trace_d(rho, g, d); end
        nu  = real(sum(chi(g2))) / n;

        imag_before = max_abs_imag(rho);
        ip = struct('name', irreps(p).name, 'd', d, 'nu', nu, ...
                    'realified', false, 'max_imag_before', imag_before, ...
                    'max_imag_after', imag_before, 'hom_err', NaN, ...
                    'uni_err', NaN, 'char_err', NaN);

        if abs(nu - 1) > 0.5
            % nu ~ 0 (complex type) or nu ~ -1 (quaternionic): NOT realifiable
            % to a real orthogonal form. Leave the irrep unchanged (complex).
            info(p) = ip;  %#ok<AGROW>
            continue;
        end

        % --- Fast path: already real (e.g. d=1 momentum-real chars) ---------
        scale = max(1, max(abs(rho(:))));
        if imag_before <= opts.tol * scale
            sigma = real(rho);
            ip.realified      = true;
            ip.max_imag_after = 0;
            ip.hom_err        = hom_residual(sigma, mul, d, n, opts.seed + p);
            ip.uni_err        = uni_residual(sigma, d, n, opts.seed + p);
            ip.char_err       = char_residual(sigma, chi, d, n);
            irreps(p).mats = sigma;
            info(p) = ip;  %#ok<AGROW>
            n_realified = n_realified + 1;
            continue;
        end

        % --- General construction: W from the symmetric unitary intertwiner -
        sigma   = [];
        success = false;
        for attempt = 0 : 3
            rng(opts.seed + 1000 * attempt + p);   % deterministic, varied per retry
            W = realifying_transform(rho, d, n);
            if isempty(W), continue; end
            cand = apply_transform(rho, W, d, n);  % sigma(g) = W' rho(g) W
            if max_abs_imag(cand) <= opts.tol * max(1, max(abs(real(cand(:)))))
                sigma   = cand;
                success = true;
                break;
            end
        end

        if ~success
            % FS said +1 but the numerical realisation did not converge: keep
            % the irrep complex (safe fallback) and warn so it is not silent.
            warning('realify_irreps:NoRealForm', ...
                ['Irrep %s (d=%d, nu=%.3f) has FS=+1 but no real form was found ', ...
                 'within tol=%.1e; leaving it complex.'], ...
                irreps(p).name, d, nu, opts.tol);
            info(p) = ip;  %#ok<AGROW>
            continue;
        end

        ip.hom_err  = hom_residual(sigma, mul, d, n, opts.seed + p);
        ip.uni_err  = uni_residual(sigma, d, n, opts.seed + p);
        ip.char_err = char_residual(sigma, chi, d, n);
        if ip.hom_err > opts.tol || ip.uni_err > opts.tol || ip.char_err > opts.tol
            warning('realify_irreps:BadRealForm', ...
                ['Irrep %s (d=%d): realified form failed a check ', ...
                 '(hom=%.2e uni=%.2e char=%.2e > tol=%.1e); leaving it complex.'], ...
                irreps(p).name, d, ip.hom_err, ip.uni_err, ip.char_err, opts.tol);
            ip.realified = false;
            info(p) = ip;  %#ok<AGROW>
            continue;
        end

        ip.max_imag_after = max_abs_imag(sigma);    % pre-strip residual
        irreps(p).mats    = real(sigma);            % null the imaginary part exactly
        ip.realified      = true;
        info(p) = ip;  %#ok<AGROW>
        n_realified = n_realified + 1;
    end

    %% Compact summary (always) + optional per-irrep table.
    max_after = 0;
    for p = 1 : n_irr
        if info(p).realified, max_after = max(max_after, info(p).max_imag_after); end
    end
    fprintf(['realify_irreps: %d/%d irreps realified to real orthogonal form ', ...
             '(FS=+1); max residual imag stripped = %.2e\n'], ...
            n_realified, n_irr, max_after);
    if opts.verbose
        fprintf('  %-10s %3s %8s %8s %10s %10s %9s %9s %9s\n', ...
            'name', 'd', 'nu', 'real?', 'imag_pre', 'imag_post', 'hom', 'uni', 'char');
        for p = 1 : n_irr
            fprintf('  %-10s %3d %8.4f %8d %10.2e %10.2e %9.1e %9.1e %9.1e\n', ...
                info(p).name, info(p).d, info(p).nu, info(p).realified, ...
                info(p).max_imag_before, info(p).max_imag_after, ...
                info(p).hom_err, info(p).uni_err, info(p).char_err);
        end
    end
end


% ================================================================
% Helpers
% ================================================================
function W = realifying_transform(rho, d, n)
%REALIFYING_TRANSFORM  W with W' rho(g) W real (symmetric-intertwiner method).
%   Returns [] if the random seed produced a degenerate intertwiner.
    X = randn(d) + 1i * randn(d);
    X = X + X.';                                   % symmetric complex seed
    S0 = zeros(d);
    for g = 1 : n
        R  = rho(:, :, g);
        S0 = S0 + conj(R) * X * R';                % R' = R^-1 (rho unitary)
    end
    S0 = 0.5 * (S0 + S0.');                        % enforce exact symmetry
    sc = sqrt(real(mean(diag(S0' * S0))));         % |c| (S0 = c * unitary)
    if ~(sc > 0) || ~all(isfinite(S0(:)))
        W = [];  return;                           % degenerate seed -> retry
    end
    S = S0 / sc;                                   % symmetric, ~unitary
    Sh = sqrtm(S);
    W  = conj(Sh);                                 % W = S^(-1/2) = conj(S^(1/2))
    if ~all(isfinite(W(:))), W = []; end
end

function sigma = apply_transform(rho, W, d, n)
    sigma = complex(zeros(d, d, n));
    Wt = W';
    for g = 1 : n
        sigma(:, :, g) = Wt * rho(:, :, g) * W;
    end
end

function e = hom_residual(mats, mul, d, n, seed)
%HOM_RESIDUAL  max ||rho(a)rho(b) - rho(mul(a,b))|| over a random pair sample.
    rng(seed);
    npair = min(n * n, 300);
    aa = randi(n, npair, 1);  bb = randi(n, npair, 1);
    e = 0;
    for s = 1 : npair
        A = slice(mats, aa(s), d);
        B = slice(mats, bb(s), d);
        C = slice(mats, mul(aa(s), bb(s)), d);
        e = max(e, norm(A * B - C, 'fro'));
    end
end

function e = uni_residual(mats, d, n, seed)
%UNI_RESIDUAL  max ||rho(g) rho(g)' - I|| over a random sample (orthogonality).
    rng(seed + 7);
    ns = min(n, 120);
    gg = randi(n, ns, 1);
    e = 0;
    I = eye(d);
    for s = 1 : ns
        M = slice(mats, gg(s), d);
        e = max(e, norm(M * M' - I, 'fro'));
    end
end

function e = char_residual(mats, chi_ref, ~, n)
%CHAR_RESIDUAL  max |trace(sigma(g)) - chi_ref(g)| (characters are invariant).
    e = 0;
    for g = 1 : n
        e = max(e, abs(trace(mats(:, :, g)) - chi_ref(g)));
    end
end

function v = trace_d(rho, g, d)
    if d == 1, v = rho(1, 1, g); else, v = trace(rho(:, :, g)); end
end

function M = slice(mats, g, ~)
    M = mats(:, :, g);
end

function m = max_abs_imag(A)
    m = max(abs(imag(A(:))));
    if isempty(m), m = 0; end
end
