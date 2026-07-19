function full = realify_irrep_table(full, group)
%REALIFY_IRREP_TABLE  Realify a named-irrep cell table via REALIFY_IRREPS.
%
%   FULL = REALIFY_IRREP_TABLE(FULL, GROUP) takes the driver-side irrep table
%   (cell of structs {name, d, data} from BUILD_FULL_IRREP_TABLE, where .data
%   is an [order x 1] character vector for d == 1 or a [d x d x order] matrix
%   stack), lifts it into the struct-array format REALIFY_IRREPS consumes
%   (.name / .d / .mats), realifies every FS = +1 irrep to real orthogonal
%   form, and converts back.
%
%   Why: the I_h polyhedra (icosahedron / dodecahedron / icosidodecahedron)
%   store their named irreps T1g..Hu partly COMPLEX, although all ten I_h
%   irreps have Frobenius-Schur indicator +1 (real type). Realifying them
%   switches the whole downstream pipeline -- real V projectors, real H
%   blocks on the CPU/ED paths, and the REAL FP32 GPU kernel path (half the
%   Krylov VRAM + gather traffic; the dodecahedron s=3/2 enabler). Characters
%   are invariant under the basis change, so weights/spectra are unchanged
%   (up to numerical noise ~1e-13 in the ED gates).
%
%   The old complex-irrep baseline stays reproducible via the drivers'
%   force_complex option, which skips this call.
%
%   GROUP only needs .order and .mul (the I_h providers carry both).
%
%   See also REALIFY_IRREPS, FTLM_OBSERVABLES_PG_IH, FTLM_OBSERVABLES_PG_GPU_IH.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    n   = numel(full);
    irr = struct('name', cell(1, n), 'd', cell(1, n), 'mats', cell(1, n));
    for p = 1 : n
        c = full{p};
        irr(p).name = c.name;
        irr(p).d    = c.d;
        if c.d == 1
            irr(p).mats = reshape(c.data, 1, 1, []);   % [order x 1] -> [1 x 1 x order]
        else
            irr(p).mats = c.data;                      % [d x d x order]
        end
    end

    irr = realify_irreps(irr, group);

    for p = 1 : n
        if full{p}.d == 1
            full{p}.data = reshape(irr(p).mats, [], 1);
        else
            full{p}.data = irr(p).mats;
        end
    end
end
