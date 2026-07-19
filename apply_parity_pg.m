function Pv = apply_parity_pg(v, P_idx, P_phase)
%APPLY_PARITY_PG  Apply the spin-inversion operator P in the rep basis.
%
%   PV = APPLY_PARITY_PG(V, P_IDX, P_PHASE)
%
%   computes P*V in the (M = 0, p) representative basis described by the
%   data (P_IDX, P_PHASE) returned by PARITY_ACTION_PG. V is an array of
%   shape [dim x B] (one or many vectors as columns); PV has the same
%   shape and class.
%
%   Because P is a permutation-and-phase matrix on the rep basis (each
%   rep is mapped to exactly one other rep with a phase factor), the
%   scatter
%       PV(P_IDX(t), :) = P_PHASE(t) * V(t, :)
%   has no aliasing and can be written as a single direct assignment in
%   MATLAB. The assignment also handles complex P_PHASE and complex V
%   naturally via element-wise broadcasting.
%
%   Inputs:
%       v        [dim x B] real or complex
%       P_idx    [dim x 1] int32 permutation
%       P_phase  [dim x 1] complex double phase per rep
%
%   Output:
%       Pv       [dim x B] same class as V (or promoted to complex when
%                P_phase has non-zero imaginary part and V is real)
%
%   See also PARITY_ACTION_PG.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    assert(size(v, 1) == numel(P_idx), ...
        'apply_parity_pg: V has %d rows, expected %d.', size(v, 1), numel(P_idx));

    if isreal(v) && isreal(P_phase)
        Pv = zeros(size(v), 'like', v);
        Pv(P_idx, :) = P_phase .* v;
    else
        % Promote to complex if either side has imaginary content.
        Pv = complex(zeros(size(v)));
        Pv(P_idx, :) = P_phase .* v;
    end
end
