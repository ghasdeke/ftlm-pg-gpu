function test_gpu_sizing_invariants()
%TEST_GPU_SIZING_INVARIANTS  Sizing formulas hold on EVERY GPU size (CPU-only).
%
%   The failure class "behavior depends on the GPU size" has no natural
%   local test coverage (two 2^31 elements-per-gpuArray incidents shipped
%   despite a green suite on the 20-GB dev card). This test sweeps the
%   VRAM-adaptive sizing formulas across FAKE card sizes via
%   FTLM_FAKE_FREE_VRAM_GB (see GPU_FREE_BYTES) and asserts the
%   device-INDEPENDENT invariants:
%     (1) adaptive_enum_chunk: N_sites * chunk stays below the 2^31
%         elements-per-gpuArray cap (with margin) for every card size and
%         every supported N; floors/caps respected; no-GPU fallback capped.
%     (2) the V0 element cap formula (n_basis * B <= 2^31-1) used by
%         run_ftlm's B clamp.
%   Pure host logic -- runs without any GPU.
%
%   See also GPU_FREE_BYTES, ADAPTIVE_ENUM_CHUNK, RUN_FTLM_PG_SECTOR_GPU_IH,
%            docs/AUDIT_gpu_portability_2026-07-03.md.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    prev = getenv('FTLM_FAKE_FREE_VRAM_GB');
    restore = onCleanup(@() setenv('FTLM_FAKE_FREE_VRAM_GB', prev));

    fake_gb = [8, 20, 48, 96, 200, 2000];        % workstation ... beyond-B200
    N_list  = [12, 20, 30, 36, 67];              % shipped geometries + extreme
    ELEM    = 2^31;

    for gb = fake_gb
        setenv('FTLM_FAKE_FREE_VRAM_GB', sprintf('%g', gb));
        assert(abs(gpu_free_bytes() - gb * 1e9) < 1, 'fake VRAM override broken');
        for N = N_list
            ch = double(adaptive_enum_chunk(N, true));
            % THE invariant that failed on the B200 (2026-07-03):
            assert(N * ch <= 0.95 * ELEM, ...
                'chunk breaks the 2^31 cap: N=%d, fake=%g GB, chunk=%g', N, gb, ch);
            assert(ch >= 4e6 || ch >= floor(ELEM / N * 0.9), ...
                'chunk below floor without cap reason: N=%d, fake=%g GB', N, gb);
            assert(ch <= 256e6, 'chunk above global cap: N=%d, fake=%g GB', N, gb);
        end
    end

    % No-GPU fallback path is capped too (regression: the pre-fix fallback
    % silently restored an uncapped 32e6 -- harmless for 32e6, but the cap
    % must bind on EVERY path).
    for N = N_list
        ch = double(adaptive_enum_chunk(N, false));
        assert(N * ch <= 0.95 * ELEM && ch <= 32e6, 'no-GPU fallback uncapped: N=%d', N);
    end

    % V0 element-cap formula (mirrors run_ftlm's clamp): below the cap the
    % one-shot draw fits; beyond it B clamps to 1 and the SPLIT-V0 path
    % (chunked 'set_v0' uploads, gated by test_split_v0) takes over.
    for n_basis = [1, 1e6, 2^31 - 1, 1.81e9]
        B_elem_cap = max(1, floor((2^31 - 1) / max(n_basis, 1)));
        assert(n_basis * B_elem_cap <= 2^31 - 1, ...
            'V0 cap formula violated at n_basis=%g', n_basis);
    end
    assert(max(1, floor((2^31 - 1) / 2.4e9)) == 1 && 2.4e9 * 1 > 2^31 - 1, ...
        'n_basis > 2^31 must clamp to B=1 and route to split-V0');

    fprintf('  PASS: sizing invariants hold across fake card sizes %s GB.\n', ...
            strjoin(string(fake_gb), '/'));
end
