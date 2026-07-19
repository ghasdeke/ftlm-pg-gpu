function assert_kernel_abi()
%ASSERT_KERNEL_ABI  Once-per-session ABI handshake with the CUDA kernel MEX.
%
%   Errors with an actionable message when the compiled
%   CUDA_LANCZOS_CLUT_BLOCK_PG_IH binary predates the 64-bit basis-offset
%   ABI (v2, 2026-07: int64 rep_offsets/entry_offsets). Without this, a
%   stale MEX would read the pipeline's int64 offset arrays as int32 and
%   compute silent garbage.
%
%   Call this before any direct MEX use ('init*' / 'spmv'); the production
%   driver RUN_FTLM_PG_SECTOR_GPU_IH calls it, and the standalone
%   kernel-level tests (test_real_kernel, test_c4v_spmv, ...) should too,
%   since they bypass the driver.
%
%   Only the kernel's own unknown-mode error is interpreted as "old MEX";
%   anything else (no CUDA device, driver mismatch, forward-compat init
%   failure) is rethrown as-is so it is not misdiagnosed as a stale build.
%
%   See also RUN_FTLM_PG_SECTOR_GPU_IH, BUILD_ALL.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    EXPECTED_ABI = 3;   % v3 (2026-07): keep-table modes ('cleanup_keep_table')

    persistent abi_ok
    if ~isempty(abi_ok), return; end

    assert(exist('cuda_lanczos_clut_block_pg_Ih', 'file') == 3, ...
        'MEX file cuda_lanczos_clut_block_pg_Ih not found. Run build_all first.');

    try
        abi_v = cuda_lanczos_clut_block_pg_Ih('abi_version');
    catch err
        if strcmp(err.identifier, 'clut_block_pg_Ih:mode')
            abi_v = 0;   % pre-v2 binary: 'abi_version' is an unknown mode there
        else
            rethrow(err);   % GPU/driver problem, NOT a stale MEX
        end
    end

    % Exact match, not >=: an ABI bump SIGNALS an arg-type change, so a
    % future v3 MEX is just as incompatible with this v2 pipeline as v1 is.
    assert(abi_v == EXPECTED_ABI, ['cuda_lanczos_clut_block_pg_Ih MEX has ABI v%d, ', ...
        'but the .m pipeline expects v%d (64-bit basis offsets, 2026-07): ', ...
        'run build_all to recompile the kernel against the current sources.'], ...
        abi_v, EXPECTED_ABI);

    abi_ok = true;
end
