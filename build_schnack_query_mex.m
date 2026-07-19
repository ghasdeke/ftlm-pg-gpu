function build_schnack_query_mex()
%BUILD_SCHNACK_QUERY_MEX  Compile the fused Schnack query MEX (std::thread).
%
%   Builds SCHNACK_QUERY_MEX from the C++ source. Parallelism uses
%   std::thread (NOT OpenMP) to avoid the MSVC vcomp / MATLAB libiomp5
%   runtime clash that faults at process teardown. On gcc/MinGW the
%   pthread flag is added; MSVC needs no extra flags. After building it
%   prints the hardware-concurrency the kernel will use.
%
%   See also SCHNACK_QUERY_MEX, TEST_SCHNACK_QUERY_MEX.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================

    here = fileparts(mfilename('fullpath'));
    src  = fullfile(here, 'schnack_query_mex.cpp');
    assert(exist(src, 'file') == 2, 'Source not found: %s', src);

    cc = mex.getCompilerConfigurations('C++', 'Selected');
    name = ''; if ~isempty(cc), name = cc(1).Name; end
    fprintf('Selected C++ compiler: %s\n', name);

    is_gcc = contains(name, 'MinGW', 'IgnoreCase', true) || ...
             contains(name, 'GCC',   'IgnoreCase', true) || ...
             contains(name, 'g++',   'IgnoreCase', true);

    args = {'-R2018a', '-O', '-outdir', here};
    if is_gcc
        args = [args, {'CXXFLAGS=$CXXFLAGS -O3 -pthread', 'LDFLAGS=$LDFLAGS -pthread'}];
    end

    mex(args{:}, src);
    rehash;

    info = schnack_query_mex('info');      % [threaded, hardware_concurrency]
    fprintf('Built schnack_query_mex (std::thread, up to %d lanes).\n', info(2));
end
