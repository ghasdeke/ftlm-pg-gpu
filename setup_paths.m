function setup_paths()
%SETUP_PATHS  Add the project's source folders to the MATLAB path.
%
%   Call once per MATLAB session before using the pipeline:
%       >> setup_paths
%
%   Adds the project root plus the tests/, examples/ and docs/
%   sub-folders if they exist (the moderate package layout). The core drivers,
%   providers and kernels live in the root and are always on the path. Inputs in
%   examples/ are then found by name by the drivers' run(input_file) call.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
% Licensed under the Apache License, Version 2.0.
% ================================================================
    here = fileparts(mfilename('fullpath'));
    addpath(here);
    subs = {'tests', 'examples', 'docs', 'figures'};
    for k = 1 : numel(subs)
        p = fullfile(here, subs{k});
        if exist(p, 'dir'), addpath(p); end
    end
end
