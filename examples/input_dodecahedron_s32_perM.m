% Physics production deck: dodecahedron N=20, s=3/2 -- ONE M sector per job
% (paper campaign layout; set FTLM_M_SECTOR = 0..30 per job, e.g. from a
% batch-array index). Totals afterwards:
%   merge_observables_pg('runs_dodec_s32_perM/M*/ftlm_pg_gpu_Ih_dodec_s3o2.mat', T)
% (sum rule vs n_total = 4^20 ~ 1.0995e12 once all 31 sectors are merged).
%
% SCALE WARNING: the low-M sectors are heavy (M=0 is ~10^11-dimensional and
% takes many hours even on a data-center GPU); per-irrep checkpointing is ON,
% so a timed-out job can simply be resubmitted and resumes. ed_thresh=0:
% n_total > 2^32, the dense-ED builder cannot run (tiny high-M blocks go
% through the GPU kernel instead). The per-M on-disk entry table (up to
% ~88 GB at M=0, transiently ~2x during the finalize) is per-run scratch --
% point ondisk_scratch_dir at fast node-local storage if output_dir is on a
% shared filesystem. Optional: FTLM_FP16=1 roughly halves the campaign cost
% (see docs/INPUT_REFERENCE.md).
geometry         = 'dodecahedron';
s_val            = 1.5;
J                = 1.0;
R                = 24;
M_lz             = 60;
use_spin_flip    = true;           % acts on the M=0 sector only
ed_thresh        = 0;
lz_diag          = true;
lookup_method    = 'schnack';      % n_total = 4^20 > 2^32 -> mandatory
entries_on_disk  = true;
precompute_cache = false;          % on-disk files are per-run scratch here
checkpoint       = true;

ftlm_m = str2double(getenv('FTLM_M_SECTOR'));
assert(isfinite(ftlm_m) && ftlm_m >= 0 && ftlm_m == round(ftlm_m), ...
    'input_dodecahedron_s32_perM: set the environment variable FTLM_M_SECTOR (0..30).');
M_sectors  = ftlm_m;
T_range    = logspace(-2, 1.5, 60);
output_dir = sprintf('runs_dodec_s32_perM/M%02d', ftlm_m);
