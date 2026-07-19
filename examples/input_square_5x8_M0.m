% Physics production deck: 5x8 square lattice s=1/2, M=0 sector (space group).
geometry         = 'square_lattice_c4v';
Lx               = 5;
Ly               = 8;
s_val            = 0.5;
J                = 1.0;
R                = 24;
M_lz             = 60;
use_spin_flip    = true;           % M = 0 only: |G| 160 -> 320
ed_thresh        = 0;
lz_diag          = true;
lookup_method    = 'schnack';      % n_total = 2^40 > 2^32 -> mandatory
entries_on_disk  = true;
if ~isempty(getenv('SLURM_JOB_ID'))
    ondisk_scratch_dir = sprintf('/tmp/%s_sq58_od_%s', getenv('USER'), getenv('SLURM_JOB_ID'));
end
precompute_cache = false;          % od files stay per-job scratch (quota)
checkpoint       = true;
M_sectors        = 0;
T_range          = logspace(-2, 1.5, 60);
output_dir       = 'runs_sq58_M0_R24';
