/*
 * cuda_lanczos_clut_block_pg_Ih.cu
 *
 * Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
 * and Helmholtz-Zentrum Dresden-Rossendorf e.V.
 *
 * Licensed under the Apache License, Version 2.0.
 *
 * ================================================================
 * Matrix-free Heisenberg BLOCK Lanczos on GPU (FP32) — I_h PG variant
 *   for the 12-site icosahedron, all 10 irreducible representations of I_h.
 *
 * The kernel consumes a precomputed compressed lookup table (CLT) built
 * by BUILD_CLT_PG_IH.M: per output rep t a list of incoming entries,
 * each storing the source rep index and the precomputed d_Gamma x d_Gamma
 * matrix-element kernel
 *
 *     M^(e)_{k', k} = sqrt(lambda_t/lambda_i) * c_a *
 *                     [V_t^dag * rho_Gamma(g)^T * V_i]_{k', k}.
 *
 * One CUDA thread is launched per output rep (NOT per basis state); the
 * thread accumulates contributions across its CLT entries and writes
 * n_Gamma(t) complex output values per Krylov column. No runtime
 * min-image search, no compressed bit-mask lookup, no atomicAdds.
 *
 * Two real FP32 vectors V_re, V_im carry the complex Krylov state in
 * interleaved layout V[r * B + b] where r is the row index in the
 * (M, Gamma) basis of length n_basis = sum_i n_Gamma(i) and b is the
 * Krylov-block column index. The Lanczos coefficients alpha and beta
 * are real.
 *
 * Modes:
 *   cuda_lanczos_clut_block_pg_Ih('init',
 *                                  diag_vals_gpu, rep_offsets_gpu,
 *                                  n_per_rep_gpu, entries_per_rep_gpu,
 *                                  entry_offsets_gpu, src_idx_gpu,
 *                                  M_re_gpu, M_im_gpu,
 *                                  n_basis, n_reps, n_entries,
 *                                  d_irrep, B_batch)
 *   [AL, BE] = cuda_lanczos_clut_block_pg_Ih('block_lanczos',
 *                                             V0_re_gpu, V0_im_gpu, M_lz)
 *   cuda_lanczos_clut_block_pg_Ih('cleanup')
 *
 * Compile (from MATLAB):
 *   mexcuda cuda_lanczos_clut_block_pg_Ih.cu
 * ================================================================
 */

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>          /* getenv (Lever A enable flag) */
#include <stdint.h>          /* uintptr_t (mmap pointer unmarshalling). MUST be
                              * explicit: it used to arrive transitively via
                              * cublas_v2.h -- removing cuBLAS (2026-07) broke
                              * the LINUX build only (MSVC includes it anyway;
                              * caught by the in-job node rebuild). */
#include <thread>            /* R3 staging worker. std::thread, NOT OpenMP --
                              * vcomp crashes MATLAB at MEX teardown (repo policy). */
#include <atomic>            /* R3 pipeline progress flags */
#include <chrono>            /* R3 wait timeout (audit K5a: stall -> error, not hang) */
#include <cuda_fp16.h>       /* R1: FP16 STORAGE of the real Lanczos vectors */

#define MAX_D     12           /* max d_Gamma (OTF SpMV path): I_h H_g/H_u = 5;
                                * square C_4v generic-k star = 8; triangular (6,0)
                                * has a d=12 irrep. The OTF kernel keeps only
                                * 2*MAX_D + 2*MAX_B live floats/thread (~40 at
                                * d=12, B=8) so this stays well below a spill. */
#define MAX_D_LEGACY 8         /* the legacy precomputed-M SpMV kernel keeps
                                * result[MAX_D_LEGACY*MAX_B] per thread, so it must
                                * stay at 8 (d=12 would spill 192 floats/thread).
                                * It only runs on the small-system 'init'/'init_ref'
                                * paths (s_use_otf == false), never the d>8 OTF path. */
#define MAX_B     8            /* batched Lanczos block size (kept at the
                                 * release-paper default; pulling this from
                                 * 16 down to 8 frees ~ 80 registers per
                                 * SpMV thread, eliminating local-memory
                                 * spills on d = 4, 5 irreps). */
#define SPMV_BS   64           /* block size: LEGACY (precomputed-M) SpMV */

/* d-dependent OTF block size (wave-2 [12]): lcm(d, 32) removes the idle
 * lanes of a fixed 64 (6.25% at d=5/6/10/12, 14% at d=11) WITHOUT touching
 * any per-thread arithmetic -- each thread's (t, kp) work is independent,
 * so blockDim only re-partitions the same index set (bit-gates verify).
 * Floor 64 keeps d = 1/2/4/8 exactly as before. Max: 352 at d = 11. */
static int otf_block_size(int d)
{
    int g = d, b = 32, t;
    while (b) { t = g % b; g = b; b = t; }   /* g = gcd(d, 32) */
    int bs = d * (32 / g);                   /* lcm(d, 32) */
    return bs < 64 ? 64 : bs;
}
static int s_otf_bs = 64;   /* set per run in block_lanczos/spmv */
#define FUSED_BS  256          /* block size for the Lanczos helpers */

/* ================================================================
 * Constant memory: irrep dimension (per init)
 * ================================================================ */
__constant__ int c_d_irrep;

/* ================================================================
 * Persistent device state
 * ================================================================ */

/* CLT-side */
static float        *s_d_diag_vals       = NULL;
static long long    *s_d_rep_offsets     = NULL;   /* int64: basis offsets reach n_basis > 2^31 */
static int          *s_d_n_per_rep       = NULL;
static int          *s_d_entries_per_rep = NULL;
static long long    *s_d_entry_offsets   = NULL;   /* int64: offsets reach n_entries > 2^31 (B2) */
static int          *s_d_src_idx         = NULL;
static const unsigned int *s_d_srcg      = NULL;   /* packed src(25b)|g(7b) (G2); NULL otherwise */
static const unsigned char *s_d_triv     = NULL;   /* per-rep trivial-stabiliser flag (#1); NULL otherwise */
static const int           *s_d_v_slot   = NULL;   /* per-rep V/sqrt slot (D2 compact-V); NULL -> rep==slot */
static const float         *s_d_Qbar_re  = NULL;   /* per-g reduced d x d block (#1): triv-triv fast path */
static const float         *s_d_Qbar_im  = NULL;
static float        *s_d_M_re            = NULL;
static float        *s_d_M_im            = NULL;
/* When true, s_d_M_re/s_d_M_im point into MATLAB-owned gpuArrays
 * (set via the 'init_ref' mode). cleanup_all must NOT cudaFree them
 * in that case - ownership stays on the MATLAB side. Set to false by
 * the legacy 'init' mode that owns its own cudaMalloc'd copies. */
static bool          s_M_is_ref          = false;

/* On-the-fly (otf) SpMV buffers: kept persistent so the SpMV kernel
 * can recompute the d x d matrix-element kernel on-the-fly per CLT
 * entry instead of loading a precomputed M tensor. Saves ~ 2 GB
 * VRAM on s = 2 H_g/M = 0 and reduces SpMV memory traffic
 * substantially. */
/* tgt_idx removed (Stufe 1a, May 2026): the per-(rep, k') OTF kernel
 * derives the target rep from the thread/block index, so the n_entries-
 * sized tgt_idx array is dead weight. Saves 320-640 MB per sector at
 * icosidodecahedron scale. The static is kept (NULL) for binary
 * compatibility with any leftover code that references it. */
static int          *s_d_tgt_idx         = NULL;
static unsigned short *s_d_g_idx         = NULL;   /* uint16 group index 1..|G| (G1; widened from uint8 to allow |G|>255) */
static unsigned char *s_d_g_idx8         = NULL;   /* G1b: uint8 g (|G|<=255, B2-resident); NULL -> use uint16 s_d_g_idx */
static float        *s_d_c_a_vec         = NULL;     /* may be NULL when c_a is constant */
static float         s_c_a_const         = 0.0f;     /* used when s_d_c_a_vec == NULL */
static const unsigned char *s_d_c_idx    = NULL;     /* uint8 c-index (s>1/2); NULL otherwise */
static const float         *s_d_c_table  = NULL;     /* distinct c values indexed by c_idx */
static float        *s_d_V_re_all        = NULL;
static float        *s_d_V_im_all        = NULL;
static float        *s_d_rho_re_all      = NULL;
static float        *s_d_rho_im_all      = NULL;
static float        *s_d_sqrt_eig_all    = NULL;
static bool          s_use_otf           = false;
static bool          s_b2                = false;   /* B2: per-entry buffers cudaMalloc'd from host (owned) */

/* Real FP32 path (June 2026): when the irreps are REAL (realified space-group
 * irreps or realified I_h irreps), MATLAB passes EMPTY V_im / rho_im / Qbar_im
 * and the init_skel_* modes set s_is_real. The SpMV/Lanczos then run a real
 * fork: half the Krylov buffers (v/vp/w _re only), half the random V[src]
 * gather traffic, and block_lanczos takes ONE V0 (prhs[2] empty). With exact-
 * zero imaginary inputs the complex kernel is exactly imaginary-free in IEEE
 * FP32, so the real fork is gated BIT-identically against it (test_real_kernel)
 * before production halves the RNG draws. The complex path is untouched. */
static bool          s_is_real           = false;

/* Streaming-B2 (June 2026): the ~11.7 GB per-entry src/g (or packed srcg)
 * arrays stay on the (pinned) HOST; each SpMV streams the entries in rep-tiles
 * into small device buffers, so VRAM holds only the Lanczos vectors + per-rep
 * tensors + one tile. Entries are sorted by target rep, so a contiguous rep
 * range maps to a contiguous entry slice; each rep is fully processed inside
 * exactly one tile -> the per-rep OTF kernel writes W[t] once, NO atomics, NO
 * cross-tile accumulation. The host arrays are borrowed (caller keeps clt
 * alive); the device tile buffers + partition copies are owned (freed in
 * cleanup when s_stream). */
static bool          s_stream            = false;
static const int          *s_h_src_idx   = NULL;   /* host (borrowed) int32 src (unpacked path) */
static const unsigned short*s_h_g_idx    = NULL;   /* host (borrowed) uint16 g (unpacked path) */
static const unsigned char *s_h_g_idx8   = NULL;   /* host (borrowed) uint8 g (G1b, |G|<=255 streaming) */
static const unsigned int *s_h_srcg      = NULL;   /* host (borrowed) uint32 packed src|g */
static int          *s_d_src_tile        = NULL;   /* device tile buffer (int32),  owned */
static unsigned short *s_d_g_tile        = NULL;   /* device tile buffer (uint16), owned */
static unsigned char *s_d_g_tile8        = NULL;   /* device tile buffer (uint8, G1b), owned */
static unsigned int  *s_d_srcg_tile      = NULL;   /* device tile buffer (uint32), owned */
static const unsigned char *s_h_c_idx    = NULL;   /* host (borrowed) uint8 c-index (s>=1 streaming) */
static unsigned char *s_d_c_idx_tile     = NULL;   /* device tile buffer (uint8), owned (s>=1 streaming) */
static int       *s_h_tile_rep_ptr = NULL;   /* host OWNED copy: n_tiles+1 rep boundaries (0-based) */
static long long *s_h_tile_e_start = NULL;   /* host OWNED copy: n_tiles entry base offsets */
static long long *s_h_tile_e_count = NULL;   /* host OWNED copy: n_tiles entry counts */
static int           s_n_tiles           = 0;

/* Lever A (June 2026): pinned + double-buffered streaming overlap. The borrowed
 * host entry table is copied ONCE into our OWN cudaHostAlloc'd (page-locked)
 * buffers (s_own_pin_*) and the s_h_* pointers are repointed at them; a 2nd
 * ping-pong device tile set + a copy/compute NON-BLOCKING stream pair + per-buffer
 * ready/free events then let the H2D copy of tile k+1 overlap the SpMV of tile k
 * (serial T_copy+T_compute -> max(T_copy,T_compute)). We NEVER cudaHostRegister
 * borrowed MATLAB memory: registering its page-aligned superset locks foreign
 * heap pages and corrupts MATLAB's heap (the prior 0xc0000409 crash). We own
 * these buffers start to finish. Best-effort: any alloc failure, a single tile,
 * an mmap'd (file-backed) source, or FTLM_LEVER_A != "1" -> stay on the
 * synchronous single-buffer path (s_dbuf = false). Default OFF (dormant). */
static int            *s_d_src_tile2     = NULL;
static unsigned short *s_d_g_tile2       = NULL;
static unsigned char  *s_d_g_tile8_2     = NULL;
static unsigned int   *s_d_srcg_tile2    = NULL;
static unsigned char  *s_d_c_idx_tile2   = NULL;
static cudaStream_t    s_copy_stream     = NULL;
static cudaStream_t    s_compute_stream  = NULL;
static cudaEvent_t     s_tile_ready[2]   = { NULL, NULL };  /* tile-copy-into-buf done */
static cudaEvent_t     s_buf_free[2]     = { NULL, NULL };  /* kernel done reading buf */
static cudaEvent_t     s_stream_done     = NULL;            /* pre/post barrier vs default stream */
static bool            s_dbuf            = false;   /* double-buffered streaming active */
static bool            s_h_is_mmap       = false;   /* streaming host ptrs are mmap'd file pages */
static void           *s_own_pin_src     = NULL;    /* OWN pinned copies (cudaFreeHost at cleanup) */
static void           *s_own_pin_g       = NULL;
static void           *s_own_pin_srcg    = NULL;
static void           *s_own_pin_cidx    = NULL;

/* R3 (2026-07): mmap-compatible double-buffered streaming.
 * Lever A is excluded for mmap sources because it copies+PINS the whole tail
 * (defeats the on-disk design). R3 instead stages tile-by-tile through TWO
 * own pinned ring buffers (bounded: 2 x tile_cap entries, ~1 GB): a CPU
 * worker thread pulls mmap pages -> pinned (this is where the page-cache /
 * NVMe read happens, overlapped), the copy stream DMAs pinned -> device tile
 * (full PCIe speed instead of the ~3.6 GB/s pageable-memcpy path), and the
 * compute stream runs the tile kernels. Tiles, buffers and launch order are
 * IDENTICAL to the synchronous path -> bit-identical (test_r3_stream gates it).
 * Env-gated: FTLM_R3=1 (+ FTLM_R3_THREADS=n memcpy slices; Default = min(8, Kerne/2)). */
static bool  s_r3 = false;
static void *s_r3_pin_src[2]  = { NULL, NULL };
static void *s_r3_pin_g[2]    = { NULL, NULL };   /* uint16 OR uint8 layout */
static void *s_r3_pin_srcg[2] = { NULL, NULL };
static void *s_r3_pin_cidx[2] = { NULL, NULL };
static int   s_r3_slices      = 8;                /* memcpy slice threads per tile.
                                                   * Init setzt hardwareabhaengig min(8, Kerne/2)
                                                   * (empirisch auf B200: 8 optimal DORT); Env
                                                   * FTLM_R3_THREADS 1..8 hat Vorrang. */
static std::atomic<int> s_r3_staged_upto(-1);     /* tail-local j staged by the worker */
static std::atomic<int> s_r3_pin_free_upto(-1);   /* tail-local j whose pin DMA is done */
static std::atomic<int> s_r3_abort(0);            /* R3 stall/CUDA error -> worker exits
                                                   * cleanly (no mex API in the worker!) */
static size_t s_r3_tile_bytes = 0;                /* groesstes Tile in Bytes (Timeout-Basis) */

/* R1 (2026-07): FP16 STORAGE of the real-path Lanczos vectors
 * (v/vp/w), env-gated FTLM_FP16=1, real irreps only. ALL arithmetic stays
 * FP32 (load->convert, float register accumulation, convert->store); only
 * the vector bytes halve -- that halves the dominant random V[src]-gather
 * traffic AND the vector VRAM (doubles the feasible B). NOT bit-identical
 * to FP32 storage (per-element rounding on store). Validation contract:
 * sum rule exact + observable A/B vs FP32 + exact s=3/2 reference curves.
 * v1 limitation: excluded for the chunked-V0 path (n_basis > 2^31). */
static bool s_fp16 = false;

/* FP16 storage scale: normalized Lanczos components are ~n_basis^-1/2; for
 * n_basis > 2^28 that falls below the smallest NORMAL fp16 value 2^-14, so
 * the vector would be quantised on the SUBNORMAL grid (absolute quantum
 * 2^-24) and the effective storage roundoff would grow like
 * sqrt(n_basis)*2^-24 instead of 2^-11. The store therefore scales by 2^k,
 * the load by 2^-k, with k = round(log2 sqrt(n_basis)) per block. Both
 * factors are exact powers of two -> the fp32 multiplies are EXACT (no
 * extra rounding); only the fp16 quantisation grid moves. env
 * FTLM_FP16_SCALE=0 forces k = 0 (the unscaled grid, bit-identical to
 * scale-free storage). RAW (unnormalised) V0 draws are pre-scaled by 2^-k
 * at ingestion (scatter_v0_col / transpose_col2interleaved 'presc'
 * argument): the stored start-vector bytes are then independent of k, and
 * the 2^-k on the logical v0 cancels exactly in the first normalisation
 * (power-of-two scaling commutes exactly through dot, sqrt and divide;
 * beta0 never leaves the MEX -- it is consumed by the normalisation). */
__constant__ float c_vt_sc  = 1.0f;   /* 2^k  applied on store */
__constant__ float c_vt_isc = 1.0f;   /* 2^-k applied on load  */
static float s_vt_presc = 1.0f;       /* 2^-k host-side, raw-V0 ingestion */
static int   s_vt_k     = 0;

/* Storage-type overloads: identical codegen to the direct load/store for
 * float (ld_vec == __ldg, st_vec == plain store, qz_vec == identity), so
 * the FP32 instantiations keep the bit-identity gates green. qz_vec
 * quantises a value to its STORED representation -- the ortho/norm kernel
 * uses it so beta is computed from exactly the vector that is stored.
 * The scale multiplies are __fmul_rn ON PURPOSE: powers of two make them
 * exact, and a plain '*' would let ptxas re-decide FMA contraction in the
 * surrounding expressions -- __fmul_rn pins the operand as one
 * un-contractible value, so FTLM_FP16_SCALE=0 reproduces the unscaled
 * storage grid bit-exactly and the outer rounding sequences are stable. */
__device__ __forceinline__ float ld_vec(const float  *p) { return __ldg(p); }
__device__ __forceinline__ float ld_vec(const __half *p) { return __fmul_rn(__half2float(__ldg(p)), c_vt_isc); }
__device__ __forceinline__ void  st_vec(float  *p, float v) { *p = v; }
__device__ __forceinline__ void  st_vec(__half *p, float v) { *p = __float2half_rn(__fmul_rn(v, c_vt_sc)); }
__device__ __forceinline__ float qz_vec(float v, const float  *) { return v; }
__device__ __forceinline__ float qz_vec(float v, const __half *) { return __fmul_rn(__half2float(__float2half_rn(__fmul_rn(v, c_vt_sc))), c_vt_isc); }

/* d-Templating (Klasse A, 2026-07-10): the OTF SpMV loops over the
 * irrep dimension d, which so far was a RUNTIME constant (c_d_irrep) -- so
 * nvcc could neither unroll the m/n/k loops nor size the V_t register cache
 * exactly (it always burned MAX_D registers). The kernel now carries d as a
 * template parameter DT for the production dimensions; DT = 0 keeps the
 * exact historical runtime path as the fallback for other d. Arithmetic and
 * its order are UNCHANGED -- same values in a different schedule -> the
 * bit-identity gates certify equality for free. Compile-time block size for
 * __launch_bounds__ mirrors otf_block_size(): lcm(d, 32), floor 64. */
/* R2-v1 (getilter SpMV, 2026-07-10): Entries liegen (src_tile, tgt)-
 * sortiert vor (BUILD_TILED_ENTRIES); pro Tile bilden gleiche tgt RUNS.
 * Ein Launch pro src-Tile: die V[src]-Gathers eines Launches treffen nur das
 * L2-residente Fenster des Tiles -> der latenzgebundene Random-Gather wird
 * zum Cache-Hit. W wird ADDITIV akkumuliert (Diag-Pass zuerst); pro Tile hat
 * jedes tgt hoechstens einen Run und Tiles laufen sequenziell -> keine
 * Atomics, deterministische Summationsreihenfolge (Klasse C gegenueber der
 * historischen Bond-Ordnung; Sum Rule bleibt exakt). v1: resident real path. */
static bool             s_tiled        = false;
static const long long *s_d_run_ptr    = NULL;   /* [n_runs+1] borrowed gpuArray */
static const int       *s_d_run_tgt    = NULL;   /* [n_runs]   borrowed gpuArray */
static long long       *s_h_tile_runp  = NULL;   /* [n_tiles+1] OWNED host copy */
static int              s_n_tiles_r2   = 0;

constexpr int otf_gcd_ct(int a, int b) { return b == 0 ? a : otf_gcd_ct(b, a % b); }
constexpr int otf_bs_ct(int d) {
    return d <= 0 ? 352
                  : ((d * 32 / otf_gcd_ct(d, 32)) < 64 ? 64
                                                       : d * 32 / otf_gcd_ct(d, 32));
}

/* Resident-prefix streaming (June 2026): keep the LEADING tiles of the streamed
 * entry table permanently on the GPU (copied ONCE at init, sized to the VRAM
 * left over after the Lanczos buffers), and stream only the TAIL tiles per
 * SpMV. The per-SpMV PCIe traffic shrinks from the full table to the tail
 * (e.g. kagome N=36 d=3: ~11.5 GB -> ~1 GB). Entries are sorted by target rep
 * and the partition is per-tile, so a whole-tile prefix is a contiguous entry
 * range [0, s_pref_e_count) covering reps [0, s_pref_rep_hi) -> ONE extra
 * kernel launch with entry_base = 0, no other indexing change. Best-effort:
 * any alloc/copy failure -> prefix off, full streaming (today's path). When
 * Lever A is also active it pins/copies ONLY the tail (s_pin_base_e shifts the
 * borrowed-host indexing), so the pinned host cost shrinks identically. */
static int             s_pref_tiles      = 0;       /* tiles 0..P-1 resident */
static int             s_pref_rep_hi     = 0;       /* prefix covers reps [0, rep_hi) */
static long long       s_pref_e_count    = 0;       /* prefix covers entries [0, e_count) */
static int            *s_d_src_pref      = NULL;    /* owned device prefix buffers */
static unsigned short *s_d_g_pref        = NULL;
static unsigned char  *s_d_g8_pref       = NULL;
static unsigned int   *s_d_srcg_pref     = NULL;
static unsigned char  *s_d_cidx_pref     = NULL;
static long long       s_pin_base_e      = 0;       /* s_h_* hold entries [base, n) (Lever A tail pin) */

/* Keep-table (July 2026, wave-2 [13]): 'cleanup_keep_table' preserves the
 * irrep-INDEPENDENT streaming table state (tile partition + device tile
 * buffers + resident prefix + Lever-A pinned copies) across the per-irrep
 * init/cleanup cycle of one M sector. The next init_skel_stream matches the
 * incoming table against the stored fingerprint (ORIGINAL host/mmap pointers
 * as passed by MATLAB + the full tile partition + n_reps) and skips the
 * table setup -- at dodec s=3/2 on B200 (86-GB mmap table fully prefix-
 * resident) that is the entire per-irrep table re-read from disk. ANY
 * mismatch -> full cleanup + fresh init; correctness never depends on a
 * match, only speed does. */
static bool               s_kept_table       = false;
static bool               s_last_init_reused = false;   /* test observability */
static unsigned long long s_fp_src = 0, s_fp_g = 0, s_fp_srcg = 0, s_fp_cidx = 0;
static int                s_fp_nreps = 0;
/* Ownership of the skel arrays. When true, all the s_d_* skeleton
 * pointers (src_idx, g_idx, c_a_vec, V_*, rho_*, sqrt_eig, diag_vals,
 * rep_offsets, n_per_rep, entries_per_rep, entry_offsets) point into
 * MATLAB-owned gpuArrays; cleanup_all must NOT cudaFree them. Set true
 * by the 'init_skel_ref' mode. */
static bool          s_skel_is_ref       = false;

/* Lanczos buffers (interleaved [n_basis x B]) */
static float        *s_d_v_re   = NULL, *s_d_v_im   = NULL;
static float        *s_d_vp_re  = NULL, *s_d_vp_im  = NULL;
static float        *s_d_w_re   = NULL, *s_d_w_im   = NULL;

/* Reduction buffers */
static float        *s_d_partial   = NULL;
static float        *s_d_alpha     = NULL;
static float        *s_d_beta      = NULL;
static float        *s_d_beta_prev = NULL;
static int           s_n_reduce_blocks = 0;

/* Sector dimensions */
static long long s_n_basis = 0;         /* 64-bit: n_basis = d*n_reps can exceed 2^31 */
static int s_n_reps   = 0;              /* rep indices stay int32 (n_reps <= 2^31-1, guarded) */
static long long s_n_entries = 0;       /* can exceed 2^31 (B2 / N=36) */
static int s_d_irrep  = 0;
static int s_B_batch  = 0;
static bool s_init    = false;

/* FP16 storage-scale init: called at every buffer-init site right after
 * s_fp16 is decided. k is a pure function of n_basis, so a resumed or
 * repeated block reproduces the same grid deterministically. The symbols
 * are set even for k = 0 / fp32 so a later run in the same MEX session
 * cannot inherit a stale scale. */
static void init_vt_scale(void)
{
    s_vt_k = 0;
    if (s_fp16) {
        const char *sc = getenv("FTLM_FP16_SCALE");
        if (sc == NULL || sc[0] != '0') {
            s_vt_k = (int)floor(0.5 * log2((double)s_n_basis) + 0.5);
            if (s_vt_k < 0)  s_vt_k = 0;
            if (s_vt_k > 20) s_vt_k = 20;   /* stored |v| stays orders below fp16-max 65504 */
        }
        mexPrintf("[FP16] storage scale 2^%d (n_basis = %lld%s).\n",
                  s_vt_k, s_n_basis,
                  (sc != NULL && sc[0] == '0') ? "; disabled via FTLM_FP16_SCALE=0" : "");
    }
    float sc2  = ldexpf(1.0f,  s_vt_k);
    float isc2 = ldexpf(1.0f, -s_vt_k);
    s_vt_presc = isc2;
    cudaMemcpyToSymbol(c_vt_sc,  &sc2,  sizeof(float));
    cudaMemcpyToSymbol(c_vt_isc, &isc2, sizeof(float));
}

/* ================================================================
 * 64-bit basis-offset ABI (July 2026 audit): rep_offsets is int64 so
 * n_basis = d*n_reps may exceed 2^31 (icosahedron s=5, square 6x7).
 * Rep INDICES (src_idx, thread ids) remain int32 -> n_reps is capped.
 * The two helpers below fail LOUDLY on the two ways this can go wrong.
 * ================================================================ */
#define KERNEL_ABI_VERSION 3   /* bumped when the MEX arg/mode surface changes (v3: keep-table modes) */

/* Version-skew tripwire: an OLD .m pipeline would pass int32 offsets
 * (silently read as garbage int64), a NEW pipeline against an old MEX
 * fails via the 'abi_version' handshake in assert_kernel_abi.m. Checked
 * for BOTH rep_offsets and entry_offsets (the kernel strides both as
 * long long; an int32 array would be read 2x past its end). */
static void check_offsets_int64(const mxGPUArray *g, const char *name)
{
    if (mxGPUGetClassID(g) != mxINT64_CLASS)
        mexErrMsgIdAndTxt("clut_block_pg_Ih:abi",
            "%s must be int64 (64-bit basis-offset ABI v%d). "
            "Kernel MEX and MATLAB pipeline are out of sync -- run build_all.",
            name, KERNEL_ABI_VERSION);
}

/* n_reps must stay below the int32 rep-index cap (src_idx, thread ids).
 * Cap is INT_MAX - 256, not INT_MAX: grid-size math (n_reps + SPMV_BS - 1)
 * and the padding threads' t = blockIdx.x*reps_per_block + rep_in_block
 * (computed BEFORE the range guard) must not overflow int either. */
static int checked_nreps(double v)
{
    if (!(v >= 0.0) || v > 2147483391.0)
        mexErrMsgIdAndTxt("clut_block_pg_Ih:nreps",
            "n_reps = %.0f exceeds the int32 rep-index cap (2^31-257). "
            "Basis offsets are 64-bit, but rep indices are not -- this "
            "system needs a src_idx/thread-index widening first.", v);
    return (int)v;
}

static void cleanup_core(bool keep_stream_table) {
    /* keep: preserve the streaming-table state (see s_kept_table above). */
    bool keep = keep_stream_table && s_stream;
    /* Skeleton-side: only cudaFree what we own. When s_skel_is_ref is
     * true, the pointers point into MATLAB-owned gpuArrays; freeing
     * them would corrupt MATLAB state and crash. */
    if (!s_skel_is_ref) {
        if (s_d_diag_vals)       cudaFree(s_d_diag_vals);
        if (s_d_rep_offsets)     cudaFree(s_d_rep_offsets);
        if (s_d_n_per_rep)       cudaFree(s_d_n_per_rep);
        if (s_d_entries_per_rep) cudaFree(s_d_entries_per_rep);
        if (s_d_entry_offsets)   cudaFree(s_d_entry_offsets);
        if (s_d_src_idx)         cudaFree(s_d_src_idx);
        if (s_d_g_idx)           cudaFree(s_d_g_idx);
        if (s_d_c_a_vec)         cudaFree(s_d_c_a_vec);
        if (s_d_V_re_all)        cudaFree(s_d_V_re_all);
        if (s_d_V_im_all)        cudaFree(s_d_V_im_all);
        if (s_d_rho_re_all)      cudaFree(s_d_rho_re_all);
        if (s_d_rho_im_all)      cudaFree(s_d_rho_im_all);
        if (s_d_sqrt_eig_all)    cudaFree(s_d_sqrt_eig_all);
    }
    /* B2: the per-entry buffers are cudaMalloc'd-from-host (owned by us) even
     * though the per-rep arrays are borrowed (s_skel_is_ref true). Free them. */
    if (s_b2) {
        if (s_d_src_idx) cudaFree(s_d_src_idx);
        if (s_d_srcg)    cudaFree((void *)s_d_srcg);
        if (s_d_g_idx)   cudaFree(s_d_g_idx);
        if (s_d_g_idx8)  cudaFree(s_d_g_idx8);
        if (s_d_c_a_vec) cudaFree(s_d_c_a_vec);
        if (s_d_c_idx)   cudaFree((void *)s_d_c_idx);
    }
    /* Streaming-B2: free the OWNED device tile buffers. The per-entry host
     * arrays (s_h_src_idx/g_idx/srcg) and the tile-partition host arrays are
     * BORROWED from MATLAB-owned mxArrays -> not freed here. */
    if (s_stream && !keep) {
        if (s_d_src_tile)   cudaFree(s_d_src_tile);
        if (s_d_g_tile)     cudaFree(s_d_g_tile);
        if (s_d_g_tile8)    cudaFree(s_d_g_tile8);
        if (s_d_srcg_tile)  cudaFree(s_d_srcg_tile);
        if (s_d_c_idx_tile) cudaFree(s_d_c_idx_tile);   /* s>=1 streaming c-index tile */
        /* Lever A: 2nd ping-pong tiles + streams/events + OWN pinned host copies. */
        if (s_d_src_tile2)   cudaFree(s_d_src_tile2);
        if (s_d_g_tile2)     cudaFree(s_d_g_tile2);
        if (s_d_g_tile8_2)   cudaFree(s_d_g_tile8_2);
        if (s_d_srcg_tile2)  cudaFree(s_d_srcg_tile2);
        if (s_d_c_idx_tile2) cudaFree(s_d_c_idx_tile2);
        if (s_copy_stream)    cudaStreamDestroy(s_copy_stream);
        if (s_compute_stream) cudaStreamDestroy(s_compute_stream);
        if (s_tile_ready[0])  cudaEventDestroy(s_tile_ready[0]);
        if (s_tile_ready[1])  cudaEventDestroy(s_tile_ready[1]);
        if (s_buf_free[0])    cudaEventDestroy(s_buf_free[0]);
        if (s_buf_free[1])    cudaEventDestroy(s_buf_free[1]);
        if (s_stream_done)    cudaEventDestroy(s_stream_done);
        if (s_own_pin_src)  cudaFreeHost(s_own_pin_src);
        if (s_own_pin_g)    cudaFreeHost(s_own_pin_g);
        if (s_own_pin_srcg) cudaFreeHost(s_own_pin_srcg);
        if (s_own_pin_cidx) cudaFreeHost(s_own_pin_cidx);
        /* R3 pinned ring staging buffers. */
        for (int b = 0; b < 2; b++) {
            if (s_r3_pin_src[b])  cudaFreeHost(s_r3_pin_src[b]);
            if (s_r3_pin_g[b])    cudaFreeHost(s_r3_pin_g[b]);
            if (s_r3_pin_srcg[b]) cudaFreeHost(s_r3_pin_srcg[b]);
            if (s_r3_pin_cidx[b]) cudaFreeHost(s_r3_pin_cidx[b]);
        }
        /* Resident-prefix: owned device copies of the leading tiles. */
        if (s_d_src_pref)   cudaFree(s_d_src_pref);
        if (s_d_g_pref)     cudaFree(s_d_g_pref);
        if (s_d_g8_pref)    cudaFree(s_d_g8_pref);
        if (s_d_srcg_pref)  cudaFree(s_d_srcg_pref);
        if (s_d_cidx_pref)  cudaFree(s_d_cidx_pref);
    }
    if (s_d_M_re && !s_M_is_ref) cudaFree(s_d_M_re);
    if (s_d_M_im && !s_M_is_ref) cudaFree(s_d_M_im);

    /* tgt_idx: defensively free in case some old code path allocated
     * it. The Stufe-1a refactor stops allocating it, but old MEX
     * artefacts in memory might still hold one. */
    if (s_d_tgt_idx && !s_skel_is_ref) cudaFree(s_d_tgt_idx);

    /* Lanczos / reduction buffers are always owned by us. */
    if (s_d_v_re)            cudaFree(s_d_v_re);
    if (s_d_v_im)            cudaFree(s_d_v_im);
    if (s_d_vp_re)           cudaFree(s_d_vp_re);
    if (s_d_vp_im)           cudaFree(s_d_vp_im);
    if (s_d_w_re)            cudaFree(s_d_w_re);
    if (s_d_w_im)            cudaFree(s_d_w_im);
    if (s_d_partial)         cudaFree(s_d_partial);
    if (s_d_alpha)           cudaFree(s_d_alpha);
    if (s_d_beta)            cudaFree(s_d_beta);
    if (s_d_beta_prev)       cudaFree(s_d_beta_prev);
    s_d_diag_vals = NULL; s_d_rep_offsets = NULL;
    s_d_n_per_rep = NULL; s_d_entries_per_rep = NULL;
    s_d_entry_offsets = NULL; s_d_src_idx = NULL;
    s_d_M_re = NULL; s_d_M_im = NULL; s_M_is_ref = false;
    s_d_tgt_idx = NULL; s_d_g_idx = NULL; s_d_g_idx8 = NULL; s_d_c_a_vec = NULL;
    s_c_a_const = 0.0f;
    s_d_c_idx = NULL; s_d_c_table = NULL;
    s_d_srcg = NULL;
    s_d_triv = NULL; s_d_Qbar_re = NULL; s_d_Qbar_im = NULL; s_d_v_slot = NULL;
    s_d_V_re_all = NULL; s_d_V_im_all = NULL;
    s_d_rho_re_all = NULL; s_d_rho_im_all = NULL;
    s_d_sqrt_eig_all = NULL;
    s_skel_is_ref = false;
    s_d_v_re = NULL; s_d_v_im = NULL;
    s_d_vp_re = NULL; s_d_vp_im = NULL;
    s_d_w_re = NULL; s_d_w_im = NULL;
    s_d_partial = NULL; s_d_alpha = NULL;
    s_d_beta = NULL; s_d_beta_prev = NULL;
    s_init = false; s_use_otf = false; s_b2 = false;
    s_is_real = false;
    s_fp16 = false;          /* R1 reset */
    /* R2-v1 reset (per-irrep skel state: immer zuruecksetzen; run_ptr/
     * run_tgt sind OWNED cudaMalloc-Kopien, tile_runp OWNED host). */
    if (s_d_run_ptr) cudaFree((void *)s_d_run_ptr);
    if (s_d_run_tgt) cudaFree((void *)s_d_run_tgt);
    if (s_h_tile_runp) mxFree(s_h_tile_runp);
    s_h_tile_runp = NULL; s_d_run_ptr = NULL; s_d_run_tgt = NULL;
    s_tiled = false; s_n_tiles_r2 = 0;
    s_otf_bs = 64;
    if (!keep) {
        /* Streaming-B2 reset. The tile-partition arrays are host-OWNED copies
         * (mxMalloc'd in init_skel_stream) -> free them. The big per-entry host
         * arrays (s_h_src_idx/g_idx/srcg) are borrowed -> just drop the pointers. */
        if (s_h_tile_rep_ptr) mxFree(s_h_tile_rep_ptr);
        if (s_h_tile_e_start) mxFree(s_h_tile_e_start);
        if (s_h_tile_e_count) mxFree(s_h_tile_e_count);
        s_stream = false; s_n_tiles = 0;
        s_h_src_idx = NULL; s_h_g_idx = NULL; s_h_g_idx8 = NULL; s_h_srcg = NULL;  s_h_c_idx = NULL;
        s_d_src_tile = NULL; s_d_g_tile = NULL; s_d_g_tile8 = NULL; s_d_srcg_tile = NULL;  s_d_c_idx_tile = NULL;
        /* Lever A reset. */
        s_d_src_tile2 = NULL; s_d_g_tile2 = NULL; s_d_g_tile8_2 = NULL; s_d_srcg_tile2 = NULL; s_d_c_idx_tile2 = NULL;
        s_copy_stream = NULL; s_compute_stream = NULL;
        s_tile_ready[0] = NULL; s_tile_ready[1] = NULL;
        s_buf_free[0] = NULL; s_buf_free[1] = NULL; s_stream_done = NULL;
        s_dbuf = false; s_h_is_mmap = false;
        s_own_pin_src = NULL; s_own_pin_g = NULL; s_own_pin_srcg = NULL; s_own_pin_cidx = NULL;
        /* R3 reset. */
        s_r3 = false;
        for (int b = 0; b < 2; b++) {
            s_r3_pin_src[b] = NULL; s_r3_pin_g[b] = NULL;
            s_r3_pin_srcg[b] = NULL; s_r3_pin_cidx[b] = NULL;
        }
        s_h_tile_rep_ptr = NULL; s_h_tile_e_start = NULL; s_h_tile_e_count = NULL;
        /* Resident-prefix reset. */
        s_pref_tiles = 0; s_pref_rep_hi = 0; s_pref_e_count = 0;
        s_d_src_pref = NULL; s_d_g_pref = NULL; s_d_g8_pref = NULL;
        s_d_srcg_pref = NULL; s_d_cidx_pref = NULL;
        s_pin_base_e = 0;
        /* Keep-table reset: no table survives a full cleanup. */
        s_kept_table = false;
        s_fp_src = 0; s_fp_g = 0; s_fp_srcg = 0; s_fp_cidx = 0; s_fp_nreps = 0;
    } else {
        s_kept_table = true;   /* s_stream stays true: mexAtExit still frees */
    }
    /* An EXTERNAL device reset (the driver does reset(gpu_h) at startup;
     * tests may too) destroys the CUDA context UNDER any state these statics
     * still point at -- e.g. a KEPT streaming table from a previous run. The
     * cudaFree/cudaEventDestroy calls above then return cudaErrorInvalidValue
     * for every such pointer. That is harmless (the reset already released
     * the memory; the statics are NULLed above either way), but the SOFT
     * error must not leak into the next mode's error checks: it surfaced as
     * a bogus "init_skel_ref: CUDA allocation of the Lanczos buffers failed
     * (invalid argument)" when a kept table outlived a reset (keep_table ->
     * entries_on_disk suite sequence, 2026-07-09). Absorb it here, with a
     * note so a REAL teardown failure stays observable. */
    { cudaError_t ce = cudaGetLastError();
      if (ce != cudaSuccess)
          mexPrintf("[cleanup] note: CUDA reported '%s' while dropping kernel "
                    "state (harmless after an external GPU reset; state fully "
                    "dropped either way).\n", cudaGetErrorString(ce)); }
}
static void cleanup_all(void) { cleanup_core(false); }

/* ================================================================
 * Build M-tensor on GPU from the skeleton.
 *
 * One thread per CLT entry. For entry e the thread reads the four
 * skeleton scalars (src, tgt, g_min, c_a), then the small irrep / orbit
 * tensors (V_re, V_im, rho_re, rho_im, sqrt_eig) at the indexed slices,
 * and writes the precomputed d x d matrix-element kernel
 *
 *     M^(e)_{k', k} = c_a * sqrt(lambda_t / lambda_i) *
 *                     [V_t^dag * rho_Gamma(g_min)^T * V_i]_{k', k}
 *
 * directly into the persistent device M_re / M_im buffers in the
 * same layout the SpMV kernel later consumes:
 *
 *     M_re[e * d^2 + k * d + k']
 *
 * (column-major within the d x d slice, matching MATLAB's storage of
 * V_per_rep and the V_re_all / V_im_all uploaded host tensors).
 *
 * This eliminates the host-side O(n_entries * d^2) bandwidth-bound
 * unpack-and-upload step that previously dominated wall time for s = 2
 * H_g/M = 0 (~ 60 s per block). Per thread cost for d = 5 is about
 * 600 floating-point operations; the full M build runs in milliseconds
 * to tens of milliseconds on a modern GPU.
 *
 * Padding convention:
 *   V_re/V_im have zero columns beyond n_per_rep(i) for each slice i;
 *   sqrt_eig has 1 in those slots so that the c_a * sqrt_t / sqrt_r
 *   scaling never divides by zero, and the matrix elements in the
 *   padding region come out exactly zero because of the V zero.
 * ================================================================ */
/* DEAD CODE: no launch site since the OTF rewrite (kept for binary
 * compatibility only). NOTE it is 32-bit-only ('int e', 'e * d2'): do
 * not re-wire it for systems with n_entries * d^2 > 2^31 without
 * widening the flat index to long long first. */
__global__ void build_M_tensor_pg_Ih(
    float       * __restrict__ M_re,
    float       * __restrict__ M_im,
    const int   * __restrict__ src_idx,
    const int   * __restrict__ tgt_idx,
    const unsigned short * __restrict__ g_idx,
    const float * __restrict__ c_a_vec,
    const float * __restrict__ V_re,
    const float * __restrict__ V_im,
    const float * __restrict__ rho_re,
    const float * __restrict__ rho_im,
    const float * __restrict__ sqrt_eig,
    int n_entries)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_entries) return;

    int d  = c_d_irrep;
    int d2 = d * d;

    int src = src_idx[e] - 1;     /* MATLAB 1-based -> C 0-based */
    int tgt = tgt_idx[e] - 1;
    int g   = g_idx[e]   - 1;
    float c_a = c_a_vec[e];

    int V_t_base = tgt * d2;
    int V_r_base = src * d2;
    int rho_base = g   * d2;
    int sqrt_t_base = tgt * d;
    int sqrt_r_base = src * d;
    int M_base   = e   * d2;

    /* For each output element M[k', k] = inner[k', k] * c_a * sqrt_t / sqrt_r */
    for (int kp = 0; kp < d; kp++) {
        float sqrt_t = sqrt_eig[sqrt_t_base + kp];

        for (int k = 0; k < d; k++) {
            float sqrt_r = sqrt_eig[sqrt_r_base + k];

            /* inner[k', k] = sum_{m, n} conj(V_t[m, k']) * rho[m, n] * V_r[n, k] */
            float inner_re = 0.f;
            float inner_im = 0.f;

            for (int m = 0; m < d; m++) {
                /* V_t[m, k']  (target column k', row m) */
                int idx_vt = V_t_base + kp * d + m;
                float vt_re = V_re[idx_vt];
                float vt_im = V_im[idx_vt];
                /* conj(V_t) = (vt_re, -vt_im) */

                for (int n = 0; n < d; n++) {
                    int idx_rho = rho_base + n * d + m;
                    float ro_re = rho_re[idx_rho];
                    float ro_im = rho_im[idx_rho];

                    int idx_vr  = V_r_base + k * d + n;
                    float vr_re = V_re[idx_vr];
                    float vr_im = V_im[idx_vr];

                    /* a = conj(V_t[m, k']) * rho[m, n]
                       = (vt_re - i*vt_im)*(ro_re + i*ro_im)
                       = (vt_re*ro_re + vt_im*ro_im) +
                         i*(vt_re*ro_im - vt_im*ro_re) */
                    float a_re = vt_re * ro_re + vt_im * ro_im;
                    float a_im = vt_re * ro_im - vt_im * ro_re;

                    /* inner += a * V_r[n, k] */
                    inner_re += a_re * vr_re - a_im * vr_im;
                    inner_im += a_re * vr_im + a_im * vr_re;
                }
            }

            float scale = c_a * sqrt_t / sqrt_r;
            M_re[M_base + k * d + kp] = inner_re * scale;
            M_im[M_base + k * d + kp] = inner_im * scale;
        }
    }
}

/* ================================================================
 * SpMV kernel: W = H_block * V (ON-THE-FLY variant, per-(rep, kp)).
 *
 * STATUS: PRODUCTION (May 2026 rewrite). Replaces the previous
 * one-thread-per-rep design that spilled registers for d >= 4.
 *
 * Parallelisation. Each output rep r' is handled by D = d_irrep
 * threads. Thread (r', k') accumulates ONE output row W[base(r')+k', :]
 * for all B input vectors. Per-thread accumulator footprint is just
 * 2 * B floats instead of 2 * d * B floats; with B = 8 and d = 5 the
 * pressure drops from 80 to 16 register floats for the output, plus
 * 2 * d = 10 floats for the V_t[:, k'] cache (loaded once per thread).
 *
 * Thread layout. One CUDA block holds REPS_PER_BLOCK reps:
 *   threadIdx.x in [0, REPS_PER_BLOCK * D).
 *   rep_in_block = threadIdx.x / D
 *   kp           = threadIdx.x % D
 *   t            = blockIdx.x * REPS_PER_BLOCK + rep_in_block
 * Threads with kp >= n_per_rep[t] do nothing (this happens for reps
 * with stabilisers reducing the effective basis dimension below d).
 *
 * Per CLT entry, computes the d x d matrix-element kernel
 *     M^(e)_{k', k} = c_a * sqrt(lambda_t / lambda_i) *
 *                     (V_t^dag * rho_Gamma(g) * V_i)_{k', k}
 * inline. NO M tensor exists in VRAM. V_t[:, k'] sits in registers
 * (one column slice per thread, d=5 complex). rho and V_r are read
 * through __ldg per inner iteration -- L1 broadcast across the D
 * threads of a rep absorbs the redundancy.
 *
 * Memory budget for H_g at s = 1/2 icosidodecahedron scale:
 *   - V_re/im (padded n_super_reps x d^2 single): ~ 260 MB
 *   - sqrt_eig (n_super_reps x d single): ~ 26 MB
 *   - rho (120 x d^2 single complex): ~ 24 KB
 *   - entries (src, tgt, g, c_a): ~ 1.3 GB
 *   - Lanczos vectors: ~ 830 MB
 *   - TOTAL: ~ 2.4 GB  (vs 16 GB for the precomputed-M path)
 * ================================================================ */
__global__ void heisenberg_clut_block_spmv_pg_Ih_otf(
    float       * __restrict__ W_re,
    float       * __restrict__ W_im,
    const float * __restrict__ V_re,
    const float * __restrict__ V_im,
    const float * __restrict__ diag_vals,
    const long long * __restrict__ rep_offsets,  /* int64: basis offsets reach n_basis > 2^31 */
    const int   * __restrict__ n_per_rep,
    const int   * __restrict__ entries_per_rep,
    const long long * __restrict__ entry_offsets,
    const int   * __restrict__ src_idx,
    const unsigned short * __restrict__ g_idx,
    const unsigned char * __restrict__ g_idx8,  /* G1b: uint8 g (|G|<=255); NULL -> use uint16 g_idx */
    const unsigned int  * __restrict__ srcg,    /* packed src(25b)|g(7b) (G2); NULL -> use src_idx/g_idx */
    const float * __restrict__ c_a_vec,    /* may be NULL when c_a is constant */
    float                       c_a_const, /* used when c_a_vec == NULL */
    const unsigned char * __restrict__ c_idx,   /* uint8 index (s>1/2); NULL otherwise */
    const float * __restrict__ c_table,         /* distinct c values indexed by c_idx */
    const float * __restrict__ V_all_re,
    const float * __restrict__ V_all_im,
    const float * __restrict__ rho_all_re,
    const float * __restrict__ rho_all_im,
    const float * __restrict__ sqrt_eig_all,
    const unsigned char * __restrict__ triv,    /* per-rep trivial flag (#1); NULL -> no fast path */
    const float * __restrict__ Qbar_re,         /* per-g reduced d x d block (#1) */
    const float * __restrict__ Qbar_im,
    const int * __restrict__ v_slot,            /* per-rep V/sqrt slot (D2 compact-V); NULL -> rep==slot */
    int n_reps, int B, int reps_per_block,
    int rep_lo, int rep_hi, long long entry_base)
    /* Streaming-B2: process only output reps [rep_lo, rep_hi); src_idx/g_idx/srcg
     * point into a TILE device buffer holding entries [entry_base, ...), so the
     * per-entry index becomes (entry_offsets[t] + ei) - entry_base. For the
     * resident path the caller passes rep_lo=0, rep_hi=n_reps, entry_base=0 ->
     * byte-identical to the pre-streaming kernel. */
{
    int d  = c_d_irrep;
    int d2 = d * d;

    int rep_in_block = threadIdx.x / d;
    int kp           = threadIdx.x % d;
    int t            = rep_lo + blockIdx.x * reps_per_block + rep_in_block;

    /* Guard: out-of-range blocks (last block may be partially active)
     * and oversized thread groups (threads with kp >= d are unused). */
    if (rep_in_block >= reps_per_block) return;
    if (t >= rep_hi) return;

    int n_t = __ldg(&n_per_rep[t]);
    if (kp >= n_t) return;       /* padded thread for this rep */

    long long off_t = __ldg(&rep_offsets[t]);

    /* #1: is the TARGET rep trivial-stabiliser? If so (and Qbar provided) an
     * entry from a trivial SOURCE can use the precomputed per-g block. */
    bool t_triv = (triv != NULL) && (Qbar_re != NULL) && (__ldg(&triv[t]) != 0);

    /* V_t[:, kp] cached in registers: d floats real + d floats imag.
     * D2: V/sqrt are indexed through v_slot (the ~all trivial-stabiliser reps
     * share ONE slot) when provided; else slot == rep (full per-rep V_all). */
    int vt_rep   = (v_slot != NULL) ? v_slot[t] : t;
    long long V_t_base = (long long)vt_rep * d2;   /* V_all is d^2*n_reps floats: 64-bit */
    float vt_re_local[MAX_D];
    float vt_im_local[MAX_D];
    for (int m = 0; m < d; m++) {
        vt_re_local[m] = __ldg(&V_all_re[V_t_base + kp * d + m]);
        vt_im_local[m] = __ldg(&V_all_im[V_t_base + kp * d + m]);
    }
    float sqrt_t = __ldg(&sqrt_eig_all[(long long)vt_rep * d + kp]);   /* d*n_reps = n_basis: 64-bit */

    /* Per-thread accumulators: only B floats per real/imag. */
    float result_re[MAX_B];
    float result_im[MAX_B];

    /* Diagonal contribution applied to this thread's output row. */
    float diag_t = __ldg(&diag_vals[t]);
    long long v_base_t = (off_t + kp) * B;   /* 64-bit: n_basis*B may exceed 2^31 */
    for (int b = 0; b < B; b++) {
        result_re[b] = diag_t * __ldg(&V_re[v_base_t + b]);
        result_im[b] = diag_t * __ldg(&V_im[v_base_t + b]);
    }

    /* Off-diagonal entries: one per CLT entry whose target is t. */
    int n_e   = __ldg(&entries_per_rep[t]);
    long long e_off = __ldg(&entry_offsets[t]);

    for (int ei = 0; ei < n_e; ei++) {
        long long e   = e_off + ei - entry_base;   /* tile-local index (entry_base=0 resident) */
        int src, g;
        if (srcg != NULL) {                       /* G2: unpack src(25b)|g(7b) */
            unsigned int pk = __ldg(&srcg[e]);
            src = (int)(pk & 0x1FFFFFFu) - 1;
            g   = (int)(pk >> 25) - 1;
        } else {
            src = __ldg(&src_idx[e]) - 1;
            g   = (g_idx8 != NULL) ? ((int)__ldg(&g_idx8[e]) - 1)   /* uint8 g (|G|<=255) */
                                   : ((int)__ldg(&g_idx[e])  - 1);  /* uint16 g */
        }
        /* c_a is either per-entry (s > 1/2) or constant (s = 1/2).
         * The constant branch is taken when c_a_vec is NULL; this
         * trades a runtime branch for ~ 320-640 MB VRAM saved at
         * icosidodecahedron scale. */
        float c_a;
        if (c_idx != NULL)        c_a = c_table[(int)c_idx[e] - 1];         /* indexed (s>1/2) */
        else if (c_a_vec != NULL) c_a = __ldg(&c_a_vec[e]);                 /* per-entry single */
        else                      c_a = c_a_const;                         /* constant (s=1/2) */

        long long off_s = __ldg(&rep_offsets[src]);
        int n_s      = __ldg(&n_per_rep[src]);

        if (t_triv && __ldg(&triv[src]) != 0) {
            /* #1 FAST PATH (trivial source AND trivial target): the reduced
             * d x d block M_e[kp,k] = c_a * Qbar_g[kp,k] is precomputed per g
             * (Qbar already folds in sqrt_t/sqrt_r and Ve' rho Ve), so we skip
             * reading rho + V_s and the m,n contraction entirely. */
            int Q_base = g * d2;
            for (int k = 0; k < n_s; k++) {
                /* Qbar_g[kp, k] in column-major flat = kp + k*d. */
                float Mr = c_a * __ldg(&Qbar_re[Q_base + kp + k * d]);
                float Mi = c_a * __ldg(&Qbar_im[Q_base + kp + k * d]);
                long long v_base_s = (off_s + k) * B;
                for (int b = 0; b < B; b++) {
                    float vr_b = __ldg(&V_re[v_base_s + b]);
                    float vi_b = __ldg(&V_im[v_base_s + b]);
                    result_re[b] += Mr * vr_b - Mi * vi_b;
                    result_im[b] += Mr * vi_b + Mi * vr_b;
                }
            }
        } else {
            int vs_rep = (v_slot != NULL) ? v_slot[src] : src;
            long long V_r_base    = (long long)vs_rep * d2;   /* 64-bit: d^2*n_reps */
            int rho_base = g * d2;                            /* |G|*d^2: small, int OK */
            long long sqrt_r_base = (long long)vs_rep * d;    /* 64-bit: n_basis */

            /* Loop over the source rep's effective basis dimension. */
            for (int k = 0; k < n_s; k++) {
                float sqrt_r = __ldg(&sqrt_eig_all[sqrt_r_base + k]);
                float inv_sqrt_r = 1.0f / sqrt_r;

                /* inner[kp, k] = sum_{m, n} conj(V_t[m, kp]) * rho[m, n] * V_r[n, k]
                 * V_t[:, kp] sits in registers; rho and V_r read via __ldg.
                 * The D threads of this rep all read the same rho / V_r per
                 * iteration, so L1 broadcast keeps cost low. */
                float inner_re = 0.0f, inner_im = 0.0f;
                for (int m = 0; m < d; m++) {
                    float vt_re = vt_re_local[m];
                    float vt_im = vt_im_local[m];
                    for (int n = 0; n < d; n++) {
                        float ro_re = __ldg(&rho_all_re[rho_base + n * d + m]);
                        float ro_im = __ldg(&rho_all_im[rho_base + n * d + m]);
                        float vr_re = __ldg(&V_all_re[V_r_base + k * d + n]);
                        float vr_im = __ldg(&V_all_im[V_r_base + k * d + n]);

                        /* a = conj(V_t[m, kp]) * rho[m, n] */
                        float a_re = vt_re * ro_re + vt_im * ro_im;
                        float a_im = vt_re * ro_im - vt_im * ro_re;

                        /* inner += a * V_r[n, k] */
                        inner_re += a_re * vr_re - a_im * vr_im;
                        inner_im += a_re * vr_im + a_im * vr_re;
                    }
                }

                float scale = c_a * sqrt_t * inv_sqrt_r;
                float Mr = inner_re * scale;
                float Mi = inner_im * scale;

                long long v_base_s = (off_s + k) * B;
                for (int b = 0; b < B; b++) {
                    float vr_b = __ldg(&V_re[v_base_s + b]);
                    float vi_b = __ldg(&V_im[v_base_s + b]);
                    result_re[b] += Mr * vr_b - Mi * vi_b;
                    result_im[b] += Mr * vi_b + Mi * vr_b;
                }
            }
        }
    }

    /* Write this thread's output row. */
    long long v_base_out = (off_t + kp) * B;
    for (int b = 0; b < B; b++) {
        W_re[v_base_out + b] = result_re[b];
        W_im[v_base_out + b] = result_im[b];
    }
}

/* ================================================================
 * SpMV kernel: REAL-arithmetic fork of the OTF kernel above (s_is_real).
 *
 * Used when the irrep data is REAL (realified space-group / I_h irreps):
 * all *_im loads, accumulators and stores are dropped, halving the
 * dominant random V[src] gather traffic and the output writes. The
 * structure (Qbar fast path, c paths, packed/unpacked g, rep_lo/rep_hi/
 * entry_base streaming windows) is identical to the complex kernel.
 *
 * BIT-IDENTITY CONTRACT (test_real_kernel): with V_im = 0 and
 * rho_im = 0 every imaginary term in the complex kernel is an exact
 * IEEE zero (products with 0.0f are +-0.0f; adding/subtracting them
 * never changes a value), so W_re here must equal the complex kernel's
 * W_re EXACTLY. The complex kernel rounds each re-product separately
 * and then adds it to the accumulator with a second rounding: in
 *     inner_re += a_re * vr_re - a_im * vr_im;
 * FMA contraction can only fuse a_re*vr_re into the subtraction (whose
 * other operand is an exact zero -> value = round(a_re*vr_re)); the
 * outer += stays a plain add. A naive `inner_re += a_re * vr_re;` here
 * would instead contract into ONE fused multiply-add (single rounding)
 * and break the gate, so the accumulations below use explicit
 * __fmul_rn / __fadd_rn (never contracted by the compiler). Pure-real
 * expressions shared with the complex kernel (diag product, scale, Mr)
 * keep the identical source shape and compile identically.
 * ================================================================ */
template <typename VT, int DT>   /* VT = float | __half (R1); DT = compile-time d, 0 = runtime */
/* KEIN __launch_bounds__ (Bisektion 2026-07-11): das nachtraeglich eingefuehrte
 * __launch_bounds__(otf_bs_ct(DT)) veraenderte den Codegen AUCH der DT=0-
 * Instanz (352-Kappe, die der historische Kernel nie hatte) und trug einen
 * Teil der +11-14%-Ada-Regression. Der Default-Pfad soll exakt den
 * historischen Codegen liefern; das Templating (FTLM_D_TEMPLATES=1) ist
 * ein Experiment und braucht keine Bounds-Garantie. */
__global__ void
heisenberg_clut_block_spmv_pg_Ih_otf_real(
    VT          * __restrict__ W_re,
    const VT    * __restrict__ V_re,
    const float * __restrict__ diag_vals,
    const long long * __restrict__ rep_offsets,  /* int64: basis offsets reach n_basis > 2^31 */
    const int   * __restrict__ n_per_rep,
    const int   * __restrict__ entries_per_rep,
    const long long * __restrict__ entry_offsets,
    const int   * __restrict__ src_idx,
    const unsigned short * __restrict__ g_idx,
    const unsigned char * __restrict__ g_idx8,  /* G1b: uint8 g (|G|<=255); NULL -> use uint16 g_idx */
    const unsigned int  * __restrict__ srcg,    /* packed src(25b)|g(7b) (G2); NULL -> use src_idx/g_idx */
    const float * __restrict__ c_a_vec,    /* may be NULL when c_a is constant */
    float                       c_a_const, /* used when c_a_vec == NULL */
    const unsigned char * __restrict__ c_idx,   /* uint8 index (s>1/2); NULL otherwise */
    const float * __restrict__ c_table,         /* distinct c values indexed by c_idx */
    const float * __restrict__ V_all_re,
    const float * __restrict__ rho_all_re,
    const float * __restrict__ sqrt_eig_all,
    const unsigned char * __restrict__ triv,    /* per-rep trivial flag (#1); NULL -> no fast path */
    const float * __restrict__ Qbar_re,         /* per-g reduced d x d block (#1) */
    const int * __restrict__ v_slot,            /* per-rep V/sqrt slot (D2 compact-V); NULL -> rep==slot */
    int n_reps, int B, int reps_per_block,
    int rep_lo, int rep_hi, long long entry_base)
{
    const int d  = (DT > 0) ? DT : c_d_irrep;
    const int d2 = d * d;

    int rep_in_block = threadIdx.x / d;
    int kp           = threadIdx.x % d;
    int t            = rep_lo + blockIdx.x * reps_per_block + rep_in_block;

    if (rep_in_block >= reps_per_block) return;
    if (t >= rep_hi) return;

    int n_t = __ldg(&n_per_rep[t]);
    if (kp >= n_t) return;       /* padded thread for this rep */

    long long off_t = __ldg(&rep_offsets[t]);

    bool t_triv = (triv != NULL) && (Qbar_re != NULL) && (__ldg(&triv[t]) != 0);

    /* V_t[:, kp] cached in registers: d floats (real only). With DT > 0 the
     * array is sized EXACTLY d (no MAX_D register waste) and the fill loop
     * unrolls; DT == 0 keeps the historical layout. */
    int vt_rep   = (v_slot != NULL) ? v_slot[t] : t;
    long long V_t_base = (long long)vt_rep * d2;   /* V_all is d^2*n_reps floats: 64-bit */
    float vt_re_local[(DT > 0) ? DT : MAX_D];
    /* Unroll nur bei Kompilezeit-d (Bisektion 2026-07-11: erzwungenes
     * Unrolling der Laufzeit-d-Schleifen kostete +7-8 % auf sm_89). */
    #pragma unroll ((DT > 0) ? DT : 1)
    for (int m = 0; m < d; m++) {
        vt_re_local[m] = __ldg(&V_all_re[V_t_base + kp * d + m]);
    }
    float sqrt_t = __ldg(&sqrt_eig_all[(long long)vt_rep * d + kp]);   /* d*n_reps = n_basis: 64-bit */

    /* Per-thread accumulators: only B floats. */
    float result_re[MAX_B];

    /* Diagonal contribution (same plain-product shape as the complex kernel). */
    float diag_t = __ldg(&diag_vals[t]);
    long long v_base_t = (off_t + kp) * B;   /* 64-bit: n_basis*B may exceed 2^31 */
    for (int b = 0; b < B; b++) {
        result_re[b] = diag_t * ld_vec(&V_re[v_base_t + b]);
    }

    /* Off-diagonal entries: one per CLT entry whose target is t. */
    int n_e   = __ldg(&entries_per_rep[t]);
    long long e_off = __ldg(&entry_offsets[t]);

    for (int ei = 0; ei < n_e; ei++) {
        long long e   = e_off + ei - entry_base;   /* tile-local index (entry_base=0 resident) */
        int src, g;
        if (srcg != NULL) {                       /* G2: unpack src(25b)|g(7b) */
            unsigned int pk = __ldg(&srcg[e]);
            src = (int)(pk & 0x1FFFFFFu) - 1;
            g   = (int)(pk >> 25) - 1;
        } else {
            src = __ldg(&src_idx[e]) - 1;
            g   = (g_idx8 != NULL) ? ((int)__ldg(&g_idx8[e]) - 1)   /* uint8 g (|G|<=255) */
                                   : ((int)__ldg(&g_idx[e])  - 1);  /* uint16 g */
        }
        float c_a;
        if (c_idx != NULL)        c_a = c_table[(int)c_idx[e] - 1];         /* indexed (s>1/2) */
        else if (c_a_vec != NULL) c_a = __ldg(&c_a_vec[e]);                 /* per-entry single */
        else                      c_a = c_a_const;                         /* constant (s=1/2) */

        long long off_s = __ldg(&rep_offsets[src]);
        int n_s      = __ldg(&n_per_rep[src]);

        if (t_triv && __ldg(&triv[src]) != 0) {
            /* #1 FAST PATH: Qbar is purely real for real irreps. */
            int Q_base = g * d2;
            for (int k = 0; k < n_s; k++) {
                float Mr = c_a * __ldg(&Qbar_re[Q_base + kp + k * d]);
                long long v_base_s = (off_s + k) * B;
                for (int b = 0; b < B; b++) {
                    float vr_b = ld_vec(&V_re[v_base_s + b]);
                    result_re[b] = __fadd_rn(result_re[b], __fmul_rn(Mr, vr_b));
                }
            }
        } else {
            int vs_rep = (v_slot != NULL) ? v_slot[src] : src;
            long long V_r_base    = (long long)vs_rep * d2;   /* 64-bit: d^2*n_reps */
            int rho_base = g * d2;                            /* |G|*d^2: small, int OK */
            long long sqrt_r_base = (long long)vs_rep * d;    /* 64-bit: n_basis */

            for (int k = 0; k < n_s; k++) {
                float sqrt_r = __ldg(&sqrt_eig_all[sqrt_r_base + k]);
                float inv_sqrt_r = 1.0f / sqrt_r;

                /* inner[kp, k] = sum_{m, n} V_t[m, kp] * rho[m, n] * V_r[n, k]
                 * (everything real). Explicit round-to-nearest per product +
                 * per accumulate to match the complex kernel's value sequence
                 * (see the bit-identity contract above). */
                float inner_re = 0.0f;
                #pragma unroll ((DT > 0) ? DT : 1)
                for (int m = 0; m < d; m++) {
                    float vt_re = vt_re_local[m];
                    #pragma unroll ((DT > 0) ? DT : 1)
                    for (int n = 0; n < d; n++) {
                        float ro_re = __ldg(&rho_all_re[rho_base + n * d + m]);
                        float vr_re = __ldg(&V_all_re[V_r_base + k * d + n]);

                        float a_re = __fmul_rn(vt_re, ro_re);
                        inner_re = __fadd_rn(inner_re, __fmul_rn(a_re, vr_re));
                    }
                }

                float scale = c_a * sqrt_t * inv_sqrt_r;
                float Mr = inner_re * scale;

                long long v_base_s = (off_s + k) * B;
                for (int b = 0; b < B; b++) {
                    float vr_b = ld_vec(&V_re[v_base_s + b]);
                    result_re[b] = __fadd_rn(result_re[b], __fmul_rn(Mr, vr_b));
                }
            }
        }
    }

    /* Write this thread's output row. */
    long long v_base_out = (off_t + kp) * B;
    for (int b = 0; b < B; b++) {
        st_vec(&W_re[v_base_out + b], result_re[b]);
    }
}

/* ================================================================
 * SpMV kernel (legacy, M-tensor based): W = H_block * V on the
 * (M, Gamma) basis.
 *
 * One thread per output rep t. The thread:
 *   1. Loads its diagonal value diag_t and applies it to V[t, :, :].
 *   2. Iterates the CLT entries for t. Per entry e it loads the
 *      precomputed d x d matrix M^(e) (padded with zeros to d_irrep
 *      x d_irrep) and the n_per_rep(src(e)) input values from V.
 *      The contribution is added into local accumulators
 *      result_re[k'], result_im[k'] for each Krylov-block column.
 *   3. Writes n_per_rep(t) complex output values back to W.
 *
 * Layout:
 *   V[(rep_offsets[t] + k) * B + b] is the (k, b)-th value of rep t.
 *
 * Register usage: per thread we hold 2 * d_irrep * B floats in the
 * result accumulators; with d_irrep = 5 and B = 16 that is 160 floats.
 * Block size 64 keeps occupancy reasonable.
 * ================================================================ */
__global__ void heisenberg_clut_block_spmv_pg_Ih(
    float       * __restrict__ W_re,
    float       * __restrict__ W_im,
    const float * __restrict__ V_re,
    const float * __restrict__ V_im,
    const float * __restrict__ diag_vals,
    const long long * __restrict__ rep_offsets,  /* int64: basis offsets reach n_basis > 2^31 */
    const int   * __restrict__ n_per_rep,
    const int   * __restrict__ entries_per_rep,
    const long long * __restrict__ entry_offsets,
    const int   * __restrict__ src_idx,
    const float * __restrict__ M_re,
    const float * __restrict__ M_im,
    int n_reps, int B)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_reps) return;

    long long off_t = __ldg(&rep_offsets[t]);
    int n_t   = __ldg(&n_per_rep[t]);
    float diag_t = __ldg(&diag_vals[t]);

    int d  = c_d_irrep;
    int d2 = d * d;

    /* Per-output accumulators result[k' * B + b]. This legacy one-thread-per-rep
     * kernel holds d*B accumulators, so it is sized to MAX_D_LEGACY (not MAX_D):
     * it only runs on the small-system precomputed-M path where d <= 8. */
    float result_re[MAX_D_LEGACY * MAX_B];
    float result_im[MAX_D_LEGACY * MAX_B];

    /* Initialise with the diagonal contribution. */
    for (int kp = 0; kp < n_t; kp++) {
        long long v_base = (off_t + kp) * B;
        for (int b = 0; b < B; b++) {
            result_re[kp * B + b] = diag_t * V_re[v_base + b];
            result_im[kp * B + b] = diag_t * V_im[v_base + b];
        }
    }

    /* CLT loop */
    int n_e   = __ldg(&entries_per_rep[t]);
    long long e_off = __ldg(&entry_offsets[t]);

    for (int ei = 0; ei < n_e; ei++) {
        long long e   = e_off + ei;
        int src = __ldg(&src_idx[e]);
        long long off_s = __ldg(&rep_offsets[src]);
        int n_s   = __ldg(&n_per_rep[src]);

        const float *M_re_e = M_re + (size_t)e * d2;
        const float *M_im_e = M_im + (size_t)e * d2;

        /* For each input partner k load x[src, k, :] for all B columns,
         * then propagate into all output partners k' via M_e[k', k].
         *
         * Storage convention: M_re is built in MATLAB as a 3-D padded
         * tensor of size [d_irrep x d_irrep x n_entries] and flattened
         * column-major. MATLAB Me(kp, k) thus lives at the linear
         * position kp + k * d_irrep within entry e, so the C-side
         * access must be M_re_e[kp + k * d] (NOT M_re_e[kp * d + k],
         * which would read the transpose of M_e). */
        for (int k = 0; k < n_s; k++) {
            long long v_base = (off_s + k) * B;
            float vr_b[MAX_B];
            float vi_b[MAX_B];
            for (int b = 0; b < B; b++) {
                vr_b[b] = V_re[v_base + b];
                vi_b[b] = V_im[v_base + b];
            }
            for (int kp = 0; kp < n_t; kp++) {
                float mr = M_re_e[kp + k * d];
                float mi = M_im_e[kp + k * d];
                for (int b = 0; b < B; b++) {
                    result_re[kp * B + b] += mr * vr_b[b] - mi * vi_b[b];
                    result_im[kp * B + b] += mr * vi_b[b] + mi * vr_b[b];
                }
            }
        }
    }

    /* Write results */
    for (int kp = 0; kp < n_t; kp++) {
        long long w_base = (off_t + kp) * B;
        for (int b = 0; b < B; b++) {
            W_re[w_base + b] = result_re[kp * B + b];
            W_im[w_base + b] = result_im[kp * B + b];
        }
    }
}

/* ================================================================
 * Lanczos infrastructure: identical algorithmic content to
 * cuda_lanczos_clut_block_pg_cplx.cu. Operates on vectors of length
 * n_basis (not n_reps) because the Krylov-state lives in the full
 * (M, Gamma) basis.
 * ================================================================ */

/* ---- R2-v1 kernels ----------------------------------------------------
 * Diag-Pass: W = diag .* V elementweise (ein Thread pro (t, kp), exakt die
 * Kopfzeile des Standard-OTF-Kernels). Danach akkumulieren die Run-Launches
 * additiv hinein. */
template <typename VT>
__global__ void otf_real_tiled_diag(
    VT          * __restrict__ W_re,
    const VT    * __restrict__ V_re,
    const float * __restrict__ diag_vals,
    const long long * __restrict__ rep_offsets,
    const int   * __restrict__ n_per_rep,
    int n_reps, int B, int reps_per_block, int d)
{
    int rep_in_block = threadIdx.x / d;
    int kp           = threadIdx.x % d;
    int t            = blockIdx.x * reps_per_block + rep_in_block;
    if (rep_in_block >= reps_per_block) return;
    if (t >= n_reps) return;
    int n_t = __ldg(&n_per_rep[t]);
    if (kp >= n_t) return;
    long long off_t = __ldg(&rep_offsets[t]);
    float diag_t = __ldg(&diag_vals[t]);
    long long base = (off_t + kp) * B;
    for (int b = 0; b < B; b++)
        st_vec(&W_re[base + b], diag_t * ld_vec(&V_re[base + b]));
}

/* Run-Pass: ein Thread pro (Run, kp). Identische Entry-Arithmetik wie der
 * Standard-OTF-Kernel (inkl. Qbar-Fastpath und indexiertem c); die Entry-
 * Arrays sind PERMUTIERT ((src_tile, tgt)-Ordnung), der Entry-Bereich des
 * Runs kommt aus run_ptr. W wird per Read-Modify-Write akkumuliert -- pro
 * Launch (= ein Tile) schreibt hoechstens EIN Run auf jedes (t, kp). */
template <typename VT>
__global__ void otf_real_tiled_runs(
    VT          * __restrict__ W_re,
    const VT    * __restrict__ V_re,
    const long long * __restrict__ rep_offsets,
    const int   * __restrict__ n_per_rep,
    const long long * __restrict__ run_ptr,
    const int   * __restrict__ run_tgt,
    const int   * __restrict__ src_idx,
    const unsigned short * __restrict__ g_idx,
    const unsigned char  * __restrict__ g_idx8,
    const unsigned int   * __restrict__ srcg,
    const float * __restrict__ c_a_vec,
    float                       c_a_const,
    const unsigned char * __restrict__ c_idx,
    const float * __restrict__ c_table,
    const float * __restrict__ V_all_re,
    const float * __restrict__ rho_all_re,
    const float * __restrict__ sqrt_eig_all,
    const unsigned char * __restrict__ triv,
    const float * __restrict__ Qbar_re,
    const int * __restrict__ v_slot,
    int B, int runs_per_block, long long run_lo, long long run_hi, int d)
{
    int run_in_block = threadIdx.x / d;
    int kp           = threadIdx.x % d;
    long long r      = run_lo + (long long)blockIdx.x * runs_per_block + run_in_block;
    if (run_in_block >= runs_per_block) return;
    if (r >= run_hi) return;

    int t   = __ldg(&run_tgt[r]) - 1;
    int n_t = __ldg(&n_per_rep[t]);
    if (kp >= n_t) return;
    int d2 = d * d;

    long long off_t = __ldg(&rep_offsets[t]);
    bool t_triv = (triv != NULL) && (Qbar_re != NULL) && (__ldg(&triv[t]) != 0);

    int vt_rep   = (v_slot != NULL) ? v_slot[t] : t;
    long long V_t_base = (long long)vt_rep * d2;
    float vt_re_local[MAX_D];
    for (int m = 0; m < d; m++)
        vt_re_local[m] = __ldg(&V_all_re[V_t_base + kp * d + m]);
    float sqrt_t = __ldg(&sqrt_eig_all[(long long)vt_rep * d + kp]);

    float result_re[MAX_B];
    for (int b = 0; b < B; b++) result_re[b] = 0.0f;

    long long e_lo = __ldg(&run_ptr[r]);
    long long e_hi = __ldg(&run_ptr[r + 1]);
    for (long long e = e_lo; e < e_hi; e++) {
        int src, g;
        if (srcg != NULL) {
            unsigned int pk = __ldg(&srcg[e]);
            src = (int)(pk & 0x1FFFFFFu) - 1;
            g   = (int)(pk >> 25) - 1;
        } else {
            src = __ldg(&src_idx[e]) - 1;
            g   = (g_idx8 != NULL) ? ((int)__ldg(&g_idx8[e]) - 1)
                                   : ((int)__ldg(&g_idx[e])  - 1);
        }
        float c_a;
        if (c_idx != NULL)        c_a = c_table[(int)c_idx[e] - 1];
        else if (c_a_vec != NULL) c_a = __ldg(&c_a_vec[e]);
        else                      c_a = c_a_const;

        long long off_s = __ldg(&rep_offsets[src]);
        int n_s      = __ldg(&n_per_rep[src]);

        if (t_triv && __ldg(&triv[src]) != 0) {
            int Q_base = g * d2;
            for (int k = 0; k < n_s; k++) {
                float Mr = c_a * __ldg(&Qbar_re[Q_base + kp + k * d]);
                long long v_base_s = (off_s + k) * B;
                for (int b = 0; b < B; b++) {
                    float vr_b = ld_vec(&V_re[v_base_s + b]);
                    result_re[b] = __fadd_rn(result_re[b], __fmul_rn(Mr, vr_b));
                }
            }
        } else {
            int vs_rep = (v_slot != NULL) ? v_slot[src] : src;
            long long V_r_base    = (long long)vs_rep * d2;
            int rho_base = g * d2;
            long long sqrt_r_base = (long long)vs_rep * d;
            for (int k = 0; k < n_s; k++) {
                float sqrt_r = __ldg(&sqrt_eig_all[sqrt_r_base + k]);
                float inv_sqrt_r = 1.0f / sqrt_r;
                float inner_re = 0.0f;
                for (int m = 0; m < d; m++) {
                    float vt_re = vt_re_local[m];
                    for (int n = 0; n < d; n++) {
                        float ro_re = __ldg(&rho_all_re[rho_base + n * d + m]);
                        float vr_re = __ldg(&V_all_re[V_r_base + k * d + n]);
                        float a_re = __fmul_rn(vt_re, ro_re);
                        inner_re = __fadd_rn(inner_re, __fmul_rn(a_re, vr_re));
                    }
                }
                float scale = c_a * sqrt_t * inv_sqrt_r;
                float Mr = inner_re * scale;
                long long v_base_s = (off_s + k) * B;
                for (int b = 0; b < B; b++) {
                    float vr_b = ld_vec(&V_re[v_base_s + b]);
                    result_re[b] = __fadd_rn(result_re[b], __fmul_rn(Mr, vr_b));
                }
            }
        }
    }

    /* Additiv in W (RMW; konfliktfrei innerhalb des Launches). */
    long long v_base_out = (off_t + kp) * B;
    for (int b = 0; b < B; b++) {
        float acc = ld_vec(&W_re[v_base_out + b]);
        st_vec(&W_re[v_base_out + b], __fadd_rn(acc, result_re[b]));
    }
}

/* Segmented-V0 (B-unlock, July 2026): scatter ONE start-vector COLUMN into
 * the interleaved V buffer: dst[(off+i)*B + b] = src[i]. With B == 1, b == 0
 * this writes the same bytes as the historical memcpy ('set_v0' keeps that
 * fast path); B > 1 lets the driver draw V0 per COLUMN in chunks, so
 * MATLAB's 2^31-elements-per-gpuArray cap no longer clamps B on the real
 * path. VT dispatch covers FTLM_FP16. */
template <typename VT>
__global__ void scatter_v0_col(VT *dst, const float *src,
                               long long n, int B, int b, long long off,
                               float presc)   /* 2^-k for raw fp16 draws, else 1 */
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    st_vec(&dst[(off + i) * (long long)B + b], __fmul_rn(src[i], presc));
}

template <typename VT>   /* VT = float | __half (R1: dst is the stored vector) */
__global__ void transpose_col2interleaved(
    VT          * __restrict__ dst,
    const float * __restrict__ src,
    long long dim, int B,                        /* 64-bit: dim = n_basis > 2^31 */
    float presc)                                 /* 2^-k for raw fp16 draws, else 1 */
{
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= dim) return;
    for (int b = 0; b < B; b++)
        st_vec(&dst[t * B + b], __fmul_rn(src[t + b * dim], presc));
}

__global__ void fused_dot_partial_cplx(
    float       * __restrict__ partial,
    const float * __restrict__ V_re,
    const float * __restrict__ V_im,
    const float * __restrict__ W_re,
    const float * __restrict__ W_im,
    long long dim, int B)
{
    __shared__ float sdata[FUSED_BS];

    int tid = threadIdx.x;
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;

    for (int b = 0; b < B; b++) {
        float sum = 0.0f;
        if (t < dim) {
            long long idx = t * B + b;
            sum = V_re[idx] * W_re[idx] + V_im[idx] * W_im[idx];
        }
        sdata[tid] = sum;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        if (tid == 0)
            partial[blockIdx.x * B + b] = sdata[0];
    }
}

__global__ void reduce_partial(
    float       * __restrict__ result,
    const float * __restrict__ partial,
    int n_blocks, int B)
{
    /* Accumulate in DOUBLE: for the largest blocks the fold spans millions
     * of nearly equal-sized partials, so the late summands sit at ~1 ULP of
     * a float running sum and a float fold would round systematically.
     * The double left-fold is lossless at every supported block size and
     * stays deterministic (fixed summation order). */
    int b = threadIdx.x;
    if (b >= B) return;
    double sum = 0.0;
    for (int i = 0; i < n_blocks; i++)
        sum += (double)partial[i * B + b];
    result[b] = (float)sum;
}

__global__ void fused_ortho_norm_partial_cplx(
    float       * __restrict__ W_re,
    float       * __restrict__ W_im,
    const float * __restrict__ V_re,
    const float * __restrict__ V_im,
    const float * __restrict__ Vp_re,
    const float * __restrict__ Vp_im,
    const float * __restrict__ alpha,
    const float * __restrict__ beta_prev,
    float       * __restrict__ partial_nrm,
    long long dim, int B, int use_vp)
{
    __shared__ float sdata[FUSED_BS];

    int tid = threadIdx.x;
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;

    for (int b = 0; b < B; b++) {
        float wr = 0.0f, wi = 0.0f;
        if (t < dim) {
            long long idx = t * B + b;
            wr = W_re[idx] - alpha[b] * V_re[idx];
            wi = W_im[idx] - alpha[b] * V_im[idx];
            if (use_vp) {
                wr -= beta_prev[b] * Vp_re[idx];
                wi -= beta_prev[b] * Vp_im[idx];
            }
            W_re[idx] = wr;
            W_im[idx] = wi;
        }
        sdata[tid] = wr * wr + wi * wi;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        if (tid == 0)
            partial_nrm[blockIdx.x * B + b] = sdata[0];
    }
}

__global__ void scale_interleaved_cplx(
    float       * __restrict__ W_re,
    float       * __restrict__ W_im,
    const float * __restrict__ scale,
    long long dim, int B)
{
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= dim) return;
    long long base = t * B;
    for (int b = 0; b < B; b++) {
        W_re[base + b] *= scale[b];
        W_im[base + b] *= scale[b];
    }
}

/* ----------------------------------------------------------------
 * Real twins of the three complex Lanczos helpers (s_is_real path).
 * transpose_col2interleaved / reduce_partial are scalar -> reused.
 * Bit-identity vs the complex helpers at V_im = W_im = 0: the dropped
 * im-terms are exact zeros (V_im*W_im / wi*wi contribute +-0 to the
 * shared-memory sums; the re-expressions are textually identical so
 * they compile to the same arithmetic).
 * ---------------------------------------------------------------- */
template <typename VT>
__global__ void fused_dot_partial_real(
    float       * __restrict__ partial,
    const VT    * __restrict__ V_re,
    const VT    * __restrict__ W_re,
    long long dim, int B)
{
    __shared__ float sdata[FUSED_BS];

    int tid = threadIdx.x;
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;

    for (int b = 0; b < B; b++) {
        float sum = 0.0f;
        if (t < dim) {
            long long idx = t * B + b;
            sum = ld_vec(&V_re[idx]) * ld_vec(&W_re[idx]);
        }
        sdata[tid] = sum;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        if (tid == 0)
            partial[blockIdx.x * B + b] = sdata[0];
    }
}

template <typename VT>
__global__ void fused_ortho_norm_partial_real(
    VT          * __restrict__ W_re,
    const VT    * __restrict__ V_re,
    const VT    * __restrict__ Vp_re,
    const float * __restrict__ alpha,
    const float * __restrict__ beta_prev,
    float       * __restrict__ partial_nrm,
    long long dim, int B, int use_vp)
{
    __shared__ float sdata[FUSED_BS];

    int tid = threadIdx.x;
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;

    for (int b = 0; b < B; b++) {
        float wr = 0.0f;
        if (t < dim) {
            long long idx = t * B + b;
            wr = ld_vec(&W_re[idx]) - alpha[b] * ld_vec(&V_re[idx]);
            if (use_vp) {
                wr -= beta_prev[b] * ld_vec(&Vp_re[idx]);
            }
            /* Quantise BEFORE the norm so beta is computed from exactly the
             * stored vector (identity for float -> bit-gates unaffected). */
            wr = qz_vec(wr, W_re);
            st_vec(&W_re[idx], wr);
        }
        sdata[tid] = wr * wr;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        if (tid == 0)
            partial_nrm[blockIdx.x * B + b] = sdata[0];
    }
}

template <typename VT>
__global__ void scale_interleaved_real(
    VT          * __restrict__ W_re,
    const float * __restrict__ scale,
    long long dim, int B)
{
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= dim) return;
    long long base = t * B;
    for (int b = 0; b < B; b++) {
        st_vec(&W_re[base + b], ld_vec(&W_re[base + b]) * scale[b]);
    }
}

/* ================================================================
 * OTF SpMV launcher: W = H * V on the (M, Gamma) basis.
 *
 *   Resident path (s_stream == false): ONE launch over all reps with the
 *   resident per-entry arrays (rep_lo=0, rep_hi=n_reps, entry_base=0) ->
 *   byte-identical to the pre-streaming code.
 *
 *   Streaming path (s_stream == true): loop over rep-tiles; for each tile copy
 *   its contiguous entry slice (src+g or packed srcg) HOST->DEVICE into the
 *   small tile buffer, then launch the kernel restricted to that tile's reps.
 *   Each rep is fully inside one tile -> W[t] written once, no atomics.
 *   By default pageable synchronous copies, one tile at a time. When s_dbuf
 *   (Lever A: own pinned host table + 2nd tile set + non-blocking streams) is
 *   active, the H2D copy of tile k+1 overlaps the SpMV of tile k.
 * ================================================================ */

/* Async H2D copy of tile k's entry slice into ping-pong buffer `buf` (0/1) on
 * stream `st`. No-op for an empty (diagonal-only) tile. (Lever A helper.)
 * s_pin_base_e: when Lever A pinned only the TAIL (resident-prefix active),
 * the s_h_* buffers hold entries [s_pin_base_e, n) -> shift the indexing.
 * Without a prefix (or without Lever A) the base is 0 (full arrays). */
static void stream_tile_copy(int k, int buf, cudaStream_t st)
{
    long long ebase = s_h_tile_e_start[k] - s_pin_base_e;
    long long cnt   = s_h_tile_e_count[k];
    if (cnt <= 0) return;
    if (s_h_srcg != NULL) {
        cudaMemcpyAsync(buf ? s_d_srcg_tile2 : s_d_srcg_tile, s_h_srcg + ebase,
                        (size_t)cnt * sizeof(unsigned int), cudaMemcpyHostToDevice, st);
    } else {
        cudaMemcpyAsync(buf ? s_d_src_tile2 : s_d_src_tile, s_h_src_idx + ebase,
                        (size_t)cnt * sizeof(int), cudaMemcpyHostToDevice, st);
        if (s_h_g_idx8 != NULL)
            cudaMemcpyAsync(buf ? s_d_g_tile8_2 : s_d_g_tile8, s_h_g_idx8 + ebase,
                            (size_t)cnt * sizeof(unsigned char), cudaMemcpyHostToDevice, st);
        else
            cudaMemcpyAsync(buf ? s_d_g_tile2 : s_d_g_tile, s_h_g_idx + ebase,
                            (size_t)cnt * sizeof(unsigned short), cudaMemcpyHostToDevice, st);
    }
    if (s_h_c_idx != NULL)
        cudaMemcpyAsync(buf ? s_d_c_idx_tile2 : s_d_c_idx_tile, s_h_c_idx + ebase,
                        (size_t)cnt * sizeof(unsigned char), cudaMemcpyHostToDevice, st);
}

/* ---- R3 pipeline helpers (see the static block for the design) ---- */

/* memcpy split across s_r3_slices threads: a single core moves ~8-12 GB/s
 * from page-cache-warm mmap pages; 4 slices reach the DMA-relevant 25+ GB/s.
 * Below 8 MB the thread-spawn overhead dominates -> plain memcpy. */
static void r3_sliced_memcpy(void *dst, const void *src, size_t bytes)
{
    int ns = s_r3_slices;
    if (ns <= 1 || bytes < ((size_t)8 << 20)) { memcpy(dst, src, bytes); return; }
    if (ns > 8) ns = 8;
    size_t chunk = (bytes + (size_t)ns - 1) / (size_t)ns;
    std::thread th[7];
    int nt = 0;
    for (int i = 1; i < ns; i++) {
        size_t off = (size_t)i * chunk;
        if (off >= bytes) break;
        size_t len = (off + chunk > bytes) ? bytes - off : chunk;
        char *d = (char *)dst + off;
        const char *s = (const char *)src + off;
        th[nt++] = std::thread([d, s, len] { memcpy(d, s, len); });
    }
    memcpy(dst, src, chunk > bytes ? bytes : chunk);
    for (int i = 0; i < nt; i++) th[i].join();
}

/* Staging worker: pulls tile k from the (mmap) host arrays into pin[k&1].
 * Runs ahead of the DMA by at most 2 tiles (ring depth), guarded by
 * s_r3_pin_free_upto which the copy stream bumps AFTER the DMA out of the
 * pin buffer has completed (cudaLaunchHostFunc below). */
static void r3_stage_worker(void)
{
    for (int k = s_pref_tiles; k < s_n_tiles; k++) {
        int j   = k - s_pref_tiles;
        int buf = j & 1;
        while (s_r3_pin_free_upto.load(std::memory_order_acquire) < j - 2) {
            /* After a context error the cudaLaunchHostFunc callbacks never
             * fire (unlike cudaStreamAddCallback) -> the main thread raises
             * s_r3_abort instead of us waiting forever. No CUDA/mex API
             * here: just exit; the main thread reports the error. */
            if (s_r3_abort.load(std::memory_order_acquire)) return;
            std::this_thread::yield();
        }
        if (s_r3_abort.load(std::memory_order_acquire)) return;
        long long ebase = s_h_tile_e_start[k] - s_pin_base_e;
        long long cnt   = s_h_tile_e_count[k];
        if (cnt > 0) {
            if (s_h_srcg != NULL) {
                r3_sliced_memcpy(s_r3_pin_srcg[buf], s_h_srcg + ebase,
                                 (size_t)cnt * sizeof(unsigned int));
            } else {
                r3_sliced_memcpy(s_r3_pin_src[buf], s_h_src_idx + ebase,
                                 (size_t)cnt * sizeof(int));
                if (s_h_g_idx8 != NULL)
                    r3_sliced_memcpy(s_r3_pin_g[buf], s_h_g_idx8 + ebase,
                                     (size_t)cnt * sizeof(unsigned char));
                else
                    r3_sliced_memcpy(s_r3_pin_g[buf], s_h_g_idx + ebase,
                                     (size_t)cnt * sizeof(unsigned short));
            }
            if (s_h_c_idx != NULL)
                r3_sliced_memcpy(s_r3_pin_cidx[buf], s_h_c_idx + ebase,
                                 (size_t)cnt * sizeof(unsigned char));
        }
        s_r3_staged_upto.store(j, std::memory_order_release);
    }
}

/* DMA tile k out of pin[buf] into device tile set `buf` on stream `st`.
 * Mirrors stream_tile_copy, but the source is the pinned staging buffer
 * (offset 0) -> true async DMA at PCIe speed. */
static void r3_tile_dma(int k, int buf, cudaStream_t st)
{
    long long cnt = s_h_tile_e_count[k];
    if (cnt <= 0) return;
    if (s_h_srcg != NULL) {
        cudaMemcpyAsync(buf ? s_d_srcg_tile2 : s_d_srcg_tile, s_r3_pin_srcg[buf],
                        (size_t)cnt * sizeof(unsigned int), cudaMemcpyHostToDevice, st);
    } else {
        cudaMemcpyAsync(buf ? s_d_src_tile2 : s_d_src_tile, s_r3_pin_src[buf],
                        (size_t)cnt * sizeof(int), cudaMemcpyHostToDevice, st);
        if (s_h_g_idx8 != NULL)
            cudaMemcpyAsync(buf ? s_d_g_tile8_2 : s_d_g_tile8, s_r3_pin_g[buf],
                            (size_t)cnt * sizeof(unsigned char), cudaMemcpyHostToDevice, st);
        else
            cudaMemcpyAsync(buf ? s_d_g_tile2 : s_d_g_tile, s_r3_pin_g[buf],
                            (size_t)cnt * sizeof(unsigned short), cudaMemcpyHostToDevice, st);
    }
    if (s_h_c_idx != NULL)
        cudaMemcpyAsync(buf ? s_d_c_idx_tile2 : s_d_c_idx_tile, s_r3_pin_cidx[buf],
                        (size_t)cnt * sizeof(unsigned char), cudaMemcpyHostToDevice, st);
}

/* Executes on the copy stream AFTER the tile-j DMA -> pin[j&1] is reusable.
 * Host functions must not call the CUDA API; we only bump an atomic. Issue
 * order on the stream is monotonic in j, so the store is monotonic too. */
static void CUDART_CB r3_mark_pin_free_cb(void *ud)
{
    s_r3_pin_free_upto.store((int)(intptr_t)ud, std::memory_order_release);
}

/* Free the resident-prefix device buffers and zero the prefix counters.
 * Shared by the keep-table self-heal, the FTLM_DEBUG_DROP_PREFIX test hook
 * and the re-grow path (which frees the smaller prefix before uploading the
 * larger one). Clears the soft error a failed partial alloc left behind. */
static void drop_resident_prefix(void)
{
    if (s_d_src_pref)  { cudaFree(s_d_src_pref);  s_d_src_pref  = NULL; }
    if (s_d_g_pref)    { cudaFree(s_d_g_pref);    s_d_g_pref    = NULL; }
    if (s_d_g8_pref)   { cudaFree(s_d_g8_pref);   s_d_g8_pref   = NULL; }
    if (s_d_srcg_pref) { cudaFree(s_d_srcg_pref); s_d_srcg_pref = NULL; }
    if (s_d_cidx_pref) { cudaFree(s_d_cidx_pref); s_d_cidx_pref = NULL; }
    s_pref_tiles = 0; s_pref_rep_hi = 0; s_pref_e_count = 0;
    cudaGetLastError();
}

/* ---- Resident-prefix sizing + upload (fresh init AND keep-table re-grow) ----
 * Keep the leading whole tiles permanently on the GPU and stream only the
 * tail per SpMV. pref_budget (driver prhs[30]): < 0 auto (size to free VRAM
 * minus the V0 RNG transients still to come in block_lanczos, minus
 * `extra_reserve` for a second ping-pong tile set that will only be
 * allocated AFTER this call on a fresh init, minus a 1 GB margin), = 0 off
 * (full streaming; force_stream tests keep coverage), > 0 explicit byte cap.
 * Whole tiles only: the partition guarantees every rep is wholly inside one
 * tile, so the prefix is the contiguous entry range [0, cum) -> one extra
 * kernel launch with entry_base = 0. Any failure -> prefix 0, full
 * streaming (correctness unaffected). Allowed for mmap sources (bulk read).
 *
 * fresh = false (audit K1b, re-grow under keep-table reuse): a kept prefix
 * may have been dropped by the self-heal (an earlier irrep's Lanczos
 * buffers crowded it out) or sized under tighter VRAM. Re-size against the
 * CURRENT free VRAM and re-upload if MORE tiles fit than are resident now
 * (grow-only; the current prefix bytes count as available since its buffers
 * are freed before the larger upload). Quiet when nothing grows. */
static void setup_resident_prefix(double pref_budget, size_t vec_bytes,
                                  long long tile_cap, long long n_entries_stream,
                                  bool fresh)
{
    size_t bpe = (s_h_srcg != NULL) ? sizeof(unsigned int)
               : sizeof(int) + ((s_h_g_idx8 != NULL) ? sizeof(unsigned char)
                                                     : sizeof(unsigned short));
    if (s_h_c_idx != NULL) bpe += sizeof(unsigned char);
    double budget = pref_budget;
    if (pref_budget < 0) {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        /* Test-Hook-Vertrag (Robustheits-Audit 2026-07-11): ALLE VRAM-
         * adaptiven Entscheidungen respektieren FTLM_FAKE_FREE_VRAM_GB
         * (Kartengroessen-Emulation; gpu_free_bytes.m ist das MATLAB-
         * Gegenstueck). Nur kappen, nie inflationieren. */
        { const char *fk = getenv("FTLM_FAKE_FREE_VRAM_GB");
          if (fk != NULL) {
              double fgb = atof(fk);
              if (fgb > 0.0 && (double)free_b > fgb * 1e9)
                  free_b = (size_t)(fgb * 1e9);
          } }
        double extra_reserve = 0.0;
        if (fresh) {
            /* Second device tile set that Lever A / R3 will allocate right
             * after this call. Under reuse those sets either already exist
             * (already out of free_b) or will not be created. */
            const char *la0 = getenv("FTLM_LEVER_A");
            if (la0 != NULL && la0[0] == '1' && !s_h_is_mmap)
                extra_reserve += (double)tile_cap * (double)bpe;
            const char *r3a = getenv("FTLM_R3");
            if (r3a != NULL && r3a[0] == '1' && s_h_is_mmap)
                extra_reserve += (double)tile_cap * (double)bpe;
        }
        /* V0 RNG transients still to come in block_lanczos (audit K2a):
         * below the 2^31-element gpuArray cap the driver draws V0 one-shot
         * (two n_basis*B fp32 arrays complex, ONE on the real path); beyond
         * it the real path scatters SPLIT-sized chunks via 'set_v0' and only
         * ~2 chunk transients are ever live -- reserving the full fp32 V0
         * there would cost the prefix tens of GB for nothing. */
        double v0_res;
        if ((double)s_n_basis * (double)s_B_batch <= 2147483647.0) {
            v0_res = (s_is_real ? 1.0 : 2.0) * (double)vec_bytes;
        } else {
            double chunk = fmin((double)s_n_basis, 2147483648.0 - 1048576.0)
                         * (double)sizeof(float);
            v0_res = 2.0 * chunk;
        }
        /* Marge: 1 GB absolut ODER 2 % der freien VRAM, je nachdem was
         * groesser ist. Der implizite Vertrag lautet "nach dem Init
         * allokiert bis zum Cleanup niemand mehr als V0 + Marge"; seit die
         * Treiber-B-Wahl gate-genau rechnet, wuerde eine reine 1-GB-Marge
         * auf grossen Karten exakt die Sicherheitsluft konsumieren, die
         * die B-Wahl freigelassen hat (Robustheits-Audit 2026-07-11). */
        double marge = fmax(1.0e9, 0.02 * (double)free_b);
        budget = (double)free_b + (double)s_pref_e_count * (double)bpe
               - v0_res - extra_reserve - marge;
    }
    if (budget > 0 && s_n_tiles > 0 && n_entries_stream > 0) {
        long long cum = 0; int P = 0;
        while (P < s_n_tiles &&
               (double)(cum + s_h_tile_e_count[P]) * (double)bpe <= budget) {
            cum += s_h_tile_e_count[P]; P++;
        }
        int  min_tiles = fresh ? 0 : s_pref_tiles;
        if (P > min_tiles) {
            long long old_tiles = s_pref_tiles;
            if (!fresh) drop_resident_prefix();   /* grow: free smaller prefix first */
            bool ok = true;
            if (cum > 0) {
                if (s_h_srcg != NULL) {
                    if (cudaMalloc(&s_d_srcg_pref, (size_t)cum * sizeof(unsigned int)) != cudaSuccess) ok = false;
                    if (ok && cudaMemcpy(s_d_srcg_pref, s_h_srcg, (size_t)cum * sizeof(unsigned int),
                                         cudaMemcpyHostToDevice) != cudaSuccess) ok = false;
                } else {
                    if (cudaMalloc(&s_d_src_pref, (size_t)cum * sizeof(int)) != cudaSuccess) ok = false;
                    if (ok && cudaMemcpy(s_d_src_pref, s_h_src_idx, (size_t)cum * sizeof(int),
                                         cudaMemcpyHostToDevice) != cudaSuccess) ok = false;
                    if (ok) {
                        if (s_h_g_idx8 != NULL) {
                            if (cudaMalloc(&s_d_g8_pref, (size_t)cum * sizeof(unsigned char)) != cudaSuccess) ok = false;
                            if (ok && cudaMemcpy(s_d_g8_pref, s_h_g_idx8, (size_t)cum * sizeof(unsigned char),
                                                 cudaMemcpyHostToDevice) != cudaSuccess) ok = false;
                        } else {
                            if (cudaMalloc(&s_d_g_pref, (size_t)cum * sizeof(unsigned short)) != cudaSuccess) ok = false;
                            if (ok && cudaMemcpy(s_d_g_pref, s_h_g_idx, (size_t)cum * sizeof(unsigned short),
                                                 cudaMemcpyHostToDevice) != cudaSuccess) ok = false;
                        }
                    }
                }
                /* c_idx rides along BOTH layouts (packed srcg keeps a
                 * separate per-entry c-index, exactly like the tile
                 * buffers). */
                if (ok && s_h_c_idx != NULL) {
                    if (cudaMalloc(&s_d_cidx_pref, (size_t)cum * sizeof(unsigned char)) != cudaSuccess) ok = false;
                    if (ok && cudaMemcpy(s_d_cidx_pref, s_h_c_idx, (size_t)cum * sizeof(unsigned char),
                                         cudaMemcpyHostToDevice) != cudaSuccess) ok = false;
                }
            }
            if (ok) {
                s_pref_tiles   = P;
                s_pref_rep_hi  = s_h_tile_rep_ptr[P];
                s_pref_e_count = cum;
                if (fresh)
                    mexPrintf("[prefix] %d/%d tiles resident on GPU "
                              "(%lld of %lld entries, %.2f GB); streaming the tail.\n",
                              P, s_n_tiles, cum, n_entries_stream,
                              (double)cum * (double)bpe / 1e9);
                else
                    mexPrintf("[prefix] re-grown to %d/%d tiles resident "
                              "(%lld entries, %.2f GB; was %lld tiles) after an "
                              "earlier drop/partial fit.\n",
                              P, s_n_tiles, cum,
                              (double)cum * (double)bpe / 1e9, old_tiles);
            } else {
                drop_resident_prefix();
                mexPrintf("[prefix] device alloc/copy failed -> full streaming.\n");
            }
        } else if (fresh && P == 0) {
            /* Audit K1d: budget positive but below even the first tile. */
            mexPrintf("[prefix] 0/%d tiles resident: budget %.2f GB below the "
                      "first tile (%lld entries, %.2f GB) -> full streaming.\n",
                      s_n_tiles, budget / 1e9, s_h_tile_e_count[0],
                      (double)s_h_tile_e_count[0] * (double)bpe / 1e9);
        }
    } else if (fresh && s_n_tiles > 0 && n_entries_stream > 0) {
        /* Audit K1d: full streaming used to start in complete silence. */
        mexPrintf("[prefix] off (budget %.2f GB%s) -> full streaming.\n",
                  budget / 1e9,
                  (pref_budget == 0) ? ", explicitly disabled" : "");
    }
}

/* Launch the OTF SpMV over the resident PREFIX (reps [0, s_pref_rep_hi),
 * entries [0, s_pref_e_count) already on the device) on stream `st`. The
 * prefix buffers start at entry 0, so entry_base = 0 -- identical indexing to
 * the resident path. No-op when no prefix is held. */
/* d-dispatch shared by the three real launchers: the production dimensions
 * d = 1..6 get compile-time instantiations (unrolled loops, exact register
 * arrays, per-d __launch_bounds__); every other d runs the historical
 * runtime kernel (DT = 0). Same arithmetic in the same order either way.
 *
 * DEFAULT AUS (Bisektion 2026-07-11): auf sm_89 (RTX 4000 Ada) kostet die
 * templated Variante +11-14 % GPU-Phase (icosido FP32 334 -> 382 s; der
 * per Bisektion isoliert: __launch_bounds__ traegt die volle Regression), auf sm_100 (B200)
 * war sie neutral (+-5 %, Messtag 2026-07-10) -- ein Umbau, der nirgends
 * gewinnt. Env FTLM_D_TEMPLATES=1 reaktiviert sie fuer Experimente; beide
 * Pfade sind bit-identisch (Klasse A, test_real_kernel gated beide). */
static int d_templates_on(void)
{
    /* Pro Aufruf gelesen (kein Cache): Tests koennen den Schalter in einer
     * Session togglen, um BEIDE Pfade zu gaten; getenv-Kosten sind gegen
     * die ms-Kernel vernachlaessigbar. */
    const char *e = getenv("FTLM_D_TEMPLATES");
    return (e != NULL && e[0] == '1') ? 1 : 0;
}
#define OTF_REAL_D_SWITCH(IMPL_CALL)                                   \
    if (!d_templates_on()) { IMPL_CALL(0); }                           \
    else switch (s_d_irrep) {                                          \
        case 1: { IMPL_CALL(1); break; }                               \
        case 2: { IMPL_CALL(2); break; }                               \
        case 3: { IMPL_CALL(3); break; }                               \
        case 4: { IMPL_CALL(4); break; }                               \
        case 5: { IMPL_CALL(5); break; }                               \
        case 6: { IMPL_CALL(6); break; }                               \
        default: { IMPL_CALL(0); break; }                              \
    }

template <typename VT, int DT>
static void launch_otf_real_prefix_impl(int blocks, cudaStream_t st,
                                        VT *W_re, const VT *V_re, int B, int otf_rpb)
{
    heisenberg_clut_block_spmv_pg_Ih_otf_real<VT, DT><<<blocks, s_otf_bs, 0, st>>>(
        W_re, V_re,
        s_d_diag_vals, s_d_rep_offsets, s_d_n_per_rep,
        s_d_entries_per_rep, s_d_entry_offsets,
        s_d_src_pref, s_d_g_pref, s_d_g8_pref, s_d_srcg_pref,
        s_d_c_a_vec, s_c_a_const, s_d_cidx_pref, s_d_c_table,
        s_d_V_re_all, s_d_rho_re_all,
        s_d_sqrt_eig_all, s_d_triv, s_d_Qbar_re, s_d_v_slot,
        s_n_reps, B, otf_rpb, 0, s_pref_rep_hi, (long long)0);
}
template <typename VT>   /* R1 dispatch: storage type of the Lanczos vectors */
static void launch_otf_real_prefix_T(int blocks, cudaStream_t st,
                                     VT *W_re, const VT *V_re, int B, int otf_rpb)
{
    #define PREF_CALL(DT_) launch_otf_real_prefix_impl<VT, DT_>(blocks, st, W_re, V_re, B, otf_rpb)
    OTF_REAL_D_SWITCH(PREF_CALL)
    #undef PREF_CALL
}
static void launch_prefix_kernel(float *W_re, float *W_im,
                                 float *V_re, float *V_im, int B, int otf_rpb,
                                 cudaStream_t st)
{
    if (s_pref_tiles <= 0 || s_pref_rep_hi <= 0) return;
    int blocks = (s_pref_rep_hi + otf_rpb - 1) / otf_rpb;
    if (s_is_real) {
        if (s_fp16)
            launch_otf_real_prefix_T<__half>(blocks, st, (__half *)W_re,
                                             (const __half *)V_re, B, otf_rpb);
        else
            launch_otf_real_prefix_T<float>(blocks, st, W_re, V_re, B, otf_rpb);
        return;
    }
    heisenberg_clut_block_spmv_pg_Ih_otf<<<blocks, s_otf_bs, 0, st>>>(
        W_re, W_im, V_re, V_im,
        s_d_diag_vals, s_d_rep_offsets, s_d_n_per_rep,
        s_d_entries_per_rep, s_d_entry_offsets,
        s_d_src_pref, s_d_g_pref, s_d_g8_pref, s_d_srcg_pref,
        s_d_c_a_vec, s_c_a_const, s_d_cidx_pref, s_d_c_table,
        s_d_V_re_all, s_d_V_im_all, s_d_rho_re_all, s_d_rho_im_all,
        s_d_sqrt_eig_all, s_d_triv, s_d_Qbar_re, s_d_Qbar_im, s_d_v_slot,
        s_n_reps, B, otf_rpb, 0, s_pref_rep_hi, (long long)0);
}

/* Launch the OTF SpMV for tile k reading ping-pong buffer `buf` on stream `st`.
 * Mirrors the synchronous streaming launch but with a per-buf tile + stream.
 * (Lever A helper.) */
template <typename VT, int DT>
static void launch_otf_real_tile_impl(int k, int buf, int blocks, cudaStream_t st,
                                      VT *W_re, const VT *V_re, int B, int otf_rpb,
                                      int rep_lo, int rep_hi, long long ebase)
{
    heisenberg_clut_block_spmv_pg_Ih_otf_real<VT, DT><<<blocks, s_otf_bs, 0, st>>>(
        W_re, V_re,
        s_d_diag_vals, s_d_rep_offsets, s_d_n_per_rep,
        s_d_entries_per_rep, s_d_entry_offsets,
        buf ? s_d_src_tile2  : s_d_src_tile,
        buf ? s_d_g_tile2    : s_d_g_tile,
        buf ? s_d_g_tile8_2  : s_d_g_tile8,
        buf ? s_d_srcg_tile2 : s_d_srcg_tile,
        s_d_c_a_vec, s_c_a_const,
        buf ? s_d_c_idx_tile2 : s_d_c_idx_tile, s_d_c_table,
        s_d_V_re_all, s_d_rho_re_all,
        s_d_sqrt_eig_all, s_d_triv, s_d_Qbar_re, s_d_v_slot,
        s_n_reps, B, otf_rpb, rep_lo, rep_hi, ebase);
}
template <typename VT>   /* R1 dispatch */
static void launch_otf_real_tile_T(int k, int buf, int blocks, cudaStream_t st,
                                   VT *W_re, const VT *V_re, int B, int otf_rpb,
                                   int rep_lo, int rep_hi, long long ebase)
{
    #define TILE_CALL(DT_) launch_otf_real_tile_impl<VT, DT_>(k, buf, blocks, st, \
        W_re, V_re, B, otf_rpb, rep_lo, rep_hi, ebase)
    OTF_REAL_D_SWITCH(TILE_CALL)
    #undef TILE_CALL
}

template <typename VT, int DT>
static void launch_otf_real_res_impl(int blocks, VT *W_re, const VT *V_re,
                                     int B, int otf_rpb)
{
    heisenberg_clut_block_spmv_pg_Ih_otf_real<VT, DT><<<blocks, s_otf_bs>>>(
        W_re, V_re,
        s_d_diag_vals, s_d_rep_offsets, s_d_n_per_rep,
        s_d_entries_per_rep, s_d_entry_offsets, s_d_src_idx,
        s_d_g_idx, s_d_g_idx8, s_d_srcg, s_d_c_a_vec, s_c_a_const, s_d_c_idx, s_d_c_table,
        s_d_V_re_all, s_d_rho_re_all,
        s_d_sqrt_eig_all, s_d_triv, s_d_Qbar_re, s_d_v_slot,
        s_n_reps, B, otf_rpb, 0, s_n_reps, (long long)0);
}
template <typename VT>   /* resident real launch (R1 + d-dispatch) */
static void launch_otf_real_res_T(int blocks, VT *W_re, const VT *V_re,
                                  int B, int otf_rpb)
{
    #define RES_CALL(DT_) launch_otf_real_res_impl<VT, DT_>(blocks, W_re, V_re, B, otf_rpb)
    OTF_REAL_D_SWITCH(RES_CALL)
    #undef RES_CALL
}
static void launch_tile_kernel(int k, int buf, float *W_re, float *W_im,
                               float *V_re, float *V_im, int B, int otf_rpb,
                               cudaStream_t st)
{
    int rep_lo = s_h_tile_rep_ptr[k];
    int rep_hi = s_h_tile_rep_ptr[k + 1];
    long long ebase = s_h_tile_e_start[k];
    int nrep   = rep_hi - rep_lo;
    int blocks = (nrep + otf_rpb - 1) / otf_rpb;
    if (blocks < 1) return;
    if (s_is_real) {
        if (s_fp16)
            launch_otf_real_tile_T<__half>(k, buf, blocks, st, (__half *)W_re,
                                           (const __half *)V_re, B, otf_rpb,
                                           rep_lo, rep_hi, ebase);
        else
            launch_otf_real_tile_T<float>(k, buf, blocks, st, W_re, V_re,
                                          B, otf_rpb, rep_lo, rep_hi, ebase);
        return;
    }
    heisenberg_clut_block_spmv_pg_Ih_otf<<<blocks, s_otf_bs, 0, st>>>(
        W_re, W_im, V_re, V_im,
        s_d_diag_vals, s_d_rep_offsets, s_d_n_per_rep,
        s_d_entries_per_rep, s_d_entry_offsets,
        buf ? s_d_src_tile2  : s_d_src_tile,
        buf ? s_d_g_tile2    : s_d_g_tile,
        buf ? s_d_g_tile8_2  : s_d_g_tile8,
        buf ? s_d_srcg_tile2 : s_d_srcg_tile,
        s_d_c_a_vec, s_c_a_const,
        buf ? s_d_c_idx_tile2 : s_d_c_idx_tile, s_d_c_table,
        s_d_V_re_all, s_d_V_im_all, s_d_rho_re_all, s_d_rho_im_all,
        s_d_sqrt_eig_all, s_d_triv, s_d_Qbar_re, s_d_Qbar_im, s_d_v_slot,
        s_n_reps, B, otf_rpb, rep_lo, rep_hi, ebase);
}

static void launch_otf_spmv(float *W_re, float *W_im,
                            float *V_re, float *V_im,
                            int B, int otf_reps_per_block)
{
    /* R2-v1: getilter SpMV (resident real). Diag-Pass initialisiert W, dann
     * ein Run-Launch pro src-Tile (sequenziell, default stream) -- die
     * V[src]-Gathers eines Launches bleiben im L2-Fenster des Tiles. */
    if (s_tiled && s_is_real && !s_stream) {
        int d   = s_d_irrep;
        int rpb = s_otf_bs / d;
        int blocks_d = (s_n_reps + rpb - 1) / rpb;
        if (s_fp16)
            otf_real_tiled_diag<<<blocks_d, s_otf_bs>>>((__half *)W_re,
                (const __half *)V_re, s_d_diag_vals, s_d_rep_offsets,
                s_d_n_per_rep, s_n_reps, B, rpb, d);
        else
            otf_real_tiled_diag<<<blocks_d, s_otf_bs>>>(W_re, V_re,
                s_d_diag_vals, s_d_rep_offsets, s_d_n_per_rep,
                s_n_reps, B, rpb, d);
        for (int kk = 0; kk < s_n_tiles_r2; kk++) {
            long long r_lo = s_h_tile_runp[kk], r_hi = s_h_tile_runp[kk + 1];
            if (r_hi <= r_lo) continue;
            int blocks_r = (int)((r_hi - r_lo + rpb - 1) / rpb);
            if (s_fp16)
                otf_real_tiled_runs<<<blocks_r, s_otf_bs>>>((__half *)W_re,
                    (const __half *)V_re, s_d_rep_offsets, s_d_n_per_rep,
                    s_d_run_ptr, s_d_run_tgt, s_d_src_idx, s_d_g_idx,
                    s_d_g_idx8, s_d_srcg, s_d_c_a_vec, s_c_a_const,
                    s_d_c_idx, s_d_c_table, s_d_V_re_all, s_d_rho_re_all,
                    s_d_sqrt_eig_all, s_d_triv, s_d_Qbar_re, s_d_v_slot,
                    B, rpb, r_lo, r_hi, d);
            else
                otf_real_tiled_runs<<<blocks_r, s_otf_bs>>>(W_re, V_re,
                    s_d_rep_offsets, s_d_n_per_rep,
                    s_d_run_ptr, s_d_run_tgt, s_d_src_idx, s_d_g_idx,
                    s_d_g_idx8, s_d_srcg, s_d_c_a_vec, s_c_a_const,
                    s_d_c_idx, s_d_c_table, s_d_V_re_all, s_d_rho_re_all,
                    s_d_sqrt_eig_all, s_d_triv, s_d_Qbar_re, s_d_v_slot,
                    B, rpb, r_lo, r_hi, d);
        }
        return;
    }

    if (!s_stream) {
        int blocks = (s_n_reps + otf_reps_per_block - 1) / otf_reps_per_block;
        if (s_is_real) {
            if (s_fp16)
                launch_otf_real_res_T<__half>(blocks, (__half *)W_re,
                                              (const __half *)V_re, B,
                                              otf_reps_per_block);
            else
                launch_otf_real_res_T<float>(blocks, W_re, V_re, B,
                                             otf_reps_per_block);
            return;
        }
        heisenberg_clut_block_spmv_pg_Ih_otf<<<blocks, s_otf_bs>>>(
            W_re, W_im, V_re, V_im,
            s_d_diag_vals, s_d_rep_offsets, s_d_n_per_rep,
            s_d_entries_per_rep, s_d_entry_offsets, s_d_src_idx,
            s_d_g_idx, s_d_g_idx8, s_d_srcg, s_d_c_a_vec, s_c_a_const, s_d_c_idx, s_d_c_table,
            s_d_V_re_all, s_d_V_im_all, s_d_rho_re_all, s_d_rho_im_all,
            s_d_sqrt_eig_all, s_d_triv, s_d_Qbar_re, s_d_Qbar_im, s_d_v_slot,
            s_n_reps, B, otf_reps_per_block, 0, s_n_reps, (long long)0);
        return;
    }

    /* R3: double-buffered MMAP streaming via pinned ring staging. Tiles,
     * device buffers and launch order are IDENTICAL to the synchronous
     * fallback below -> bit-identical results; only the transport changes
     * (CPU stage -> pinned -> async DMA, overlapped with compute). */
    if (s_r3) {
        /* Drain the previous SpMV's compute + order vs default stream
         * (identical barriers to the Lever-A branch below). */
        cudaStreamSynchronize(s_compute_stream);
        cudaEventRecord(s_stream_done, 0);
        cudaStreamWaitEvent(s_compute_stream, s_stream_done, 0);
        s_r3_staged_upto.store(-1, std::memory_order_relaxed);
        s_r3_pin_free_upto.store(-1, std::memory_order_relaxed);
        s_r3_abort.store(0, std::memory_order_relaxed);
        std::thread stager(r3_stage_worker);
        /* Resident-prefix first: pure compute -> the first tile's CPU staging
         * and DMA overlap with it for free. */
        launch_prefix_kernel(W_re, W_im, V_re, V_im, B, otf_reps_per_block,
                             s_compute_stream);
        /* Stall guard (audit K5a): after a TDR/context error the host-func
         * callbacks stop firing and both spin-waits would hang until the
         * SLURM walltime. Poll the CUDA error state and a wall-clock timeout
         * (default 60 s/tile, env FTLM_R3_TIMEOUT seconds) and fail LOUDLY. */
        /* Timeout-Default vom Tile-Volumen ableiten (Robustheits-Audit
         * 2026-07-11): der Spin-Wait misst auch das mmap-Page-in des
         * Stagers; auf kaltem Netz-Storage sind grosse Tiles legitim
         * langsam. Unterstellt >= 4 MB/s Worst-Case, mindestens 60 s. */
        double r3_tmo = fmax(60.0, (double)s_r3_tile_bytes / 4e6);
        { const char *ts = getenv("FTLM_R3_TIMEOUT");
          if (ts != NULL) { double v = atof(ts); if (v > 0) r3_tmo = v; } }
        for (int k = s_pref_tiles; k < s_n_tiles; k++) {
            int j   = k - s_pref_tiles;
            int buf = j & 1;
            { auto t0 = std::chrono::steady_clock::now(); unsigned spins = 0;
              while (s_r3_staged_upto.load(std::memory_order_acquire) < j) {
                  if (((++spins) & 1023u) == 0u) {
                      cudaError_t pe = cudaPeekAtLastError();
                      bool to = std::chrono::duration<double>(
                                    std::chrono::steady_clock::now() - t0).count() > r3_tmo;
                      if (pe != cudaSuccess || to) {
                          s_r3_abort.store(1, std::memory_order_release);
                          stager.join();
                          mexErrMsgIdAndTxt("clut_block_pg_Ih:r3stall",
                              "R3 pipeline stalled at tail tile %d/%d (%s after %.0f s). "
                              "Rerun without FTLM_R3 or raise FTLM_R3_TIMEOUT.",
                              j, s_n_tiles - s_pref_tiles,
                              (pe != cudaSuccess) ? cudaGetErrorString(pe) : "timeout",
                              r3_tmo);
                      }
                  }
                  std::this_thread::yield();
              } }
            /* WAR on the device tile: tile j-2 (same buf) must be done. */
            if (j >= 2)
                cudaStreamWaitEvent(s_copy_stream, s_buf_free[buf], 0);
            r3_tile_dma(k, buf, s_copy_stream);
            /* pin[buf] reusable once the DMA has drained it. Enqueued even for
             * empty tiles: the staging worker's j-2 guard needs the bump. */
            cudaLaunchHostFunc(s_copy_stream, r3_mark_pin_free_cb,
                               (void *)(intptr_t)j);
            cudaEventRecord(s_tile_ready[buf], s_copy_stream);
            cudaStreamWaitEvent(s_compute_stream, s_tile_ready[buf], 0);
            launch_tile_kernel(k, buf, W_re, W_im, V_re, V_im, B,
                               otf_reps_per_block, s_compute_stream);
            cudaEventRecord(s_buf_free[buf], s_compute_stream);
        }
        stager.join();
        cudaEventRecord(s_stream_done, s_compute_stream);
        cudaStreamWaitEvent(0, s_stream_done, 0);
        return;
    }

    /* Double-buffered overlap (Lever A): copy(tile k+1) || SpMV(tile k). */
    if (s_dbuf) {
        /* Cross-call reuse: the previous SpMV's compute may still be reading the
         * ping-pong buffers -> drain it before reusing them. (Cheap; the caller
         * already syncs past it via its per-iteration alpha/beta D2H.) */
        cudaStreamSynchronize(s_compute_stream);
        /* Pre-barrier: the SpMV reads V (written on the default stream by the
         * previous Lanczos step) and overwrites W -> make the compute stream wait
         * for all prior default-stream work before the first tile kernel. */
        cudaEventRecord(s_stream_done, 0);
        cudaStreamWaitEvent(s_compute_stream, s_stream_done, 0);
        /* Resident-prefix first: pure compute, no copy -> the tail's first tile
         * copy (below) overlaps with it for free. */
        launch_prefix_kernel(W_re, W_im, V_re, V_im, B, otf_reps_per_block,
                             s_compute_stream);
        for (int k = s_pref_tiles; k < s_n_tiles; k++) {
            int j   = k - s_pref_tiles;      /* tail-local index for buf parity */
            int buf = j & 1;
            /* WAR: do not overwrite buf until the kernel that last read it
             * (tail tile j-2) has finished. */
            if (j >= 2)
                cudaStreamWaitEvent(s_copy_stream, s_buf_free[buf], 0);
            stream_tile_copy(k, buf, s_copy_stream);
            cudaEventRecord(s_tile_ready[buf], s_copy_stream);
            /* RAW: the tile kernel waits for its own copy to land. */
            cudaStreamWaitEvent(s_compute_stream, s_tile_ready[buf], 0);
            launch_tile_kernel(k, buf, W_re, W_im, V_re, V_im, B,
                               otf_reps_per_block, s_compute_stream);
            cudaEventRecord(s_buf_free[buf], s_compute_stream);
        }
        /* Post-barrier: the caller's default-stream work waits for the SpMV. */
        cudaEventRecord(s_stream_done, s_compute_stream);
        cudaStreamWaitEvent(0, s_stream_done, 0);
        return;
    }

    /* Streaming: resident prefix (one launch, no copy), then one rep-tile at a
     * time (synchronous fallback). */
    launch_prefix_kernel(W_re, W_im, V_re, V_im, B, otf_reps_per_block, 0);
    for (int k = s_pref_tiles; k < s_n_tiles; k++) {
        long long ebase  = s_h_tile_e_start[k];
        long long cnt    = s_h_tile_e_count[k];
        long long hoff   = ebase - s_pin_base_e;   /* s_h_* hold [s_pin_base_e, n) */
        if (cnt > 0) {
            if (s_h_srcg != NULL) {
                cudaMemcpy(s_d_srcg_tile, s_h_srcg + hoff,
                           (size_t)cnt * sizeof(unsigned int), cudaMemcpyHostToDevice);
            } else {
                cudaMemcpy(s_d_src_tile, s_h_src_idx + hoff,
                           (size_t)cnt * sizeof(int), cudaMemcpyHostToDevice);
                if (s_h_g_idx8 != NULL) {       /* G1b: uint8 g tile (|G|<=255) */
                    cudaMemcpy(s_d_g_tile8, s_h_g_idx8 + hoff,
                               (size_t)cnt * sizeof(unsigned char), cudaMemcpyHostToDevice);
                } else {
                    cudaMemcpy(s_d_g_tile, s_h_g_idx + hoff,
                               (size_t)cnt * sizeof(unsigned short), cudaMemcpyHostToDevice);
                }
            }
            if (s_h_c_idx != NULL)   /* s>=1: stream the c-index tile too */
                cudaMemcpy(s_d_c_idx_tile, s_h_c_idx + hoff,
                           (size_t)cnt * sizeof(unsigned char), cudaMemcpyHostToDevice);
        }
        /* Same launch as before via the shared tile-launch helper (buf 0 =
         * the primary tile buffers, default stream) -- which also dispatches
         * the real-vs-complex kernel fork in ONE place. */
        launch_tile_kernel(k, 0, W_re, W_im, V_re, V_im, B,
                           otf_reps_per_block, 0);
    }
}

/* Borrowed-host-pointer arg for the streaming per-entry arrays. The entry
 * source is normally a typed MATLAB array (int32 src / uint16 g / uint32 srcg)
 * -> mxGetData. For OUT-OF-CORE streaming the caller instead passes a uint64
 * SCALAR holding the raw base+offset address of a memory-mapped file (see
 * mmap_file.cpp): then the kernel cudaMemcpy's tile slices straight from the
 * mapped pages (OS-paged from NVMe), so the entry table never sits resident in
 * host RAM. A normal entry array is never uint64, so the scalar test is an
 * unambiguous discriminator (and back-compatible: typed arrays take mxGetData). */
static const void *host_ptr_arg(const mxArray *a)
{
    if (mxIsUint64(a) && mxGetNumberOfElements(a) == 1)
        return (const void *)(uintptr_t)(*(const uint64_T *)mxGetData(a));
    return (const void *)mxGetData(a);
}

/* ================================================================
 * MEX Gateway
 * ================================================================ */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    mxInitGPU();
    char mode[32];
    mxGetString(prhs[0], mode, sizeof(mode));

    /* ============================================================
     * INIT
     * ============================================================ */
    if (strcmp(mode, "init") == 0)
    {
        /* s_kept_table: a keep-table cleanup preceded us but THIS init is not
         * a same-table streaming init -> the kept state must go now, or the
         * stale s_stream/tile state would poison this mode. */
        if (s_init || s_kept_table) cleanup_all();

        const mxGPUArray *g_diag  = mxGPUCreateFromMxArray(prhs[1]);
        const mxGPUArray *g_ro    = mxGPUCreateFromMxArray(prhs[2]);
        const mxGPUArray *g_npr   = mxGPUCreateFromMxArray(prhs[3]);
        const mxGPUArray *g_epr   = mxGPUCreateFromMxArray(prhs[4]);
        const mxGPUArray *g_eoff  = mxGPUCreateFromMxArray(prhs[5]);
        const mxGPUArray *g_src   = mxGPUCreateFromMxArray(prhs[6]);
        const mxGPUArray *g_Mre   = mxGPUCreateFromMxArray(prhs[7]);
        const mxGPUArray *g_Mim   = mxGPUCreateFromMxArray(prhs[8]);

        s_n_basis  = (long long)mxGetScalar(prhs[9]);
        s_n_reps   = checked_nreps(mxGetScalar(prhs[10]));
        s_n_entries = (long long)mxGetScalar(prhs[11]);
        s_d_irrep  = (int)mxGetScalar(prhs[12]);
        s_B_batch  = (int)mxGetScalar(prhs[13]);

        if (s_d_irrep > MAX_D_LEGACY) {
            mexErrMsgIdAndTxt("clut_block_pg_Ih:d",
                "d_irrep = %d exceeds MAX_D_LEGACY = %d (legacy precomputed-M "
                "kernel; the OTF/skeleton path supports up to MAX_D = %d)!",
                s_d_irrep, MAX_D_LEGACY, MAX_D);
        }
        if (s_B_batch > MAX_B) {
            mexErrMsgIdAndTxt("clut_block_pg_Ih:B",
                "B_batch = %d exceeds MAX_B = %d!", s_B_batch, MAX_B);
        }

        cudaMemcpyToSymbol(c_d_irrep, &s_d_irrep, sizeof(int));
        s_use_otf = false;   /* legacy path: precomputed M tensor */

        /* Allocate persistent CLT-side device memory */
        cudaMalloc(&s_d_diag_vals,       s_n_reps    * sizeof(float));
        cudaMalloc(&s_d_rep_offsets,     s_n_reps    * sizeof(long long));
        cudaMalloc(&s_d_n_per_rep,       s_n_reps    * sizeof(int));
        cudaMalloc(&s_d_entries_per_rep, s_n_reps    * sizeof(int));
        cudaMalloc(&s_d_entry_offsets,   s_n_reps    * sizeof(long long));
        cudaMalloc(&s_d_src_idx,         s_n_entries * sizeof(int));
        size_t M_n = (size_t)s_n_entries * s_d_irrep * s_d_irrep;
        cudaMalloc(&s_d_M_re,            M_n * sizeof(float));
        cudaMalloc(&s_d_M_im,            M_n * sizeof(float));

        cudaMemcpy(s_d_diag_vals, mxGPUGetDataReadOnly(g_diag),
                   s_n_reps * sizeof(float), cudaMemcpyDeviceToDevice);
        check_offsets_int64(g_ro,   "rep_offsets");
        check_offsets_int64(g_eoff, "entry_offsets");
        cudaMemcpy(s_d_rep_offsets, mxGPUGetDataReadOnly(g_ro),
                   s_n_reps * sizeof(long long), cudaMemcpyDeviceToDevice);
        cudaMemcpy(s_d_n_per_rep, mxGPUGetDataReadOnly(g_npr),
                   s_n_reps * sizeof(int),   cudaMemcpyDeviceToDevice);
        cudaMemcpy(s_d_entries_per_rep, mxGPUGetDataReadOnly(g_epr),
                   s_n_reps * sizeof(int),   cudaMemcpyDeviceToDevice);
        cudaMemcpy(s_d_entry_offsets, mxGPUGetDataReadOnly(g_eoff),
                   s_n_reps * sizeof(long long),   cudaMemcpyDeviceToDevice);
        cudaMemcpy(s_d_src_idx, mxGPUGetDataReadOnly(g_src),
                   s_n_entries * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(s_d_M_re, mxGPUGetDataReadOnly(g_Mre),
                   M_n * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(s_d_M_im, mxGPUGetDataReadOnly(g_Mim),
                   M_n * sizeof(float), cudaMemcpyDeviceToDevice);

        /* Lanczos buffers */
        size_t vec_bytes = (size_t)s_n_basis * s_B_batch * sizeof(float);
        cudaMalloc(&s_d_v_re,  vec_bytes);
        cudaMalloc(&s_d_v_im,  vec_bytes);
        cudaMalloc(&s_d_vp_re, vec_bytes);
        cudaMalloc(&s_d_vp_im, vec_bytes);
        cudaMalloc(&s_d_w_re,  vec_bytes);
        cudaMalloc(&s_d_w_im,  vec_bytes);

        s_n_reduce_blocks = (int)((s_n_basis + FUSED_BS - 1) / FUSED_BS);
        cudaMalloc(&s_d_partial,
                   (size_t)s_n_reduce_blocks * s_B_batch * sizeof(float));
        cudaMalloc(&s_d_alpha,     s_B_batch * sizeof(float));
        cudaMalloc(&s_d_beta,      s_B_batch * sizeof(float));
        cudaMalloc(&s_d_beta_prev, s_B_batch * sizeof(float));

        /* Crash guard (mirrors init_skel_ref): a failed cudaMalloc/cudaMemcpy
         * above (VRAM OOM) must error cleanly here, not surface later as a
         * NULL-pointer dereference in the first SpMV. */
        {
            cudaError_t cerr = cudaGetLastError();
            if (cerr != cudaSuccess)
                mexErrMsgIdAndTxt("clut_block_pg_Ih:cudaMalloc",
                    "init/init_ref/init_skel: CUDA allocation/upload failed "
                    "(%s). Too little free VRAM for this sector.",
                    cudaGetErrorString(cerr));
        }

        mxGPUDestroyGPUArray(g_diag); mxGPUDestroyGPUArray(g_ro);
        mxGPUDestroyGPUArray(g_npr);  mxGPUDestroyGPUArray(g_epr);
        mxGPUDestroyGPUArray(g_eoff); mxGPUDestroyGPUArray(g_src);
        mxGPUDestroyGPUArray(g_Mre);  mxGPUDestroyGPUArray(g_Mim);

        s_init = true;
        mexLock();
        mexAtExit(cleanup_all);
    }

    /* ============================================================
     * INIT_REF
     *
     * Same as 'init' but the M_re / M_im input gpuArrays are NOT
     * cudaMalloc'd-and-cudaMemcpy'd. We just store device pointers
     * into them. The MATLAB caller MUST keep the input gpuArrays alive
     * until 'cleanup' is called (otherwise MATLAB's GC will free the
     * memory underneath our pointers).
     *
     * This eliminates the 2x-VRAM peak during init that would
     * otherwise hit ~ 32 GB on the s = 1/2 icosidodecahedron H_g
     * sector and exceed any reasonable consumer GPU.
     * ============================================================ */
    else if (strcmp(mode, "init_ref") == 0)
    {
        /* s_kept_table: a keep-table cleanup preceded us but THIS init is not
         * a same-table streaming init -> the kept state must go now, or the
         * stale s_stream/tile state would poison this mode. */
        if (s_init || s_kept_table) cleanup_all();

        const mxGPUArray *g_diag  = mxGPUCreateFromMxArray(prhs[1]);
        const mxGPUArray *g_ro    = mxGPUCreateFromMxArray(prhs[2]);
        const mxGPUArray *g_npr   = mxGPUCreateFromMxArray(prhs[3]);
        const mxGPUArray *g_epr   = mxGPUCreateFromMxArray(prhs[4]);
        const mxGPUArray *g_eoff  = mxGPUCreateFromMxArray(prhs[5]);
        const mxGPUArray *g_src   = mxGPUCreateFromMxArray(prhs[6]);
        const mxGPUArray *g_Mre   = mxGPUCreateFromMxArray(prhs[7]);
        const mxGPUArray *g_Mim   = mxGPUCreateFromMxArray(prhs[8]);

        s_n_basis  = (long long)mxGetScalar(prhs[9]);
        s_n_reps   = checked_nreps(mxGetScalar(prhs[10]));
        s_n_entries = (long long)mxGetScalar(prhs[11]);
        s_d_irrep  = (int)mxGetScalar(prhs[12]);
        s_B_batch  = (int)mxGetScalar(prhs[13]);

        if (s_d_irrep > MAX_D_LEGACY) {
            mexErrMsgIdAndTxt("clut_block_pg_Ih:d",
                "d_irrep = %d exceeds MAX_D_LEGACY = %d (legacy precomputed-M "
                "kernel; the OTF/skeleton path supports up to MAX_D = %d)!",
                s_d_irrep, MAX_D_LEGACY, MAX_D);
        }
        if (s_B_batch > MAX_B) {
            mexErrMsgIdAndTxt("clut_block_pg_Ih:B",
                "B_batch = %d exceeds MAX_B = %d!", s_B_batch, MAX_B);
        }

        cudaMemcpyToSymbol(c_d_irrep, &s_d_irrep, sizeof(int));
        s_use_otf = false;          /* M-precomputed path */

        /* Allocate persistent CLT-side device memory for the small
         * index/coeff arrays only. M_re / M_im are borrowed. */
        cudaMalloc(&s_d_diag_vals,       s_n_reps    * sizeof(float));
        cudaMalloc(&s_d_rep_offsets,     s_n_reps    * sizeof(long long));
        cudaMalloc(&s_d_n_per_rep,       s_n_reps    * sizeof(int));
        cudaMalloc(&s_d_entries_per_rep, s_n_reps    * sizeof(int));
        cudaMalloc(&s_d_entry_offsets,   s_n_reps    * sizeof(long long));
        cudaMalloc(&s_d_src_idx,         s_n_entries * sizeof(int));

        cudaMemcpy(s_d_diag_vals, mxGPUGetDataReadOnly(g_diag),
                   s_n_reps * sizeof(float), cudaMemcpyDeviceToDevice);
        check_offsets_int64(g_ro,   "rep_offsets");
        check_offsets_int64(g_eoff, "entry_offsets");
        cudaMemcpy(s_d_rep_offsets, mxGPUGetDataReadOnly(g_ro),
                   s_n_reps * sizeof(long long), cudaMemcpyDeviceToDevice);
        cudaMemcpy(s_d_n_per_rep, mxGPUGetDataReadOnly(g_npr),
                   s_n_reps * sizeof(int),   cudaMemcpyDeviceToDevice);
        cudaMemcpy(s_d_entries_per_rep, mxGPUGetDataReadOnly(g_epr),
                   s_n_reps * sizeof(int),   cudaMemcpyDeviceToDevice);
        cudaMemcpy(s_d_entry_offsets, mxGPUGetDataReadOnly(g_eoff),
                   s_n_reps * sizeof(long long),   cudaMemcpyDeviceToDevice);
        cudaMemcpy(s_d_src_idx, mxGPUGetDataReadOnly(g_src),
                   s_n_entries * sizeof(int), cudaMemcpyDeviceToDevice);

        /* M_re / M_im: just store pointers. NO cudaMalloc, NO copy. */
        s_d_M_re   = (float *)mxGPUGetDataReadOnly(g_Mre);
        s_d_M_im   = (float *)mxGPUGetDataReadOnly(g_Mim);
        s_M_is_ref = true;

        /* Lanczos buffers (same as 'init') */
        size_t vec_bytes = (size_t)s_n_basis * s_B_batch * sizeof(float);
        cudaMalloc(&s_d_v_re,  vec_bytes);
        cudaMalloc(&s_d_v_im,  vec_bytes);
        cudaMalloc(&s_d_vp_re, vec_bytes);
        cudaMalloc(&s_d_vp_im, vec_bytes);
        cudaMalloc(&s_d_w_re,  vec_bytes);
        cudaMalloc(&s_d_w_im,  vec_bytes);

        s_n_reduce_blocks = (int)((s_n_basis + FUSED_BS - 1) / FUSED_BS);
        cudaMalloc(&s_d_partial,
                   (size_t)s_n_reduce_blocks * s_B_batch * sizeof(float));
        cudaMalloc(&s_d_alpha,     s_B_batch * sizeof(float));
        cudaMalloc(&s_d_beta,      s_B_batch * sizeof(float));
        cudaMalloc(&s_d_beta_prev, s_B_batch * sizeof(float));

        /* Crash guard (mirrors init_skel_ref): a failed cudaMalloc/cudaMemcpy
         * above (VRAM OOM) must error cleanly here, not surface later as a
         * NULL-pointer dereference in the first SpMV. */
        {
            cudaError_t cerr = cudaGetLastError();
            if (cerr != cudaSuccess)
                mexErrMsgIdAndTxt("clut_block_pg_Ih:cudaMalloc",
                    "init/init_ref/init_skel: CUDA allocation/upload failed "
                    "(%s). Too little free VRAM for this sector.",
                    cudaGetErrorString(cerr));
        }

        /* Release our mxGPUArray references. The underlying gpuArray
         * device memory stays alive because MATLAB still holds the
         * original handles in the caller's clt struct. */
        mxGPUDestroyGPUArray(g_diag); mxGPUDestroyGPUArray(g_ro);
        mxGPUDestroyGPUArray(g_npr);  mxGPUDestroyGPUArray(g_epr);
        mxGPUDestroyGPUArray(g_eoff); mxGPUDestroyGPUArray(g_src);
        mxGPUDestroyGPUArray(g_Mre);  mxGPUDestroyGPUArray(g_Mim);

        s_init = true;
        mexLock();
        mexAtExit(cleanup_all);
    }

    /* ============================================================
     * INIT_SKEL
     *
     * cuda_lanczos_clut_block_pg_Ih('init_skel',
     *     diag_vals_gpu, rep_offsets_gpu, n_per_rep_gpu,
     *     entries_per_rep_gpu, entry_offsets_gpu,
     *     src_idx_gpu, g_idx_gpu, c_a_gpu,
     *     V_re_gpu, V_im_gpu,
     *     rho_re_gpu, rho_im_gpu,
     *     sqrt_eig_gpu,
     *     n_basis, n_reps, n_entries, d_irrep, B_batch,
     *     c_a_const)
     *
     * STUFE 1a (May 2026): tgt_idx no longer in the signature -- the
     * per-(rep, k') OTF kernel derives the target rep from the thread
     * index, so the array was dead weight (320-640 MB).
     * STUFE 1b: if c_a_gpu has length 0 (empty), c_a is taken as the
     * scalar c_a_const (saves another 320-640 MB at s=1/2 scale).
     * ============================================================ */
    else if (strcmp(mode, "init_skel") == 0)
    {
        /* s_kept_table: a keep-table cleanup preceded us but THIS init is not
         * a same-table streaming init -> the kept state must go now, or the
         * stale s_stream/tile state would poison this mode. */
        if (s_init || s_kept_table) cleanup_all();

        /* New signature (Stufe 1a/1b, May 2026): no tgt_idx, c_a may be
         * empty (then c_a_const at prhs[19] is used as a scalar). */
        const mxGPUArray *g_diag  = mxGPUCreateFromMxArray(prhs[1]);
        const mxGPUArray *g_ro    = mxGPUCreateFromMxArray(prhs[2]);
        const mxGPUArray *g_npr   = mxGPUCreateFromMxArray(prhs[3]);
        const mxGPUArray *g_epr   = mxGPUCreateFromMxArray(prhs[4]);
        const mxGPUArray *g_eoff  = mxGPUCreateFromMxArray(prhs[5]);
        const mxGPUArray *g_src   = mxGPUCreateFromMxArray(prhs[6]);
        const mxGPUArray *g_gix   = mxGPUCreateFromMxArray(prhs[7]);
        const mxGPUArray *g_ca    = mxGPUCreateFromMxArray(prhs[8]);
        const mxGPUArray *g_Vre   = mxGPUCreateFromMxArray(prhs[9]);
        const mxGPUArray *g_Vim   = mxGPUCreateFromMxArray(prhs[10]);
        const mxGPUArray *g_rhoRe = mxGPUCreateFromMxArray(prhs[11]);
        const mxGPUArray *g_rhoIm = mxGPUCreateFromMxArray(prhs[12]);
        const mxGPUArray *g_sqrt  = mxGPUCreateFromMxArray(prhs[13]);

        s_n_basis  = (long long)mxGetScalar(prhs[14]);
        s_n_reps   = checked_nreps(mxGetScalar(prhs[15]));
        s_n_entries = (long long)mxGetScalar(prhs[16]);
        s_d_irrep  = (int)mxGetScalar(prhs[17]);
        s_B_batch  = (int)mxGetScalar(prhs[18]);
        s_c_a_const = (float)mxGetScalar(prhs[19]);

        if (s_d_irrep > MAX_D) {
            mexErrMsgIdAndTxt("clut_block_pg_Ih:d",
                "d_irrep = %d exceeds MAX_D = %d!", s_d_irrep, MAX_D);
        }
        if (s_B_batch > MAX_B) {
            mexErrMsgIdAndTxt("clut_block_pg_Ih:B",
                "B_batch = %d exceeds MAX_B = %d!", s_B_batch, MAX_B);
        }

        cudaMemcpyToSymbol(c_d_irrep, &s_d_irrep, sizeof(int));
        s_use_otf = true;
        s_skel_is_ref = false;

        /* Decide whether c_a is per-entry or a single constant.
         * Constant path is selected when the caller passes an empty
         * c_a array (numel == 0); the scalar c_a_const above is then
         * the value used for every entry. At s = 1/2 this is
         * 0.5 * J and saves s_n_entries * 4 bytes of VRAM. */
        size_t n_ca_in = mxGPUGetNumberOfElements(g_ca);
        bool   use_c_a_const = (n_ca_in == 0);

        /* Persistent CLT-skeleton device buffers (NO M tensor, NO tgt_idx) */
        cudaMalloc(&s_d_diag_vals,       s_n_reps    * sizeof(float));
        cudaMalloc(&s_d_rep_offsets,     s_n_reps    * sizeof(long long));
        cudaMalloc(&s_d_n_per_rep,       s_n_reps    * sizeof(int));
        cudaMalloc(&s_d_entries_per_rep, s_n_reps    * sizeof(int));
        cudaMalloc(&s_d_entry_offsets,   s_n_reps    * sizeof(long long));
        cudaMalloc(&s_d_src_idx,         s_n_entries * sizeof(int));

        /* On-the-fly persistent buffers (no tgt_idx) */
        cudaMalloc(&s_d_g_idx,           s_n_entries * sizeof(unsigned short));
        if (!use_c_a_const) {
            cudaMalloc(&s_d_c_a_vec,     s_n_entries * sizeof(float));
        } else {
            s_d_c_a_vec = NULL;
        }
        size_t V_n   = (size_t)s_n_reps * s_d_irrep * s_d_irrep;
        size_t rho_n = mxGPUGetNumberOfElements(g_rhoRe);  /* group.order * d^2 (was hardcoded 120) */
        cudaMalloc(&s_d_V_re_all,        V_n * sizeof(float));
        cudaMalloc(&s_d_V_im_all,        V_n * sizeof(float));
        cudaMalloc(&s_d_rho_re_all,      rho_n * sizeof(float));
        cudaMalloc(&s_d_rho_im_all,      rho_n * sizeof(float));
        cudaMalloc(&s_d_sqrt_eig_all,    (size_t)s_n_reps * s_d_irrep * sizeof(float));

        cudaMemcpy(s_d_diag_vals, mxGPUGetDataReadOnly(g_diag),
                   s_n_reps * sizeof(float), cudaMemcpyDeviceToDevice);
        check_offsets_int64(g_ro,   "rep_offsets");
        check_offsets_int64(g_eoff, "entry_offsets");
        cudaMemcpy(s_d_rep_offsets, mxGPUGetDataReadOnly(g_ro),
                   s_n_reps * sizeof(long long), cudaMemcpyDeviceToDevice);
        cudaMemcpy(s_d_n_per_rep, mxGPUGetDataReadOnly(g_npr),
                   s_n_reps * sizeof(int),   cudaMemcpyDeviceToDevice);
        cudaMemcpy(s_d_entries_per_rep, mxGPUGetDataReadOnly(g_epr),
                   s_n_reps * sizeof(int),   cudaMemcpyDeviceToDevice);
        cudaMemcpy(s_d_entry_offsets, mxGPUGetDataReadOnly(g_eoff),
                   s_n_reps * sizeof(long long),   cudaMemcpyDeviceToDevice);
        cudaMemcpy(s_d_src_idx, mxGPUGetDataReadOnly(g_src),
                   s_n_entries * sizeof(int), cudaMemcpyDeviceToDevice);

        cudaMemcpy(s_d_g_idx, mxGPUGetDataReadOnly(g_gix),
                   s_n_entries * sizeof(unsigned short), cudaMemcpyDeviceToDevice);
        if (!use_c_a_const) {
            cudaMemcpy(s_d_c_a_vec, mxGPUGetDataReadOnly(g_ca),
                       s_n_entries * sizeof(float), cudaMemcpyDeviceToDevice);
        }
        cudaMemcpy(s_d_V_re_all, mxGPUGetDataReadOnly(g_Vre),
                   V_n * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(s_d_V_im_all, mxGPUGetDataReadOnly(g_Vim),
                   V_n * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(s_d_rho_re_all, mxGPUGetDataReadOnly(g_rhoRe),
                   rho_n * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(s_d_rho_im_all, mxGPUGetDataReadOnly(g_rhoIm),
                   rho_n * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(s_d_sqrt_eig_all, mxGPUGetDataReadOnly(g_sqrt),
                   (size_t)s_n_reps * s_d_irrep * sizeof(float),
                   cudaMemcpyDeviceToDevice);

        /* Lanczos buffers */
        size_t vec_bytes = (size_t)s_n_basis * s_B_batch * sizeof(float);
        cudaMalloc(&s_d_v_re,  vec_bytes);
        cudaMalloc(&s_d_v_im,  vec_bytes);
        cudaMalloc(&s_d_vp_re, vec_bytes);
        cudaMalloc(&s_d_vp_im, vec_bytes);
        cudaMalloc(&s_d_w_re,  vec_bytes);
        cudaMalloc(&s_d_w_im,  vec_bytes);

        s_n_reduce_blocks = (int)((s_n_basis + FUSED_BS - 1) / FUSED_BS);
        cudaMalloc(&s_d_partial,
                   (size_t)s_n_reduce_blocks * s_B_batch * sizeof(float));
        cudaMalloc(&s_d_alpha,     s_B_batch * sizeof(float));
        cudaMalloc(&s_d_beta,      s_B_batch * sizeof(float));
        cudaMalloc(&s_d_beta_prev, s_B_batch * sizeof(float));

        /* Crash guard (mirrors init_skel_ref): a failed cudaMalloc/cudaMemcpy
         * above (VRAM OOM) must error cleanly here, not surface later as a
         * NULL-pointer dereference in the first SpMV. */
        {
            cudaError_t cerr = cudaGetLastError();
            if (cerr != cudaSuccess)
                mexErrMsgIdAndTxt("clut_block_pg_Ih:cudaMalloc",
                    "init/init_ref/init_skel: CUDA allocation/upload failed "
                    "(%s). Too little free VRAM for this sector.",
                    cudaGetErrorString(cerr));
        }

        mxGPUDestroyGPUArray(g_diag);  mxGPUDestroyGPUArray(g_ro);
        mxGPUDestroyGPUArray(g_npr);   mxGPUDestroyGPUArray(g_epr);
        mxGPUDestroyGPUArray(g_eoff);  mxGPUDestroyGPUArray(g_src);
        mxGPUDestroyGPUArray(g_gix);   mxGPUDestroyGPUArray(g_ca);
        mxGPUDestroyGPUArray(g_Vre);   mxGPUDestroyGPUArray(g_Vim);
        mxGPUDestroyGPUArray(g_rhoRe); mxGPUDestroyGPUArray(g_rhoIm);
        mxGPUDestroyGPUArray(g_sqrt);

        s_init = true;
        mexLock();
        mexAtExit(cleanup_all);
    }

    /* ============================================================
     * INIT_SKEL_REF (Stufe 2, May 2026)
     *
     * Same signature as 'init_skel' (no tgt_idx, optional empty c_a),
     * but the large skeleton arrays (src_idx, g_idx, c_a_vec, V_re,
     * V_im, rho_re, rho_im, sqrt_eig, diag_vals, rep_offsets,
     * n_per_rep, entries_per_rep, entry_offsets) are NOT cudaMalloc'd
     * + cudaMemcpy'd. We just store device pointers into the input
     * gpuArrays. The MATLAB caller MUST keep all of them alive
     * (e.g. via the `clt` struct) until 'cleanup' is called.
     *
     * Eliminates the ~ 1.4 GB doubled VRAM footprint that init_skel
     * carries during the Lanczos loop (kernel's persistent copy +
     * MATLAB-side gpuArrays).
     *
     * Lanczos / reduction buffers are still cudaMalloc'd here (we
     * own them, and they're sized differently from any MATLAB input).
     * ============================================================ */
    else if (strcmp(mode, "init_skel_ref") == 0)
    {
        /* s_kept_table: a keep-table cleanup preceded us but THIS init is not
         * a same-table streaming init -> the kept state must go now, or the
         * stale s_stream/tile state would poison this mode. */
        if (s_init || s_kept_table) cleanup_all();

        const mxGPUArray *g_diag  = mxGPUCreateFromMxArray(prhs[1]);
        const mxGPUArray *g_ro    = mxGPUCreateFromMxArray(prhs[2]);
        const mxGPUArray *g_npr   = mxGPUCreateFromMxArray(prhs[3]);
        const mxGPUArray *g_epr   = mxGPUCreateFromMxArray(prhs[4]);
        const mxGPUArray *g_eoff  = mxGPUCreateFromMxArray(prhs[5]);
        const mxGPUArray *g_src   = mxGPUCreateFromMxArray(prhs[6]);
        const mxGPUArray *g_gix   = mxGPUCreateFromMxArray(prhs[7]);
        const mxGPUArray *g_ca    = mxGPUCreateFromMxArray(prhs[8]);
        const mxGPUArray *g_Vre   = mxGPUCreateFromMxArray(prhs[9]);
        const mxGPUArray *g_Vim   = mxGPUCreateFromMxArray(prhs[10]);
        const mxGPUArray *g_rhoRe = mxGPUCreateFromMxArray(prhs[11]);
        const mxGPUArray *g_rhoIm = mxGPUCreateFromMxArray(prhs[12]);
        const mxGPUArray *g_sqrt  = mxGPUCreateFromMxArray(prhs[13]);

        s_n_basis  = (long long)mxGetScalar(prhs[14]);
        s_n_reps   = checked_nreps(mxGetScalar(prhs[15]));
        s_n_entries = (long long)mxGetScalar(prhs[16]);
        s_d_irrep  = (int)mxGetScalar(prhs[17]);
        s_B_batch  = (int)mxGetScalar(prhs[18]);
        s_c_a_const = (float)mxGetScalar(prhs[19]);

        if (s_d_irrep > MAX_D) {
            mexErrMsgIdAndTxt("clut_block_pg_Ih:d",
                "d_irrep = %d exceeds MAX_D = %d!", s_d_irrep, MAX_D);
        }
        if (s_B_batch > MAX_B) {
            mexErrMsgIdAndTxt("clut_block_pg_Ih:B",
                "B_batch = %d exceeds MAX_B = %d!", s_B_batch, MAX_B);
        }

        cudaMemcpyToSymbol(c_d_irrep, &s_d_irrep, sizeof(int));
        s_use_otf = true;
        s_skel_is_ref = true;      /* THIS is the magic: ownership stays on MATLAB */

        /* Store pointers ONLY -- no cudaMalloc, no cudaMemcpy. */
        s_d_diag_vals       = (float *)mxGPUGetDataReadOnly(g_diag);
        check_offsets_int64(g_ro,   "rep_offsets");
        check_offsets_int64(g_eoff, "entry_offsets");
        s_d_rep_offsets     = (long long *)mxGPUGetDataReadOnly(g_ro);
        s_d_n_per_rep       = (int   *)mxGPUGetDataReadOnly(g_npr);
        s_d_entries_per_rep = (int   *)mxGPUGetDataReadOnly(g_epr);
        s_d_entry_offsets   = (long long *)mxGPUGetDataReadOnly(g_eoff);
        s_d_src_idx         = (int   *)mxGPUGetDataReadOnly(g_src);
        s_d_g_idx           = (unsigned short *)mxGPUGetDataReadOnly(g_gix);

        size_t n_ca_in = mxGPUGetNumberOfElements(g_ca);
        if (n_ca_in == 0) {
            /* Constant-c_a path (s = 1/2). */
            s_d_c_a_vec = NULL;
        } else {
            s_d_c_a_vec = (float *)mxGPUGetDataReadOnly(g_ca);
        }

        /* Indexed-c path (s > 1/2): c_idx (uint8) + c_table (single) arrive
         * as two trailing args. Empty (or absent for old callers) -> NULL,
         * so the kernel falls back to the per-entry / constant c_a path. */
        if (nrhs >= 22) {
            const mxGPUArray *g_cidx   = mxGPUCreateFromMxArray(prhs[20]);
            const mxGPUArray *g_ctable = mxGPUCreateFromMxArray(prhs[21]);
            if (mxGPUGetNumberOfElements(g_cidx) == 0) {
                s_d_c_idx   = NULL;
                s_d_c_table = NULL;
            } else {
                s_d_c_idx   = (const unsigned char *)mxGPUGetDataReadOnly(g_cidx);
                s_d_c_table = (const float *)mxGPUGetDataReadOnly(g_ctable);
            }
            /* Pointers stay valid (data owned by the persistent clt fields);
             * release the wrappers like the others below. */
            mxGPUDestroyGPUArray(g_cidx);
            mxGPUDestroyGPUArray(g_ctable);
        } else {
            s_d_c_idx   = NULL;
            s_d_c_table = NULL;
        }

        /* Packed src|g (G2): a uint32 trailing arg (prhs[22]). Empty or absent
         * -> NULL, so the kernel uses the separate src_idx / g_idx instead. */
        if (nrhs >= 23) {
            const mxGPUArray *g_srcg = mxGPUCreateFromMxArray(prhs[22]);
            if (mxGPUGetNumberOfElements(g_srcg) == 0) {
                s_d_srcg = NULL;
            } else {
                s_d_srcg = (const unsigned int *)mxGPUGetDataReadOnly(g_srcg);
            }
            mxGPUDestroyGPUArray(g_srcg);
        } else {
            s_d_srcg = NULL;
        }

        /* Trivial-trivial fast-path arrays (#1): triv (uint8 per rep) +
         * Qbar_re/Qbar_im (per-g reduced d x d block). Empty/absent -> NULL,
         * so the kernel uses the full V' rho V contraction everywhere. */
        if (nrhs >= 26) {
            const mxGPUArray *g_triv = mxGPUCreateFromMxArray(prhs[23]);
            const mxGPUArray *g_Qre  = mxGPUCreateFromMxArray(prhs[24]);
            const mxGPUArray *g_Qim  = mxGPUCreateFromMxArray(prhs[25]);
            if (mxGPUGetNumberOfElements(g_triv) == 0 ||
                mxGPUGetNumberOfElements(g_Qre)  == 0) {
                s_d_triv = NULL; s_d_Qbar_re = NULL; s_d_Qbar_im = NULL;
            } else {
                s_d_triv    = (const unsigned char *)mxGPUGetDataReadOnly(g_triv);
                s_d_Qbar_re = (const float *)mxGPUGetDataReadOnly(g_Qre);
                /* Real path: Qbar_im arrives EMPTY (Qbar is purely real for
                 * real irreps) -> NULL; the real SpMV never reads it. */
                s_d_Qbar_im = (mxGPUGetNumberOfElements(g_Qim) > 0)
                            ? (const float *)mxGPUGetDataReadOnly(g_Qim) : NULL;
            }
            mxGPUDestroyGPUArray(g_triv);
            mxGPUDestroyGPUArray(g_Qre);
            mxGPUDestroyGPUArray(g_Qim);
        } else {
            s_d_triv = NULL; s_d_Qbar_re = NULL; s_d_Qbar_im = NULL;
        }

        /* D2 compact-V slot map (prhs[26]): per-rep int32 index into the
         * compacted V/sqrt arrays (trivial reps share one slot). Empty or
         * absent -> NULL -> kernel uses slot == rep (full per-rep V_all). */
        if (nrhs >= 27) {
            const mxGPUArray *g_vslot = mxGPUCreateFromMxArray(prhs[26]);
            if (mxGPUGetNumberOfElements(g_vslot) == 0) {
                s_d_v_slot = NULL;
            } else {
                s_d_v_slot = (const int *)mxGPUGetDataReadOnly(g_vslot);
            }
            mxGPUDestroyGPUArray(g_vslot);
        } else {
            s_d_v_slot = NULL;
        }

        /* Real FP32 path: an EMPTY V_im flags real irrep data (the MATLAB
         * side packs V_im/rho_im/Qbar_im as empties when isreal). All *_im
         * device pointers stay NULL and the Lanczos buffers halve. */
        s_is_real = (mxGPUGetNumberOfElements(g_Vim) == 0);
        if (s_is_real && mxGPUGetNumberOfElements(g_rhoIm) != 0)
            mexErrMsgIdAndTxt("clut_block_pg_Ih:real",
                "init_skel_ref: V_im is empty (real path) but rho_im is not.");

        s_d_V_re_all     = (float *)mxGPUGetDataReadOnly(g_Vre);
        s_d_V_im_all     = s_is_real ? NULL : (float *)mxGPUGetDataReadOnly(g_Vim);
        s_d_rho_re_all   = (float *)mxGPUGetDataReadOnly(g_rhoRe);
        s_d_rho_im_all   = s_is_real ? NULL : (float *)mxGPUGetDataReadOnly(g_rhoIm);
        s_d_sqrt_eig_all = (float *)mxGPUGetDataReadOnly(g_sqrt);

        /* R2-v1: optionale getilte Entry-Struktur -- prhs[27] run_ptr (gpu
         * int64 [n_runs+1]), prhs[28] run_tgt (gpu int32 [n_runs]), prhs[29]
         * tile_run_ptr (HOST int64 [n_tiles+1]). Die Entry-Arrays (prhs 6..)
         * muessen dann in der PERMUTIERTEN (src_tile, tgt)-Ordnung uebergeben
         * werden (BUILD_TILED_ENTRIES). Borrowed wie alle skel-ref-Arrays. */
        s_tiled = false;
        if (nrhs >= 30 && !mxIsEmpty(prhs[29])) {
            /* OWNED device copies: die uebergebenen gpuArrays koennen
             * Temporaries des Aufrufers sein (MATLAB gibt sie nach dem Init
             * frei) -- geborgte Zeiger waeren dangling. n_runs*12 B. */
            const mxGPUArray *g_rp = mxGPUCreateFromMxArray(prhs[27]);
            const mxGPUArray *g_rt = mxGPUCreateFromMxArray(prhs[28]);
            /* Audit K3e: prhs[27..29] carry the STREAM tile partition in
             * init_skel_stream (host int32/int64/int64) -- a mode/argument
             * mix-up would be read as silent garbage. Never fires on valid
             * calls. */
            if (mxGPUGetClassID(g_rp) != mxINT64_CLASS ||
                mxGPUGetClassID(g_rt) != mxINT32_CLASS ||
                mxGetClassID(prhs[29]) != mxINT64_CLASS)
                mexErrMsgIdAndTxt("clut_block_pg_Ih:r2args",
                    "init_skel_ref: prhs[27..29] must be the R2 run structure "
                    "(gpu int64 run_ptr / gpu int32 run_tgt / host int64 "
                    "tile_run_ptr) -- stream-mode tile partition passed to the "
                    "wrong mode?");
            long long n_rp = (long long)mxGPUGetNumberOfElements(g_rp);
            long long n_rt = (long long)mxGPUGetNumberOfElements(g_rt);
            long long *own_rp = NULL; int *own_rt = NULL;
            cudaMalloc(&own_rp, (size_t)n_rp * sizeof(long long));
            cudaMalloc(&own_rt, (size_t)n_rt * sizeof(int));
            cudaMemcpy(own_rp, mxGPUGetDataReadOnly(g_rp),
                       (size_t)n_rp * sizeof(long long), cudaMemcpyDeviceToDevice);
            cudaMemcpy(own_rt, mxGPUGetDataReadOnly(g_rt),
                       (size_t)n_rt * sizeof(int), cudaMemcpyDeviceToDevice);
            s_d_run_ptr = own_rp;
            s_d_run_tgt = own_rt;
            mxGPUDestroyGPUArray(g_rp); mxGPUDestroyGPUArray(g_rt);
            s_n_tiles_r2  = (int)mxGetNumberOfElements(prhs[29]) - 1;
            s_h_tile_runp = (long long *)mxMalloc((size_t)(s_n_tiles_r2 + 1) * sizeof(long long));
            memcpy(s_h_tile_runp, mxGetData(prhs[29]),
                   (size_t)(s_n_tiles_r2 + 1) * sizeof(long long));
            mexMakeMemoryPersistent(s_h_tile_runp);
            s_tiled = true;
            mexPrintf("[R2] tiled SpMV aktiv: %d Tiles.\n", s_n_tiles_r2);
        }

        /* Lanczos / reduction buffers are owned by us (re only when real). */
        s_fp16 = false;
        { const char *fp = getenv("FTLM_FP16");   /* R1: fp16 STORAGE, real only */
          if (s_is_real && fp != NULL && fp[0] == '1') { s_fp16 = true;
              mexPrintf("[FP16] real Lanczos vectors stored as fp16 (fp32 arithmetic).\n"); } }
        init_vt_scale();   /* per-block power-of-two storage grid */
        size_t vec_bytes = (size_t)s_n_basis * s_B_batch * sizeof(float);
        size_t vst_bytes = s_fp16 ? vec_bytes / 2 : vec_bytes;   /* stored size */
        cudaMalloc(&s_d_v_re,  vst_bytes);
        cudaMalloc(&s_d_vp_re, vst_bytes);
        cudaMalloc(&s_d_w_re,  vst_bytes);
        if (!s_is_real) {
            cudaMalloc(&s_d_v_im,  vec_bytes);
            cudaMalloc(&s_d_vp_im, vec_bytes);
            cudaMalloc(&s_d_w_im,  vec_bytes);
        }

        s_n_reduce_blocks = (int)((s_n_basis + FUSED_BS - 1) / FUSED_BS);
        cudaMalloc(&s_d_partial,
                   (size_t)s_n_reduce_blocks * s_B_batch * sizeof(float));
        cudaMalloc(&s_d_alpha,     s_B_batch * sizeof(float));
        cudaMalloc(&s_d_beta,      s_B_batch * sizeof(float));
        cudaMalloc(&s_d_beta_prev, s_B_batch * sizeof(float));

        /* Crash guard: if any Lanczos cudaMalloc above failed (skeleton left
         * too little contiguous free VRAM), error cleanly here instead of
         * dereferencing a NULL device pointer in the first SpMV (which
         * crashes MATLAB). */
        {
            cudaError_t cerr = cudaGetLastError();
            if (cerr != cudaSuccess) {
                mexErrMsgIdAndTxt("clut_block_pg_Ih:cudaMalloc",
                    "init_skel_ref: CUDA allocation of the Lanczos buffers "
                    "failed (%s; ~%.2f GB needed). Too little free VRAM after "
                    "the skeleton -- lower B or shrink the skeleton (c-index).",
                    cudaGetErrorString(cerr),
                    (s_is_real ? 4.0 : 7.0) * (double)vec_bytes / 1e9);
            }
        }

        /* Release our mxGPUArray wrappers; the underlying gpuArray
         * device memory stays alive on the MATLAB side. */
        mxGPUDestroyGPUArray(g_diag);  mxGPUDestroyGPUArray(g_ro);
        mxGPUDestroyGPUArray(g_npr);   mxGPUDestroyGPUArray(g_epr);
        mxGPUDestroyGPUArray(g_eoff);  mxGPUDestroyGPUArray(g_src);
        mxGPUDestroyGPUArray(g_gix);   mxGPUDestroyGPUArray(g_ca);
        mxGPUDestroyGPUArray(g_Vre);   mxGPUDestroyGPUArray(g_Vim);
        mxGPUDestroyGPUArray(g_rhoRe); mxGPUDestroyGPUArray(g_rhoIm);
        mxGPUDestroyGPUArray(g_sqrt);

        s_init = true;
        mexLock();
        mexAtExit(cleanup_all);
    }

    /* ============================================================
     * INIT_SKEL_B2 (B2 entry-tiling, June 2026)
     *
     * Same arg layout as init_skel_ref, BUT the per-ENTRY arrays
     * (src_idx prhs[6], g_idx prhs[7], c_a prhs[8], c_idx prhs[20],
     * srcg prhs[22]) arrive as HOST arrays and are cudaMalloc'd +
     * cudaMemcpy(HostToDevice) here, so n_entries can exceed the 2^31
     * MATLAB-gpuArray element cap (CUDA handles >2^31; entry indexing
     * is 64-bit). All other arrays (per-rep/per-g + int64 entry_offsets)
     * are borrowed gpuArrays as in init_skel_ref. cleanup frees the
     * B2-owned per-entry buffers (s_b2).
     * ============================================================ */
    else if (strcmp(mode, "init_skel_b2") == 0)
    {
        /* s_kept_table: a keep-table cleanup preceded us but THIS init is not
         * a same-table streaming init -> the kept state must go now, or the
         * stale s_stream/tile state would poison this mode. */
        if (s_init || s_kept_table) cleanup_all();

        const mxGPUArray *g_diag  = mxGPUCreateFromMxArray(prhs[1]);
        const mxGPUArray *g_ro    = mxGPUCreateFromMxArray(prhs[2]);
        const mxGPUArray *g_npr   = mxGPUCreateFromMxArray(prhs[3]);
        const mxGPUArray *g_epr   = mxGPUCreateFromMxArray(prhs[4]);
        const mxGPUArray *g_eoff  = mxGPUCreateFromMxArray(prhs[5]);
        const mxGPUArray *g_Vre   = mxGPUCreateFromMxArray(prhs[9]);
        const mxGPUArray *g_Vim   = mxGPUCreateFromMxArray(prhs[10]);
        const mxGPUArray *g_rhoRe = mxGPUCreateFromMxArray(prhs[11]);
        const mxGPUArray *g_rhoIm = mxGPUCreateFromMxArray(prhs[12]);
        const mxGPUArray *g_sqrt  = mxGPUCreateFromMxArray(prhs[13]);

        s_n_basis   = (long long)mxGetScalar(prhs[14]);
        s_n_reps    = checked_nreps(mxGetScalar(prhs[15]));
        s_n_entries = (long long)mxGetScalar(prhs[16]);
        s_d_irrep   = (int)mxGetScalar(prhs[17]);
        s_B_batch   = (int)mxGetScalar(prhs[18]);
        s_c_a_const = (float)mxGetScalar(prhs[19]);
        if (s_d_irrep > MAX_D) mexErrMsgIdAndTxt("clut_block_pg_Ih:d","d=%d>MAX_D=%d",s_d_irrep,MAX_D);
        if (s_B_batch > MAX_B) mexErrMsgIdAndTxt("clut_block_pg_Ih:B","B=%d>MAX_B=%d",s_B_batch,MAX_B);

        cudaMemcpyToSymbol(c_d_irrep, &s_d_irrep, sizeof(int));
        s_use_otf = true; s_skel_is_ref = true; s_b2 = true;

        /* Real FP32 path: empty V_im flags real irrep data (see init_skel_ref). */
        s_is_real = (mxGPUGetNumberOfElements(g_Vim) == 0);
        if (s_is_real && mxGPUGetNumberOfElements(g_rhoIm) != 0)
            mexErrMsgIdAndTxt("clut_block_pg_Ih:real",
                "init_skel_b2: V_im is empty (real path) but rho_im is not.");

        s_d_diag_vals       = (float *)mxGPUGetDataReadOnly(g_diag);
        check_offsets_int64(g_ro,   "rep_offsets");
        check_offsets_int64(g_eoff, "entry_offsets");
        s_d_rep_offsets     = (long long *)mxGPUGetDataReadOnly(g_ro);
        s_d_n_per_rep       = (int   *)mxGPUGetDataReadOnly(g_npr);
        s_d_entries_per_rep = (int   *)mxGPUGetDataReadOnly(g_epr);
        s_d_entry_offsets   = (long long *)mxGPUGetDataReadOnly(g_eoff);
        s_d_V_re_all        = (float *)mxGPUGetDataReadOnly(g_Vre);
        s_d_V_im_all        = s_is_real ? NULL : (float *)mxGPUGetDataReadOnly(g_Vim);
        s_d_rho_re_all      = (float *)mxGPUGetDataReadOnly(g_rhoRe);
        s_d_rho_im_all      = s_is_real ? NULL : (float *)mxGPUGetDataReadOnly(g_rhoIm);
        s_d_sqrt_eig_all    = (float *)mxGPUGetDataReadOnly(g_sqrt);

        /* B2-owned per-entry buffers copied from HOST (64-bit sizes). */
        size_t ns = mxGetNumberOfElements(prhs[6]);
        if (ns > 0) { cudaMalloc(&s_d_src_idx, ns*sizeof(int));
            cudaMemcpy(s_d_src_idx, mxGetData(prhs[6]), ns*sizeof(int), cudaMemcpyHostToDevice); }
        size_t ngp = (nrhs>=23) ? mxGetNumberOfElements(prhs[22]) : 0;
        if (ngp > 0) { unsigned int *t; cudaMalloc(&t, ngp*sizeof(unsigned int));
            cudaMemcpy(t, mxGetData(prhs[22]), ngp*sizeof(unsigned int), cudaMemcpyHostToDevice); s_d_srcg = t; }
        size_t ng = mxGetNumberOfElements(prhs[7]);
        if (ng > 0) {
            if (mxGetClassID(prhs[7]) == mxUINT8_CLASS) {   /* G1b: uint8 g (|G|<=255) */
                cudaMalloc(&s_d_g_idx8, ng*sizeof(unsigned char));
                cudaMemcpy(s_d_g_idx8, mxGetData(prhs[7]), ng*sizeof(unsigned char), cudaMemcpyHostToDevice);
            } else {
                cudaMalloc(&s_d_g_idx, ng*sizeof(unsigned short));
                cudaMemcpy(s_d_g_idx, mxGetData(prhs[7]), ng*sizeof(unsigned short), cudaMemcpyHostToDevice);
            }
        }
        size_t nca = mxGetNumberOfElements(prhs[8]);
        if (nca > 0) { cudaMalloc(&s_d_c_a_vec, nca*sizeof(float));
            cudaMemcpy(s_d_c_a_vec, mxGetData(prhs[8]), nca*sizeof(float), cudaMemcpyHostToDevice); }
        if (nrhs >= 22) {
            size_t nci = mxGetNumberOfElements(prhs[20]);
            if (nci > 0) { unsigned char *t; cudaMalloc(&t, nci*sizeof(unsigned char));
                cudaMemcpy(t, mxGetData(prhs[20]), nci*sizeof(unsigned char), cudaMemcpyHostToDevice); s_d_c_idx = t;
                const mxGPUArray *g_ct = mxGPUCreateFromMxArray(prhs[21]);
                s_d_c_table = (const float *)mxGPUGetDataReadOnly(g_ct); mxGPUDestroyGPUArray(g_ct); }
        }
        if (nrhs >= 26) {
            const mxGPUArray *g_tr=mxGPUCreateFromMxArray(prhs[23]);
            const mxGPUArray *g_qr=mxGPUCreateFromMxArray(prhs[24]);
            const mxGPUArray *g_qi=mxGPUCreateFromMxArray(prhs[25]);
            if (mxGPUGetNumberOfElements(g_tr)>0 && mxGPUGetNumberOfElements(g_qr)>0) {
                s_d_triv=(const unsigned char *)mxGPUGetDataReadOnly(g_tr);
                s_d_Qbar_re=(const float *)mxGPUGetDataReadOnly(g_qr);
                s_d_Qbar_im=(mxGPUGetNumberOfElements(g_qi)>0)   /* empty when real */
                           ? (const float *)mxGPUGetDataReadOnly(g_qi) : NULL;
            }
            mxGPUDestroyGPUArray(g_tr); mxGPUDestroyGPUArray(g_qr); mxGPUDestroyGPUArray(g_qi);
        }
        if (nrhs >= 27) {
            const mxGPUArray *g_vs=mxGPUCreateFromMxArray(prhs[26]);
            if (mxGPUGetNumberOfElements(g_vs)>0) s_d_v_slot=(const int *)mxGPUGetDataReadOnly(g_vs);
            mxGPUDestroyGPUArray(g_vs);
        }
        s_fp16 = false;
        { const char *fp = getenv("FTLM_FP16");   /* R1: fp16 STORAGE, real only */
          if (s_is_real && fp != NULL && fp[0] == '1') { s_fp16 = true;
              mexPrintf("[FP16] real Lanczos vectors stored as fp16 (fp32 arithmetic).\n"); } }
        init_vt_scale();   /* per-block power-of-two storage grid */
        size_t vec_bytes = (size_t)s_n_basis * s_B_batch * sizeof(float);
        size_t vst_bytes = s_fp16 ? vec_bytes / 2 : vec_bytes;   /* stored size */
        cudaMalloc(&s_d_v_re, vst_bytes);
        cudaMalloc(&s_d_vp_re, vst_bytes);
        cudaMalloc(&s_d_w_re, vst_bytes);
        if (!s_is_real) {
            cudaMalloc(&s_d_v_im, vec_bytes);
            cudaMalloc(&s_d_vp_im, vec_bytes);
            cudaMalloc(&s_d_w_im, vec_bytes);
        }
        s_n_reduce_blocks = (int)((s_n_basis + FUSED_BS - 1) / FUSED_BS);
        cudaMalloc(&s_d_partial, (size_t)s_n_reduce_blocks * s_B_batch * sizeof(float));
        cudaMalloc(&s_d_alpha, s_B_batch*sizeof(float));
        cudaMalloc(&s_d_beta, s_B_batch*sizeof(float));
        cudaMalloc(&s_d_beta_prev, s_B_batch*sizeof(float));
        { cudaError_t cerr = cudaGetLastError();
          if (cerr != cudaSuccess) mexErrMsgIdAndTxt("clut_block_pg_Ih:cudaMalloc",
              "init_skel_b2: CUDA allocation failed (%s).", cudaGetErrorString(cerr)); }
        mxGPUDestroyGPUArray(g_diag); mxGPUDestroyGPUArray(g_ro); mxGPUDestroyGPUArray(g_npr);
        mxGPUDestroyGPUArray(g_epr); mxGPUDestroyGPUArray(g_eoff);
        mxGPUDestroyGPUArray(g_Vre); mxGPUDestroyGPUArray(g_Vim);
        mxGPUDestroyGPUArray(g_rhoRe); mxGPUDestroyGPUArray(g_rhoIm); mxGPUDestroyGPUArray(g_sqrt);
        s_init = true; mexLock(); mexAtExit(cleanup_all);
    }

    /* ============================================================
     * INIT_SKEL_STREAM (Streaming-B2, June 2026)
     *
     * Like init_skel_b2 BUT the per-entry src/g (or packed srcg) arrays are
     * NOT uploaded in full; they stay on the borrowed HOST and are streamed in
     * rep-tiles per SpMV (launch_otf_spmv). Only the device tile buffers (sized
     * to the largest tile) + Lanczos buffers are allocated on the GPU; the
     * per-rep tensors are borrowed gpuArrays as in init_skel_ref. Extra trailing
     * args carry the rep-tile partition (host, copied to owned persistent mem):
     *   prhs[27] tile_rep_ptr (int32,  n_tiles+1, 0-based rep boundaries)
     *   prhs[28] tile_e_start (int64,  n_tiles,   entry base offset per tile)
     *   prhs[29] tile_e_count (int64,  n_tiles,   entry count per tile)
     * v1 requires CONSTANT c (s = 1/2): per-entry c_a and indexed c_idx empty.
     * ============================================================ */
    else if (strcmp(mode, "init_skel_stream") == 0)
    {
        if (s_init) cleanup_all();

        /* ---- Keep-table reuse check ----
         * A prior 'cleanup_keep_table' preserved the streaming table state.
         * Reuse it ONLY on an exact fingerprint match: same ORIGINAL host/
         * mmap pointers (compared as values, never dereferenced -- a stale
         * kept pointer is safe to compare), same n_reps, and the SAME full
         * tile partition by content (guards against a recycled heap address
         * from a DIFFERENT M sector). Any mismatch -> full cleanup here,
         * then the fresh init below. MUST run BEFORE any scalar parsing:
         * cleanup_all() resets s_c_a_const (et al.), so a mismatch after
         * parsing would zero the constant spin coefficient -> silent
         * garbage (caught by test_keep_table).  */
        bool reuse = false;
        s_last_init_reused = false;
        if (s_kept_table) {
            unsigned long long fp_src = 0, fp_g = 0, fp_srcg = 0, fp_cidx = 0;
            size_t ngp0 = (nrhs >= 23) ? mxGetNumberOfElements(prhs[22]) : 0;
            if (ngp0 > 0) {
                fp_srcg = (unsigned long long)(uintptr_t)host_ptr_arg(prhs[22]);
            } else {
                fp_src = (unsigned long long)(uintptr_t)host_ptr_arg(prhs[6]);
                fp_g   = (unsigned long long)(uintptr_t)host_ptr_arg(prhs[7]);
            }
            if (nrhs >= 22 && mxGetNumberOfElements(prhs[20]) > 0)
                fp_cidx = (unsigned long long)(uintptr_t)host_ptr_arg(prhs[20]);
            int nt_in = (nrhs >= 30) ? (int)mxGetNumberOfElements(prhs[27]) - 1 : -1;
            reuse = (nt_in == s_n_tiles) && (s_n_tiles > 0)
                && s_h_tile_rep_ptr != NULL
                && s_fp_nreps == checked_nreps(mxGetScalar(prhs[15]))
                && fp_src == s_fp_src && fp_g == s_fp_g
                && fp_srcg == s_fp_srcg && fp_cidx == s_fp_cidx
                && memcmp(s_h_tile_rep_ptr, mxGetData(prhs[27]),
                          (size_t)(s_n_tiles + 1) * sizeof(int)) == 0
                && memcmp(s_h_tile_e_start, mxGetData(prhs[28]),
                          (size_t)s_n_tiles * sizeof(long long)) == 0
                && memcmp(s_h_tile_e_count, mxGetData(prhs[29]),
                          (size_t)s_n_tiles * sizeof(long long)) == 0;
            if (!reuse) cleanup_all();   /* stale kept table */
        }


        const mxGPUArray *g_diag  = mxGPUCreateFromMxArray(prhs[1]);
        const mxGPUArray *g_ro    = mxGPUCreateFromMxArray(prhs[2]);
        const mxGPUArray *g_npr   = mxGPUCreateFromMxArray(prhs[3]);
        const mxGPUArray *g_epr   = mxGPUCreateFromMxArray(prhs[4]);
        const mxGPUArray *g_eoff  = mxGPUCreateFromMxArray(prhs[5]);
        const mxGPUArray *g_Vre   = mxGPUCreateFromMxArray(prhs[9]);
        const mxGPUArray *g_Vim   = mxGPUCreateFromMxArray(prhs[10]);
        const mxGPUArray *g_rhoRe = mxGPUCreateFromMxArray(prhs[11]);
        const mxGPUArray *g_rhoIm = mxGPUCreateFromMxArray(prhs[12]);
        const mxGPUArray *g_sqrt  = mxGPUCreateFromMxArray(prhs[13]);

        s_n_basis   = (long long)mxGetScalar(prhs[14]);
        s_n_reps    = checked_nreps(mxGetScalar(prhs[15]));
        s_n_entries = (long long)mxGetScalar(prhs[16]);
        s_d_irrep   = (int)mxGetScalar(prhs[17]);
        s_B_batch   = (int)mxGetScalar(prhs[18]);
        s_c_a_const = (float)mxGetScalar(prhs[19]);
        if (s_d_irrep > MAX_D) mexErrMsgIdAndTxt("clut_block_pg_Ih:d","d=%d>MAX_D=%d",s_d_irrep,MAX_D);
        if (s_B_batch > MAX_B) mexErrMsgIdAndTxt("clut_block_pg_Ih:B","B=%d>MAX_B=%d",s_B_batch,MAX_B);
        if (nrhs < 30)
            mexErrMsgIdAndTxt("clut_block_pg_Ih:stream",
                "init_skel_stream: needs the rep-tile partition (prhs[27..29]).");

        cudaMemcpyToSymbol(c_d_irrep, &s_d_irrep, sizeof(int));
        s_use_otf = true; s_skel_is_ref = true; s_stream = true;

        /* Real FP32 path: empty V_im flags real irrep data (see init_skel_ref).
         * The streamed entry table is arithmetic-agnostic; only the per-rep
         * tensors and the Lanczos buffers change. */
        s_is_real = (mxGPUGetNumberOfElements(g_Vim) == 0);
        if (s_is_real && mxGPUGetNumberOfElements(g_rhoIm) != 0)
            mexErrMsgIdAndTxt("clut_block_pg_Ih:real",
                "init_skel_stream: V_im is empty (real path) but rho_im is not.");

        s_d_diag_vals       = (float *)mxGPUGetDataReadOnly(g_diag);
        check_offsets_int64(g_ro,   "rep_offsets");
        check_offsets_int64(g_eoff, "entry_offsets");
        s_d_rep_offsets     = (long long *)mxGPUGetDataReadOnly(g_ro);
        s_d_n_per_rep       = (int   *)mxGPUGetDataReadOnly(g_npr);
        s_d_entries_per_rep = (int   *)mxGPUGetDataReadOnly(g_epr);
        s_d_entry_offsets   = (long long *)mxGPUGetDataReadOnly(g_eoff);
        s_d_V_re_all        = (float *)mxGPUGetDataReadOnly(g_Vre);
        s_d_V_im_all        = s_is_real ? NULL : (float *)mxGPUGetDataReadOnly(g_Vim);
        s_d_rho_re_all      = (float *)mxGPUGetDataReadOnly(g_rhoRe);
        s_d_rho_im_all      = s_is_real ? NULL : (float *)mxGPUGetDataReadOnly(g_rhoIm);
        s_d_sqrt_eig_all    = (float *)mxGPUGetDataReadOnly(g_sqrt);

        /* Coefficient c: streaming supports CONSTANT c (s=1/2) OR INDEXED c
         * (s>=1: a borrowed uint8 c_idx streamed per tile + a small resident
         * c_table). Per-entry single c_a is not supported. */
        if (mxGetNumberOfElements(prhs[8]) > 0)
            mexErrMsgIdAndTxt("clut_block_pg_Ih:stream",
                "init_skel_stream: per-entry c_a not supported (use constant or indexed c).");
        s_d_c_a_vec = NULL;  s_d_c_idx = NULL;  s_d_c_table = NULL;
        if (!reuse) s_h_c_idx = NULL;   /* on reuse it may point at the kept pinned copy */
        if (nrhs >= 22 && mxGetNumberOfElements(prhs[20]) > 0) {
            /* s>=1: borrow the host c_idx (uint8, streamed in tiles like src/g)
             * and the resident c_table (small gpuArray). */
            if (!reuse) s_h_c_idx = (const unsigned char *)host_ptr_arg(prhs[20]);
            const mxGPUArray *g_ct = mxGPUCreateFromMxArray(prhs[21]);
            s_d_c_table = (const float *)mxGPUGetDataReadOnly(g_ct);
            mxGPUDestroyGPUArray(g_ct);
        }

        /* Borrowed HOST per-entry arrays: packed srcg XOR unpacked src+g.
         * Skipped on keep-table reuse: the s_h_* pointers were possibly
         * repointed at OUR pinned copies (Lever A) and must stay put. */
        size_t ngp = (nrhs >= 23) ? mxGetNumberOfElements(prhs[22]) : 0;
        if (reuse) { /* table state kept -- pointers already set */ }
        else if (ngp > 0) {
            s_h_srcg    = (const unsigned int  *)host_ptr_arg(prhs[22]);
            s_h_src_idx = NULL; s_h_g_idx = NULL;
        } else {
            s_h_src_idx = (const int           *)host_ptr_arg(prhs[6]);
            if (mxGetClassID(prhs[7]) == mxUINT8_CLASS) {   /* G1b: uint8 g (|G|<=255) */
                s_h_g_idx8 = (const unsigned char  *)host_ptr_arg(prhs[7]);
                s_h_g_idx  = NULL;
            } else {                                        /* uint16 g, or mmap uint64 ptr */
                s_h_g_idx  = (const unsigned short *)host_ptr_arg(prhs[7]);
                s_h_g_idx8 = NULL;
            }
            s_h_srcg    = NULL;
        }

        /* mmap (out-of-core) path: src/g (or srcg) arrive as a uint64 SCALAR raw
         * pointer (file-backed pages) -> Lever A must never copy/pin/double-buffer
         * them (it would defeat the on-disk design + risk faulting huge ranges). */
        if (!reuse)
        s_h_is_mmap = (ngp > 0)
            ? (mxIsUint64(prhs[22]) && mxGetNumberOfElements(prhs[22]) == 1)
            : (mxIsUint64(prhs[6])  && mxGetNumberOfElements(prhs[6])  == 1);
        /* The s>=1 c-index block (prhs[20]) can be a mapped pointer too (the
         * sorted file's third section). With today's MATLAB callers it always
         * co-occurs with mapped src/srcg, but detect it independently so a
         * mapped c_idx alone can never slip past the Lever A exclusion. */
        if (s_h_c_idx != NULL && nrhs >= 22 &&
            mxIsUint64(prhs[20]) && mxGetNumberOfElements(prhs[20]) == 1)
            s_h_is_mmap = true;

        /* Keep-table fingerprint: the ORIGINAL pointers as passed by MATLAB
         * (recorded BEFORE Lever A may repoint s_h_* at pinned copies). */
        if (!reuse) {
            s_fp_src = 0; s_fp_g = 0; s_fp_srcg = 0; s_fp_cidx = 0;
            if (ngp > 0) {
                s_fp_srcg = (unsigned long long)(uintptr_t)host_ptr_arg(prhs[22]);
            } else {
                s_fp_src = (unsigned long long)(uintptr_t)host_ptr_arg(prhs[6]);
                s_fp_g   = (unsigned long long)(uintptr_t)host_ptr_arg(prhs[7]);
            }
            if (nrhs >= 22 && mxGetNumberOfElements(prhs[20]) > 0)
                s_fp_cidx = (unsigned long long)(uintptr_t)host_ptr_arg(prhs[20]);
            s_fp_nreps = s_n_reps;
        }

        /* triv / Qbar (#1) + v_slot (D2): borrowed gpuArrays (resident, small). */
        if (nrhs >= 26) {
            const mxGPUArray *g_tr=mxGPUCreateFromMxArray(prhs[23]);
            const mxGPUArray *g_qr=mxGPUCreateFromMxArray(prhs[24]);
            const mxGPUArray *g_qi=mxGPUCreateFromMxArray(prhs[25]);
            if (mxGPUGetNumberOfElements(g_tr)>0 && mxGPUGetNumberOfElements(g_qr)>0) {
                s_d_triv=(const unsigned char *)mxGPUGetDataReadOnly(g_tr);
                s_d_Qbar_re=(const float *)mxGPUGetDataReadOnly(g_qr);
                s_d_Qbar_im=(mxGPUGetNumberOfElements(g_qi)>0)   /* empty when real */
                           ? (const float *)mxGPUGetDataReadOnly(g_qi) : NULL;
            }
            mxGPUDestroyGPUArray(g_tr); mxGPUDestroyGPUArray(g_qr); mxGPUDestroyGPUArray(g_qi);
        }
        if (nrhs >= 27) {
            const mxGPUArray *g_vs=mxGPUCreateFromMxArray(prhs[26]);
            if (mxGPUGetNumberOfElements(g_vs)>0) s_d_v_slot=(const int *)mxGPUGetDataReadOnly(g_vs);
            mxGPUDestroyGPUArray(g_vs);
        }

        /* Rep-tile partition: copy to OWNED persistent host memory (tiny).
         * On keep-table reuse the kept copy IS this partition (memcmp above). */
        if (!reuse) {
        /* Audit K3e: same slots carry the R2 run structure in init_skel_ref
         * (gpu int64/int32 + host int64) -- guard against a mode mix-up. */
        if (mxGetClassID(prhs[27]) != mxINT32_CLASS ||
            mxGetClassID(prhs[28]) != mxINT64_CLASS ||
            mxGetClassID(prhs[29]) != mxINT64_CLASS)
            mexErrMsgIdAndTxt("clut_block_pg_Ih:streamargs",
                "init_skel_stream: prhs[27..29] must be int32 tile_rep_ptr / "
                "int64 tile_e_start / int64 tile_e_count.");
        s_n_tiles = (int)mxGetNumberOfElements(prhs[27]) - 1;
        s_h_tile_rep_ptr = (int *)mxMalloc((size_t)(s_n_tiles + 1) * sizeof(int));
        s_h_tile_e_start = (long long *)mxMalloc((size_t)s_n_tiles * sizeof(long long));
        s_h_tile_e_count = (long long *)mxMalloc((size_t)s_n_tiles * sizeof(long long));
        memcpy(s_h_tile_rep_ptr, mxGetData(prhs[27]), (size_t)(s_n_tiles + 1) * sizeof(int));
        memcpy(s_h_tile_e_start, mxGetData(prhs[28]), (size_t)s_n_tiles * sizeof(long long));
        memcpy(s_h_tile_e_count, mxGetData(prhs[29]), (size_t)s_n_tiles * sizeof(long long));
        mexMakeMemoryPersistent(s_h_tile_rep_ptr);
        mexMakeMemoryPersistent(s_h_tile_e_start);
        mexMakeMemoryPersistent(s_h_tile_e_count);
        }

        /* Absorb any STALE soft error before our own allocations (audit K1a):
         * the self-heal guard below must only ever see errors raised by the
         * allocations that follow, never a leftover from an earlier call --
         * a stale cudaErrorInvalidValue would spuriously drop a healthy kept
         * prefix. (Do NOT clear again between the tile and Lanczos allocs:
         * a failed tile alloc must stay visible to the guard.) */
        { cudaError_t stale = cudaGetLastError();
          if (stale != cudaSuccess)
              mexPrintf("[init] note: absorbed stale CUDA soft error '%s' "
                        "before device allocations.\n", cudaGetErrorString(stale)); }

        /* Device tile buffers sized to the largest tile. */
        long long tile_cap = 0;
        for (int k = 0; k < s_n_tiles; k++)
            if (s_h_tile_e_count[k] > tile_cap) tile_cap = s_h_tile_e_count[k];
        if (reuse) { /* kept device tile buffers */ }
        else if (s_h_srcg != NULL) {
            cudaMalloc(&s_d_srcg_tile, (size_t)tile_cap * sizeof(unsigned int));
        } else {
            cudaMalloc(&s_d_src_tile, (size_t)tile_cap * sizeof(int));
            if (s_h_g_idx8 != NULL)   /* G1b: uint8 g tile */
                cudaMalloc(&s_d_g_tile8, (size_t)tile_cap * sizeof(unsigned char));
            else
                cudaMalloc(&s_d_g_tile,  (size_t)tile_cap * sizeof(unsigned short));
        }
        if (!reuse && s_h_c_idx != NULL)   /* s>=1: per-tile c-index buffer (1 B/entry) */
            cudaMalloc(&s_d_c_idx_tile, (size_t)tile_cap * sizeof(unsigned char));
        /* Eigener Guard fuer die Tile-Puffer (Robustheits-Audit 2026-07-11):
         * Tile-OOM und Lanczos-OOM getrennt melden -- ein Tile, das nicht
         * einmal alleine passt, ist ein Partitionierungsproblem (kleinere
         * Tiles im Skeleton waehlen), kein B-Problem; und der Self-Heal
         * unten darf nur echte Lanczos-Fehler sehen. */
        { cudaError_t terr = cudaGetLastError();
          if (terr != cudaSuccess)
              mexErrMsgIdAndTxt("clut_block_pg_Ih:tileAlloc",
                  "init_skel_stream: device tile buffer allocation failed "
                  "(%s; largest tile %lld entries = %.2f GB). The tile "
                  "partition exceeds this card -- rebuild the skeleton with "
                  "smaller tiles.", cudaGetErrorString(terr),
                  tile_cap, (double)tile_cap * 7.0 / 1e9); }

        /* Lanczos / reduction buffers (owned; re only when real). */
        s_fp16 = false;
        { const char *fp = getenv("FTLM_FP16");   /* R1: fp16 STORAGE, real only */
          if (s_is_real && fp != NULL && fp[0] == '1') { s_fp16 = true;
              mexPrintf("[FP16] real Lanczos vectors stored as fp16 (fp32 arithmetic).\n"); } }
        init_vt_scale();   /* per-block power-of-two storage grid */
        size_t vec_bytes = (size_t)s_n_basis * s_B_batch * sizeof(float);
        size_t vst_bytes = s_fp16 ? vec_bytes / 2 : vec_bytes;   /* stored size */
        cudaMalloc(&s_d_v_re, vst_bytes);
        cudaMalloc(&s_d_vp_re, vst_bytes);
        cudaMalloc(&s_d_w_re, vst_bytes);
        if (!s_is_real) {
            cudaMalloc(&s_d_v_im, vec_bytes);
            cudaMalloc(&s_d_vp_im, vec_bytes);
            cudaMalloc(&s_d_w_im, vec_bytes);
        }
        s_n_reduce_blocks = (int)((s_n_basis + FUSED_BS - 1) / FUSED_BS);
        cudaMalloc(&s_d_partial, (size_t)s_n_reduce_blocks * s_B_batch * sizeof(float));
        cudaMalloc(&s_d_alpha, s_B_batch*sizeof(float));
        cudaMalloc(&s_d_beta, s_B_batch*sizeof(float));
        cudaMalloc(&s_d_beta_prev, s_B_batch*sizeof(float));
        { cudaError_t cerr = cudaGetLastError();
          /* Self-heal ONLY on a genuine allocation failure (OOM): any other
           * error class must fall through to the hard error below instead of
           * silently costing the resident prefix (audit K1a). */
          if (cerr == cudaErrorMemoryAllocation && reuse && s_pref_e_count > 0 && !s_dbuf) {
              /* Keep-table self-heal: the KEPT resident prefix was sized for an
               * earlier (smaller-d) irrep and now crowds out this irrep's Lanczos
               * buffers. Drop the prefix (-> full tail streaming, still correct,
               * bit-identical) and retry. Excluded under Lever A: its pinned host
               * copy holds only the TAIL entries, so the prefix cannot go. */
              if (s_d_v_re)  cudaFree(s_d_v_re);   if (s_d_v_im)  cudaFree(s_d_v_im);
              if (s_d_vp_re) cudaFree(s_d_vp_re);  if (s_d_vp_im) cudaFree(s_d_vp_im);
              if (s_d_w_re)  cudaFree(s_d_w_re);   if (s_d_w_im)  cudaFree(s_d_w_im);
              if (s_d_partial) cudaFree(s_d_partial);
              if (s_d_alpha) cudaFree(s_d_alpha);  if (s_d_beta) cudaFree(s_d_beta);
              if (s_d_beta_prev) cudaFree(s_d_beta_prev);
              s_d_v_re = NULL; s_d_v_im = NULL; s_d_vp_re = NULL; s_d_vp_im = NULL;
              s_d_w_re = NULL; s_d_w_im = NULL; s_d_partial = NULL;
              s_d_alpha = NULL; s_d_beta = NULL; s_d_beta_prev = NULL;
              drop_resident_prefix();
              mexPrintf("[keep-table] kept prefix dropped to fit this irrep's "
                        "Lanczos buffers -> streaming this irrep; re-grow is "
                        "attempted at the next reuse init.\n");
              cudaMalloc(&s_d_v_re, vst_bytes);
              cudaMalloc(&s_d_vp_re, vst_bytes);
              cudaMalloc(&s_d_w_re, vst_bytes);
              if (!s_is_real) {
                  cudaMalloc(&s_d_v_im, vec_bytes);
                  cudaMalloc(&s_d_vp_im, vec_bytes);
                  cudaMalloc(&s_d_w_im, vec_bytes);
              }
              cudaMalloc(&s_d_partial, (size_t)s_n_reduce_blocks * s_B_batch * sizeof(float));
              cudaMalloc(&s_d_alpha, s_B_batch*sizeof(float));
              cudaMalloc(&s_d_beta, s_B_batch*sizeof(float));
              cudaMalloc(&s_d_beta_prev, s_B_batch*sizeof(float));
              cerr = cudaGetLastError();
          }
          else if (cerr == cudaErrorMemoryAllocation && reuse && s_dbuf) {
              /* Lever-A-Reuse-OOM (Robustheits-Audit 2026-07-11): der Self-
               * Heal ist hier zu Recht ausgeschlossen (die gepinnte Host-
               * Kopie haelt nur den TAIL -- kein Quellmaterial, um den
               * Prefix neu aufzubauen), aber ein nackter Fehler liesse den
               * halb-allokierten Zustand zurueck. Alles fallen lassen und
               * mit klarer Anweisung abbrechen: der naechste Aufruf ist ein
               * frischer Init. */
              cleanup_all();
              mexErrMsgIdAndTxt("clut_block_pg_Ih:leverAReuseOOM",
                  "init_skel_stream: Lanczos buffers do not fit next to the "
                  "kept Lever-A state (pinned tail copy); kernel state was "
                  "fully dropped. Re-run this irrep (fresh init) or disable "
                  "FTLM_LEVER_A for mixed-size irrep sweeps.");
          }
          if (cerr != cudaSuccess) mexErrMsgIdAndTxt("clut_block_pg_Ih:cudaMalloc",
              "init_skel_stream: CUDA allocation failed (%s).", cudaGetErrorString(cerr)); }
        /* The total streamed entry count is the span the tiles cover. Do NOT
         * use s_n_entries: in the streaming call n_entries arrives as a
         * gpuArray scalar (double(sum(gpuArray)) stays on-device), so
         * mxGetScalar(prhs[16]) reads 0 -- harmless for the sync path (it is
         * unused there) but it would mis-size the prefix / pinned buffers
         * here. The tile partition (sum of tile_e_count) is the source of
         * truth, and matches numel of the borrowed src/g arrays we copy from. */
        long long n_entries_stream = 0;
        for (int k = 0; k < s_n_tiles; k++) n_entries_stream += s_h_tile_e_count[k];

        /* ---- Resident-prefix ----
         * Sizing/upload live in setup_resident_prefix (shared with the
         * keep-table re-grow below); budget semantics documented there. */
        if (!reuse) {
            double pref_budget = (nrhs >= 31) ? mxGetScalar(prhs[30]) : -1.0;
            setup_resident_prefix(pref_budget, vec_bytes, tile_cap,
                                  n_entries_stream, true /*fresh*/);
        }

        /* ---- Keep-table prefix re-grow (audit K1b) ----
         * A kept prefix may have been dropped by the self-heal above (an
         * earlier irrep's larger Lanczos buffers crowded it out) or sized
         * under tighter VRAM. Without this, every later irrep of the kept
         * table would stream the FULL table from host per SpMV (the prod_m0
         * prefix-0 finding). Grow-only against CURRENT free VRAM; excluded
         * under Lever A (its pinned host copy holds only the tail -> no
         * source data for the leading tiles). FTLM_DEBUG_DROP_PREFIX=1 is
         * the test hook: drop the kept prefix NOW and skip the re-grow for
         * this one init (simulates the pre-fix behaviour deterministically). */
        if (reuse && !s_dbuf) {
            const char *dbg = getenv("FTLM_DEBUG_DROP_PREFIX");
            if (dbg != NULL && dbg[0] == '1') {
                if (s_pref_e_count > 0) {
                    drop_resident_prefix();
                    mexPrintf("[keep-table] TEST hook FTLM_DEBUG_DROP_PREFIX: "
                              "kept prefix dropped; re-grow skipped for this init.\n");
                }
            } else if (s_pref_tiles < s_n_tiles) {
                double pref_budget = (nrhs >= 31) ? mxGetScalar(prhs[30]) : -1.0;
                if (pref_budget != 0)
                    setup_resident_prefix(pref_budget, vec_bytes, tile_cap,
                                          n_entries_stream, false /*re-grow*/);
            }
        }

        /* ---- Lever A: own-pinned + double-buffered streaming (best-effort) ----
         * Copy the borrowed host entry table ONCE into OUR OWN cudaHostAlloc'd
         * page-locked buffers, repoint s_h_* at them, allocate a 2nd ping-pong
         * device tile set + non-blocking streams/events, then enable the overlap
         * pipeline (launch_otf_spmv). Gated by FTLM_LEVER_A=1 (default OFF). Any
         * failure rolls back cleanly to the synchronous single-buffer path. We
         * never pin BORROWED MATLAB memory (the prior heap-corruption crash) --
         * we own these buffers start to finish. The mmap path is excluded.
         * With a resident prefix only the TAIL entries [s_pref_e_count, n) are
         * pinned/copied (s_pin_base_e shifts the host indexing) -- the pinned
         * host cost shrinks by exactly the prefix size. */
        if (!reuse) {
        s_dbuf = false;
        {
            const char *la = getenv("FTLM_LEVER_A");
            bool la_on = (la != NULL && la[0] == '1');
            long long n_tail = n_entries_stream - s_pref_e_count;
            if (la_on && (s_n_tiles - s_pref_tiles) > 1 && !s_h_is_mmap && n_tail > 0) {
                bool ok = true;
                void *pin_src = NULL, *pin_g = NULL, *pin_srcg = NULL, *pin_cidx = NULL;
                /* 1) OWN pinned host copies of the borrowed TAIL entries. */
                if (s_h_srcg != NULL) {
                    if (cudaHostAlloc(&pin_srcg, (size_t)n_tail * sizeof(unsigned int),
                                      cudaHostAllocDefault) != cudaSuccess) ok = false;
                } else {
                    if (cudaHostAlloc(&pin_src, (size_t)n_tail * sizeof(int),
                                      cudaHostAllocDefault) != cudaSuccess) ok = false;
                    if (ok) {
                        size_t gsz = (s_h_g_idx8 != NULL) ? sizeof(unsigned char)
                                                          : sizeof(unsigned short);
                        if (cudaHostAlloc(&pin_g, (size_t)n_tail * gsz,
                                          cudaHostAllocDefault) != cudaSuccess) ok = false;
                    }
                }
                if (ok && s_h_c_idx != NULL) {
                    if (cudaHostAlloc(&pin_cidx, (size_t)n_tail * sizeof(unsigned char),
                                      cudaHostAllocDefault) != cudaSuccess) ok = false;
                }
                /* 2) 2nd ping-pong device tile set (sized to the largest tile). */
                if (ok) {
                    if (s_h_srcg != NULL) {
                        if (cudaMalloc(&s_d_srcg_tile2, (size_t)tile_cap * sizeof(unsigned int)) != cudaSuccess) ok = false;
                    } else {
                        if (cudaMalloc(&s_d_src_tile2, (size_t)tile_cap * sizeof(int)) != cudaSuccess) ok = false;
                        if (ok) {
                            if (s_h_g_idx8 != NULL) { if (cudaMalloc(&s_d_g_tile8_2, (size_t)tile_cap * sizeof(unsigned char))  != cudaSuccess) ok = false; }
                            else                    { if (cudaMalloc(&s_d_g_tile2,   (size_t)tile_cap * sizeof(unsigned short)) != cudaSuccess) ok = false; }
                        }
                    }
                    if (ok && s_h_c_idx != NULL)
                        if (cudaMalloc(&s_d_c_idx_tile2, (size_t)tile_cap * sizeof(unsigned char)) != cudaSuccess) ok = false;
                }
                if (ok) {
                    /* 3) Fill the owned buffers ONCE with the TAIL entries
                     *    [s_pref_e_count, n) and repoint the streaming host
                     *    pointers at them (the borrowed arrays are not read after
                     *    this; the caller keeps clt alive but we no longer touch
                     *    it). s_pin_base_e shifts all subsequent tile indexing. */
                    long long pe = s_pref_e_count;
                    if (s_h_srcg != NULL) {
                        memcpy(pin_srcg, s_h_srcg + pe, (size_t)n_tail * sizeof(unsigned int));
                        s_h_srcg = (const unsigned int *)pin_srcg;  s_own_pin_srcg = pin_srcg;
                    } else {
                        memcpy(pin_src, s_h_src_idx + pe, (size_t)n_tail * sizeof(int));
                        s_h_src_idx = (const int *)pin_src;  s_own_pin_src = pin_src;
                        if (s_h_g_idx8 != NULL) {
                            memcpy(pin_g, s_h_g_idx8 + pe, (size_t)n_tail * sizeof(unsigned char));
                            s_h_g_idx8 = (const unsigned char *)pin_g;
                        } else {
                            memcpy(pin_g, s_h_g_idx + pe, (size_t)n_tail * sizeof(unsigned short));
                            s_h_g_idx = (const unsigned short *)pin_g;
                        }
                        s_own_pin_g = pin_g;
                    }
                    if (s_h_c_idx != NULL) {
                        memcpy(pin_cidx, s_h_c_idx + pe, (size_t)n_tail * sizeof(unsigned char));
                        s_h_c_idx = (const unsigned char *)pin_cidx;  s_own_pin_cidx = pin_cidx;
                    }
                    s_pin_base_e = pe;
                    /* 4) Non-blocking streams + events (so they don't implicitly
                     *    serialise with the default stream; ordering via events). */
                    cudaStreamCreateWithFlags(&s_copy_stream, cudaStreamNonBlocking);
                    cudaStreamCreateWithFlags(&s_compute_stream, cudaStreamNonBlocking);
                    cudaEventCreateWithFlags(&s_tile_ready[0], cudaEventDisableTiming);
                    cudaEventCreateWithFlags(&s_tile_ready[1], cudaEventDisableTiming);
                    cudaEventCreateWithFlags(&s_buf_free[0], cudaEventDisableTiming);
                    cudaEventCreateWithFlags(&s_buf_free[1], cudaEventDisableTiming);
                    cudaEventCreateWithFlags(&s_stream_done, cudaEventDisableTiming);
                    s_dbuf = true;
                    mexPrintf("[Lever A] active: double-buffered streaming "
                              "(%d tail tiles, owned pinned host copy of %lld entries).\n",
                              s_n_tiles - s_pref_tiles, n_tail);
                } else {
                    /* Roll back: s_h_* still point at the borrowed arrays here (we
                     * repoint only on full success) -> clean synchronous path. */
                    if (pin_src)  cudaFreeHost(pin_src);
                    if (pin_g)    cudaFreeHost(pin_g);
                    if (pin_srcg) cudaFreeHost(pin_srcg);
                    if (pin_cidx) cudaFreeHost(pin_cidx);
                    if (s_d_src_tile2)   { cudaFree(s_d_src_tile2);   s_d_src_tile2 = NULL; }
                    if (s_d_g_tile2)     { cudaFree(s_d_g_tile2);     s_d_g_tile2 = NULL; }
                    if (s_d_g_tile8_2)   { cudaFree(s_d_g_tile8_2);   s_d_g_tile8_2 = NULL; }
                    if (s_d_srcg_tile2)  { cudaFree(s_d_srcg_tile2);  s_d_srcg_tile2 = NULL; }
                    if (s_d_c_idx_tile2) { cudaFree(s_d_c_idx_tile2); s_d_c_idx_tile2 = NULL; }
                    cudaGetLastError();
                    mexPrintf("[Lever A] setup failed -> synchronous streaming fallback.\n");
                }
                cudaGetLastError();   /* clear any soft alloc error -> clean sync path */
            }
        }
        }

        /* ---- R3: pinned ring staging for MMAP sources (best-effort) ----
         * The mmap counterpart of Lever A (mutually exclusive with it by the
         * s_h_is_mmap condition): bounded pinned memory (2 x tile_cap), a CPU
         * staging thread per SpMV, true async DMA. See the static block and
         * launch_otf_spmv for the pipeline. Env-gated FTLM_R3=1; any failure
         * rolls back to the synchronous single-buffer path. */
        if (!reuse) {
        s_r3 = false;
        {
            const char *r3 = getenv("FTLM_R3");
            bool r3_env = (r3 != NULL && r3[0] == '1');
            long long n_tail3 = n_entries_stream - s_pref_e_count;
            if (r3_env && s_h_is_mmap && !s_dbuf &&
                (s_n_tiles - s_pref_tiles) > 1 && n_tail3 > 0 && tile_cap > 0) {
                /* Slice-Default hardwareabhaengig (Robustheits-Audit
                 * 2026-07-11): 8 war eine B200-Node-Eichung (viele Kerne);
                 * auf 4-8-Kern-Workstations oversubscriben 8 memcpy-Threads
                 * die CPU neben MATLAB+Stager. hc/2, geklemmt auf 1..8;
                 * FTLM_R3_THREADS behaelt Vorrang. */
                { unsigned hc = std::thread::hardware_concurrency();
                  s_r3_slices = (hc >= 2)
                      ? (int)fmin(8.0, fmax(1.0, (double)hc / 2.0)) : 1; }
                s_r3_tile_bytes = 0;   /* fuer den volumenbasierten Stall-Timeout */
                {
                    size_t bpe3 = (s_h_srcg != NULL) ? sizeof(unsigned int)
                                : sizeof(int) + ((s_h_g_idx8 != NULL)
                                    ? sizeof(unsigned char) : sizeof(unsigned short));
                    if (s_h_c_idx != NULL) bpe3 += sizeof(unsigned char);
                    s_r3_tile_bytes = (size_t)tile_cap * bpe3;
                }
                const char *ts = getenv("FTLM_R3_THREADS");
                if (ts != NULL) { int v = atoi(ts); if (v >= 1 && v <= 8) s_r3_slices = v; }
                bool ok = true;
                /* 1) two pinned staging sets, tile_cap-sized (bounded, OWNED --
                 *    we never pin the mmap pages themselves). */
                for (int b = 0; b < 2 && ok; b++) {
                    if (s_h_srcg != NULL) {
                        if (cudaHostAlloc(&s_r3_pin_srcg[b], (size_t)tile_cap * sizeof(unsigned int),
                                          cudaHostAllocDefault) != cudaSuccess) ok = false;
                    } else {
                        if (cudaHostAlloc(&s_r3_pin_src[b], (size_t)tile_cap * sizeof(int),
                                          cudaHostAllocDefault) != cudaSuccess) ok = false;
                        if (ok) {
                            size_t gsz = (s_h_g_idx8 != NULL) ? sizeof(unsigned char)
                                                              : sizeof(unsigned short);
                            if (cudaHostAlloc(&s_r3_pin_g[b], (size_t)tile_cap * gsz,
                                              cudaHostAllocDefault) != cudaSuccess) ok = false;
                        }
                    }
                    if (ok && s_h_c_idx != NULL)
                        if (cudaHostAlloc(&s_r3_pin_cidx[b], (size_t)tile_cap * sizeof(unsigned char),
                                          cudaHostAllocDefault) != cudaSuccess) ok = false;
                }
                /* 2) second ping-pong device tile set (same statics as Lever A;
                 *    the two pipelines are mutually exclusive). */
                if (ok) {
                    if (s_h_srcg != NULL) {
                        if (cudaMalloc(&s_d_srcg_tile2, (size_t)tile_cap * sizeof(unsigned int)) != cudaSuccess) ok = false;
                    } else {
                        if (cudaMalloc(&s_d_src_tile2, (size_t)tile_cap * sizeof(int)) != cudaSuccess) ok = false;
                        if (ok) {
                            if (s_h_g_idx8 != NULL) { if (cudaMalloc(&s_d_g_tile8_2, (size_t)tile_cap * sizeof(unsigned char))  != cudaSuccess) ok = false; }
                            else                    { if (cudaMalloc(&s_d_g_tile2,   (size_t)tile_cap * sizeof(unsigned short)) != cudaSuccess) ok = false; }
                        }
                    }
                    if (ok && s_h_c_idx != NULL)
                        if (cudaMalloc(&s_d_c_idx_tile2, (size_t)tile_cap * sizeof(unsigned char)) != cudaSuccess) ok = false;
                }
                if (ok) {
                    /* 3) non-blocking streams + events (identical roles to
                     *    Lever A; ordering via events only). */
                    cudaStreamCreateWithFlags(&s_copy_stream, cudaStreamNonBlocking);
                    cudaStreamCreateWithFlags(&s_compute_stream, cudaStreamNonBlocking);
                    cudaEventCreateWithFlags(&s_tile_ready[0], cudaEventDisableTiming);
                    cudaEventCreateWithFlags(&s_tile_ready[1], cudaEventDisableTiming);
                    cudaEventCreateWithFlags(&s_buf_free[0], cudaEventDisableTiming);
                    cudaEventCreateWithFlags(&s_buf_free[1], cudaEventDisableTiming);
                    cudaEventCreateWithFlags(&s_stream_done, cudaEventDisableTiming);
                    s_r3 = true;
                    mexPrintf("[R3] active: pinned ring staging over mmap "
                              "(%d tail tiles, ring depth 2, %d memcpy slice(s)).\n",
                              s_n_tiles - s_pref_tiles, s_r3_slices);
                } else {
                    for (int b = 0; b < 2; b++) {
                        if (s_r3_pin_src[b])  { cudaFreeHost(s_r3_pin_src[b]);  s_r3_pin_src[b]  = NULL; }
                        if (s_r3_pin_g[b])    { cudaFreeHost(s_r3_pin_g[b]);    s_r3_pin_g[b]    = NULL; }
                        if (s_r3_pin_srcg[b]) { cudaFreeHost(s_r3_pin_srcg[b]); s_r3_pin_srcg[b] = NULL; }
                        if (s_r3_pin_cidx[b]) { cudaFreeHost(s_r3_pin_cidx[b]); s_r3_pin_cidx[b] = NULL; }
                    }
                    if (s_d_src_tile2)   { cudaFree(s_d_src_tile2);   s_d_src_tile2 = NULL; }
                    if (s_d_g_tile2)     { cudaFree(s_d_g_tile2);     s_d_g_tile2 = NULL; }
                    if (s_d_g_tile8_2)   { cudaFree(s_d_g_tile8_2);   s_d_g_tile8_2 = NULL; }
                    if (s_d_srcg_tile2)  { cudaFree(s_d_srcg_tile2);  s_d_srcg_tile2 = NULL; }
                    if (s_d_c_idx_tile2) { cudaFree(s_d_c_idx_tile2); s_d_c_idx_tile2 = NULL; }
                    cudaGetLastError();
                    mexPrintf("[R3] setup failed -> synchronous streaming fallback.\n");
                }
                cudaGetLastError();   /* clear any soft alloc error */
            }
        }
        }

        if (reuse) {
            s_kept_table = false;         /* adopted by this init */
            s_last_init_reused = true;
            mexPrintf("[keep-table] streaming table reused (%d tiles, %lld entries; "
                      "prefix %lld entries resident).\n",
                      s_n_tiles, n_entries_stream, s_pref_e_count);
        }

        mxGPUDestroyGPUArray(g_diag); mxGPUDestroyGPUArray(g_ro); mxGPUDestroyGPUArray(g_npr);
        mxGPUDestroyGPUArray(g_epr); mxGPUDestroyGPUArray(g_eoff);
        mxGPUDestroyGPUArray(g_Vre); mxGPUDestroyGPUArray(g_Vim);
        mxGPUDestroyGPUArray(g_rhoRe); mxGPUDestroyGPUArray(g_rhoIm); mxGPUDestroyGPUArray(g_sqrt);
        s_init = true; mexLock(); mexAtExit(cleanup_all);
    }

    /* ============================================================
     * BLOCK_LANCZOS
     * ============================================================ */
    else if (strcmp(mode, "block_lanczos") == 0)
    {
        if (!s_init)
            mexErrMsgIdAndTxt("clut_block_pg_Ih:run", "Call 'init' first!");

        const mxGPUArray *g_V0r = mxGPUCreateFromMxArray(prhs[1]);
        const mxGPUArray *g_V0i = mxGPUCreateFromMxArray(prhs[2]);
        const mwSize *dims_V0   = mxGPUGetDimensions(g_V0r);
        long long n = (long long)dims_V0[0];
        int B = (mxGPUGetNumberOfDimensions(g_V0r) > 1) ? (int)dims_V0[1] : 1;
        int M_lz = (int)mxGetScalar(prhs[3]);

        /* Split-V0 (July 2026): an EMPTY V0_re means the start vector was
         * already loaded chunk-wise into s_d_v_re/_im via 'set_v0' (MATLAB
         * caps ONE gpuArray at 2^31-1 elements; the kernel's own 64-bit
         * buffer has no such limit -- dodec s=3/2 M=1..5 d>=4 and the
         * icosahedron-s=5 blocks need this).
         * Segmented-V0 B-unlock (2026-07-10): on the REAL path 'set_v0'
         * scatters per COLUMN into the interleaved layout, so preloaded
         * V0 works for ANY B <= B_batch (the driver fills all columns).
         * The complex path keeps the historical B == 1 contract. */
        int v0_preloaded = (mxGPUGetNumberOfElements(g_V0r) == 0);
        if (v0_preloaded) {
            if (!s_is_real && s_B_batch != 1)
                mexErrMsgIdAndTxt("clut_block_pg_Ih:B",
                    "complex block_lanczos with preloaded V0 requires B_batch == 1.");
            n = s_n_basis;
            /* Optional 5th arg: actual column count of this Krylov block
             * (the driver's LAST R-chunk can be narrower than B_batch). */
            B = (nrhs > 4) ? (int)mxGetScalar(prhs[4]) : s_B_batch;
            if (B < 1 || B > s_B_batch)
                mexErrMsgIdAndTxt("clut_block_pg_Ih:B",
                    "preloaded block_lanczos: B = %d outside [1, B_batch = %d].",
                    B, s_B_batch);
        }
        if (n != s_n_basis)
            mexErrMsgIdAndTxt("clut_block_pg_Ih:dim",
                "V0 has %lld rows, expected n_basis = %lld!", n, s_n_basis);
        if (B > s_B_batch)
            mexErrMsgIdAndTxt("clut_block_pg_Ih:B",
                "B = %d exceeds B_batch = %d!", B, s_B_batch);
        if (M_lz > n) M_lz = (int)n;

        int spmv_blocks    = (s_n_reps  + SPMV_BS - 1) / SPMV_BS;
        int vec_blocks     = (int)((s_n_basis + 255) / 256);
        int reduce_blocks  = (int)((s_n_basis + FUSED_BS - 1) / FUSED_BS);

        /* OTF kernel uses d_irrep threads per output rep. With
         * SPMV_BS = 64 and d = 5 that gives 12 reps per block (4 idle
         * threads), 16 reps for d = 4, 21 for d = 3, 64 for d = 1.
         * Grid size scales accordingly. */
        s_otf_bs = otf_block_size(s_d_irrep);
        int otf_reps_per_block = s_otf_bs / s_d_irrep;   /* exact: lcm multiple of d */
        if (otf_reps_per_block < 1) otf_reps_per_block = 1;

        /* Real path: prhs[2] (V0_im) arrives EMPTY -- one transpose, real
         * dot/scale variants, no *_im buffers. The complex path below is
         * untouched. */
        if (s_is_real && mxGPUGetNumberOfElements(g_V0i) != 0)
            mexErrMsgIdAndTxt("clut_block_pg_Ih:real",
                "block_lanczos: real path (init with empty V_im) expects an "
                "empty V0_im (pass single([]) / an empty gpuArray).");
        /* COMPLEX path with a direct V0: V0_im must match V0_re element-for-
         * element. Without this guard the transpose below reads n*B floats
         * from an empty gpuArray's (NULL) data pointer -- an illegal access
         * on most GPUs, silent garbage on others (found by test_split_v0 on
         * the RTX 4000; the same call had sailed through on the B200). */
        if (!s_is_real && !v0_preloaded &&
            mxGPUGetNumberOfElements(g_V0i) != mxGPUGetNumberOfElements(g_V0r))
            mexErrMsgIdAndTxt("clut_block_pg_Ih:complex",
                "block_lanczos: complex path needs V0_im with the same size "
                "as V0_re (this init had a non-empty V_im).");

        /* Stage V0 (col-major) -> interleaved by reading the borrowed device
         * pointers directly (transpose's src is read-only), so no s_d_tmp_col
         * bounce buffer is needed. g_V0r/i stay valid until the destroys below:
         * the transpose launches are enqueued on the default stream first, and
         * the subsequent cudaFree inside mxGPUDestroyGPUArray synchronises. */
        /* FP16 + preloaded V0 is supported: 'set_v0' scatters via st_vec. */
        if (!v0_preloaded) {
            if (s_fp16)
                transpose_col2interleaved<<<vec_blocks, 256>>>(
                    (__half *)s_d_v_re, (const float *)mxGPUGetDataReadOnly(g_V0r), n, B,
                    s_vt_presc);   /* raw draw: 2^-k, cancels in the normalisation */
            else
                transpose_col2interleaved<<<vec_blocks, 256>>>(
                    s_d_v_re, (const float *)mxGPUGetDataReadOnly(g_V0r), n, B, 1.0f);
            if (!s_is_real)
                transpose_col2interleaved<<<vec_blocks, 256>>>(
                    s_d_v_im, (const float *)mxGPUGetDataReadOnly(g_V0i), n, B, 1.0f);
        }

        mxGPUDestroyGPUArray(g_V0r);
        mxGPUDestroyGPUArray(g_V0i);

        /* Normalise V0 */
        {
            if (s_is_real) {
                if (s_fp16)
                    fused_dot_partial_real<<<reduce_blocks, FUSED_BS>>>(
                        s_d_partial, (const __half *)s_d_v_re, (const __half *)s_d_v_re, n, B);
                else
                    fused_dot_partial_real<<<reduce_blocks, FUSED_BS>>>(
                        s_d_partial, s_d_v_re, s_d_v_re, n, B);
            } else
                fused_dot_partial_cplx<<<reduce_blocks, FUSED_BS>>>(
                    s_d_partial, s_d_v_re, s_d_v_im, s_d_v_re, s_d_v_im, n, B);
            reduce_partial<<<1, B>>>(s_d_alpha, s_d_partial,
                                     reduce_blocks, B);
            float h_nrm2[MAX_B];
            cudaMemcpy(h_nrm2, s_d_alpha, B * sizeof(float),
                       cudaMemcpyDeviceToHost);
            float h_inv[MAX_B];
            for (int b = 0; b < B; b++)
                h_inv[b] = 1.0f / sqrtf(h_nrm2[b]);
            cudaMemcpy(s_d_alpha, h_inv, B * sizeof(float),
                       cudaMemcpyHostToDevice);
            if (s_is_real) {
                if (s_fp16)
                    scale_interleaved_real<<<vec_blocks, 256>>>(
                        (__half *)s_d_v_re, s_d_alpha, n, B);
                else
                    scale_interleaved_real<<<vec_blocks, 256>>>(
                        s_d_v_re, s_d_alpha, n, B);
            } else
                scale_interleaved_cplx<<<vec_blocks, 256>>>(
                    s_d_v_re, s_d_v_im, s_d_alpha, n, B);
        }

        cudaMemset(s_d_vp_re, 0,
                   (size_t)n * B * (s_fp16 ? sizeof(__half) : sizeof(float)));
        if (!s_is_real)
            cudaMemset(s_d_vp_im, 0, (size_t)n * B * sizeof(float));

        double *h_AL = (double *)mxCalloc((size_t)M_lz * B, sizeof(double));
        double *h_BE = (double *)mxCalloc((size_t)M_lz * B, sizeof(double));

        float h_alpha[MAX_B];
        float h_beta[MAX_B];
        float h_beta_prev[MAX_B];

        int nsteps = M_lz;
        memset(h_beta_prev, 0, sizeof(h_beta_prev));

        float *p_v_re  = s_d_v_re;
        float *p_v_im  = s_d_v_im;
        float *p_vp_re = s_d_vp_re;
        float *p_vp_im = s_d_vp_im;
        float *p_w_re  = s_d_w_re;
        float *p_w_im  = s_d_w_im;

        for (int j = 0; j < M_lz; j++) {

            if (s_use_otf) {
                launch_otf_spmv(p_w_re, p_w_im, p_v_re, p_v_im, B, otf_reps_per_block);
            } else {
                heisenberg_clut_block_spmv_pg_Ih<<<spmv_blocks, SPMV_BS>>>(
                    p_w_re, p_w_im,
                    p_v_re, p_v_im,
                    s_d_diag_vals, s_d_rep_offsets, s_d_n_per_rep,
                    s_d_entries_per_rep, s_d_entry_offsets, s_d_src_idx,
                    s_d_M_re, s_d_M_im,
                    s_n_reps, B);
            }

            if (s_is_real) {
                if (s_fp16)
                    fused_dot_partial_real<<<reduce_blocks, FUSED_BS>>>(
                        s_d_partial, (const __half *)p_v_re, (const __half *)p_w_re, n, B);
                else
                    fused_dot_partial_real<<<reduce_blocks, FUSED_BS>>>(
                        s_d_partial, p_v_re, p_w_re, n, B);
            } else
                fused_dot_partial_cplx<<<reduce_blocks, FUSED_BS>>>(
                    s_d_partial, p_v_re, p_v_im, p_w_re, p_w_im, n, B);
            reduce_partial<<<1, B>>>(s_d_alpha, s_d_partial,
                                     reduce_blocks, B);

            cudaMemcpy(h_alpha, s_d_alpha, B * sizeof(float),
                       cudaMemcpyDeviceToHost);
            for (int b = 0; b < B; b++)
                h_AL[j + (size_t)b * M_lz] = (double)h_alpha[b];

            if (j > 0) {
                cudaMemcpy(s_d_beta_prev, h_beta_prev, B * sizeof(float),
                           cudaMemcpyHostToDevice);
            }

            if (s_is_real) {
                if (s_fp16)
                    fused_ortho_norm_partial_real<<<reduce_blocks, FUSED_BS>>>(
                        (__half *)p_w_re, (const __half *)p_v_re, (const __half *)p_vp_re,
                        s_d_alpha, s_d_beta_prev, s_d_partial,
                        n, B, (j > 0) ? 1 : 0);
                else
                    fused_ortho_norm_partial_real<<<reduce_blocks, FUSED_BS>>>(
                        p_w_re, p_v_re, p_vp_re,
                        s_d_alpha, s_d_beta_prev, s_d_partial,
                        n, B, (j > 0) ? 1 : 0);
            }
            else
                fused_ortho_norm_partial_cplx<<<reduce_blocks, FUSED_BS>>>(
                    p_w_re, p_w_im, p_v_re, p_v_im, p_vp_re, p_vp_im,
                    s_d_alpha, s_d_beta_prev, s_d_partial,
                    n, B, (j > 0) ? 1 : 0);

            reduce_partial<<<1, B>>>(s_d_beta, s_d_partial,
                                     reduce_blocks, B);

            float h_beta_sq[MAX_B];
            cudaMemcpy(h_beta_sq, s_d_beta, B * sizeof(float),
                       cudaMemcpyDeviceToHost);
            /* Launch/JIT failure guard (portability audit 2026-07-03): a
             * failed SpMV/reduction launch (forward-compat JIT rejection on
             * a new arch, WDDM TDR reset, OOM'd async alloc) would otherwise
             * feed SILENT garbage into AL/BE -- wrong physics that can still
             * look plausible downstream. The D2H copy above synchronised. */
            {
                cudaError_t it_err = cudaGetLastError();
                if (it_err != cudaSuccess)
                    mexErrMsgIdAndTxt("clut_block_pg_Ih:launch",
                        "block_lanczos: CUDA error at Lanczos step %d: %s",
                        j + 1, cudaGetErrorString(it_err));
            }
            int all_converged = 1;
            for (int b = 0; b < B; b++) {
                h_beta[b] = sqrtf(h_beta_sq[b]);
                h_BE[j + (size_t)b * M_lz] = (double)h_beta[b];
                if (h_beta[b] >= 1e-6f)
                    all_converged = 0;
            }

            if (all_converged) { nsteps = j + 1; break; }

            if (j < M_lz - 1) {
                float h_inv_beta[MAX_B];
                for (int b = 0; b < B; b++)
                    h_inv_beta[b] = 1.0f / h_beta[b];
                cudaMemcpy(s_d_beta, h_inv_beta, B * sizeof(float),
                           cudaMemcpyHostToDevice);
                if (s_is_real) {
                    if (s_fp16)
                        scale_interleaved_real<<<vec_blocks, 256>>>(
                            (__half *)p_w_re, s_d_beta, n, B);
                    else
                        scale_interleaved_real<<<vec_blocks, 256>>>(
                            p_w_re, s_d_beta, n, B);
                } else
                    scale_interleaved_cplx<<<vec_blocks, 256>>>(
                        p_w_re, p_w_im, s_d_beta, n, B);

                /* Pointer rotation: on the real path the *_im pointers are
                 * all NULL, so rotating them is a harmless no-op. */
                float *t1 = p_vp_re; p_vp_re = p_v_re; p_v_re = p_w_re; p_w_re = t1;
                float *t2 = p_vp_im; p_vp_im = p_v_im; p_v_im = p_w_im; p_w_im = t2;

                memcpy(h_beta_prev, h_beta, B * sizeof(float));
            }
        }

        cudaDeviceSynchronize();
        {
            cudaError_t end_err = cudaGetLastError();
            if (end_err != cudaSuccess)
                mexErrMsgIdAndTxt("clut_block_pg_Ih:launch",
                    "block_lanczos: CUDA error after the Lanczos loop: %s",
                    cudaGetErrorString(end_err));
        }

        s_d_v_re  = p_v_re;  s_d_v_im  = p_v_im;
        s_d_vp_re = p_vp_re; s_d_vp_im = p_vp_im;
        s_d_w_re  = p_w_re;  s_d_w_im  = p_w_im;

        plhs[0] = mxCreateDoubleMatrix(nsteps, B, mxREAL);
        plhs[1] = mxCreateDoubleMatrix(nsteps, B, mxREAL);
        double *AL = mxGetPr(plhs[0]);
        double *BE = mxGetPr(plhs[1]);
        for (int b = 0; b < B; b++)
            for (int j = 0; j < nsteps; j++) {
                AL[j + (size_t)b * nsteps] = h_AL[j + (size_t)b * M_lz];
                BE[j + (size_t)b * nsteps] = h_BE[j + (size_t)b * M_lz];
            }
        mxFree(h_AL);
        mxFree(h_BE);
    }

    /* ============================================================
     * SPMV_ONLY (smoke-test path: apply H once, return W)
     * ============================================================ */
    else if (strcmp(mode, "spmv") == 0)
    {
        if (!s_init)
            mexErrMsgIdAndTxt("clut_block_pg_Ih:run", "Call 'init' first!");

        const mxGPUArray *g_Vr = mxGPUCreateFromMxArray(prhs[1]);
        const mxGPUArray *g_Vi = mxGPUCreateFromMxArray(prhs[2]);
        const mwSize *dims_V   = mxGPUGetDimensions(g_Vr);
        long long n = (long long)dims_V[0];
        int B = (mxGPUGetNumberOfDimensions(g_Vr) > 1) ? (int)dims_V[1] : 1;

        if (n != s_n_basis)
            mexErrMsgIdAndTxt("clut_block_pg_Ih:dim",
                "V has %lld rows, expected n_basis = %lld!", n, s_n_basis);
        if (B > s_B_batch)
            mexErrMsgIdAndTxt("clut_block_pg_Ih:B",
                "B = %d exceeds B_batch = %d!", B, s_B_batch);

        int spmv_blocks = (s_n_reps  + SPMV_BS - 1) / SPMV_BS;
        int vec_blocks  = (int)((s_n_basis + 255) / 256);

        /* OTF grid sizing: d_irrep threads per output rep (see kernel). */
        s_otf_bs = otf_block_size(s_d_irrep);
        int otf_reps_per_block = s_otf_bs / s_d_irrep;   /* exact: lcm multiple of d */
        if (otf_reps_per_block < 1) otf_reps_per_block = 1;

        /* Real path: prhs[2] (V_im) arrives EMPTY; W_im is identically zero
         * and plhs[1] is returned as (calloc'd) zeros. */
        if (s_is_real && mxGPUGetNumberOfElements(g_Vi) != 0)
            mexErrMsgIdAndTxt("clut_block_pg_Ih:real",
                "spmv: real path (init with empty V_im) expects an empty V_im.");
        /* Complex path: V_im must match V_re (see the block_lanczos guard). */
        if (!s_is_real &&
            mxGPUGetNumberOfElements(g_Vi) != mxGPUGetNumberOfElements(g_Vr))
            mexErrMsgIdAndTxt("clut_block_pg_Ih:complex",
                "spmv: complex path needs V_im with the same size as V_re "
                "(this init had a non-empty V_im).");

        if (s_fp16)
            mexErrMsgIdAndTxt("clut_block_pg_Ih:fp16",
                "spmv: the standalone SpMV test mode is FP32-only -- unset "
                "FTLM_FP16 (the bit-identity gates run without it by design).");
        transpose_col2interleaved<<<vec_blocks, 256>>>(
            s_d_v_re, (const float *)mxGPUGetDataReadOnly(g_Vr), n, B, 1.0f);
        if (!s_is_real)
            transpose_col2interleaved<<<vec_blocks, 256>>>(
                s_d_v_im, (const float *)mxGPUGetDataReadOnly(g_Vi), n, B, 1.0f);

        mxGPUDestroyGPUArray(g_Vr);
        mxGPUDestroyGPUArray(g_Vi);

        if (s_use_otf) {
            launch_otf_spmv(s_d_w_re, s_d_w_im, s_d_v_re, s_d_v_im, B, otf_reps_per_block);
        } else {
            heisenberg_clut_block_spmv_pg_Ih<<<spmv_blocks, SPMV_BS>>>(
                s_d_w_re, s_d_w_im,
                s_d_v_re, s_d_v_im,
                s_d_diag_vals, s_d_rep_offsets, s_d_n_per_rep,
                s_d_entries_per_rep, s_d_entry_offsets, s_d_src_idx,
                s_d_M_re, s_d_M_im,
                s_n_reps, B);
        }

        cudaDeviceSynchronize();
        {
            cudaError_t sp_err = cudaGetLastError();
            if (sp_err != cudaSuccess)
                mexErrMsgIdAndTxt("clut_block_pg_Ih:launch",
                    "spmv: CUDA error: %s", cudaGetErrorString(sp_err));
        }

        /* Transpose interleaved -> column-major and copy back to MATLAB */
        plhs[0] = mxCreateNumericMatrix(n, B, mxSINGLE_CLASS, mxREAL);
        plhs[1] = mxCreateNumericMatrix(n, B, mxSINGLE_CLASS, mxREAL);
        float *out_re = (float *)mxGetData(plhs[0]);
        float *out_im = (float *)mxGetData(plhs[1]);

        /* interleaved -> column-major: inverse-transpose on host for simplicity */
        size_t bytes = (size_t)n * B * sizeof(float);
        float *h_w_re = (float *)mxMalloc(bytes);
        cudaMemcpy(h_w_re, s_d_w_re, bytes, cudaMemcpyDeviceToHost);
        if (s_is_real) {
            for (long long t = 0; t < n; t++)
                for (int b = 0; b < B; b++)
                    out_re[t + b * n] = h_w_re[t * B + b];
            /* out_im stays the zeros mxCreateNumericMatrix initialised. */
        } else {
            float *h_w_im = (float *)mxMalloc(bytes);
            cudaMemcpy(h_w_im, s_d_w_im, bytes, cudaMemcpyDeviceToHost);
            for (long long t = 0; t < n; t++)
                for (int b = 0; b < B; b++) {
                    out_re[t + b * n] = h_w_re[t * B + b];
                    out_im[t + b * n] = h_w_im[t * B + b];
                }
            mxFree(h_w_im);
        }
        mxFree(h_w_re);
    }

    /* ============================================================
     * SET_V0 (July 2026, split-V0 upload): copy ONE chunk of the initial
     * Lanczos vector into the kernel's own 64-bit s_d_v_re/_im buffer at a
     * given element offset. Blocks with n_basis > 2^31-1 (MATLAB's cap on
     * a single gpuArray) draw V0 in chunks and hand them over one by one;
     * 'block_lanczos' is then called with an EMPTY V0. B_batch must be 1
     * (interleaved layout == plain vector). Usage:
     *   cuda_...('set_v0', chunk_single_gpuArray, offset0based, to_im)
     * ============================================================ */
    else if (strcmp(mode, "set_v0") == 0) {
        if (!s_init)
            mexErrMsgIdAndTxt("clut_block_pg_Ih:run", "Call 'init' first!");
        const mxGPUArray *g_c = mxGPUCreateFromMxArray(prhs[1]);
        long long off   = (long long)mxGetScalar(prhs[2]);
        int       to_im = (nrhs > 3) && (mxGetScalar(prhs[3]) != 0.0);
        int       col   = (nrhs > 4) ? (int)mxGetScalar(prhs[4]) : 0;
        long long nc    = (long long)mxGPUGetNumberOfElements(g_c);
        if (mxGPUGetClassID(g_c) != mxSINGLE_CLASS)
            mexErrMsgIdAndTxt("clut_block_pg_Ih:setv0", "chunk must be single.");
        if (off < 0 || off + nc > s_n_basis)
            mexErrMsgIdAndTxt("clut_block_pg_Ih:setv0",
                "chunk [%lld, %lld) outside n_basis = %lld.", off, off + nc, s_n_basis);
        if (col < 0 || col >= s_B_batch)
            mexErrMsgIdAndTxt("clut_block_pg_Ih:setv0",
                "column %d outside B_batch = %d.", col, s_B_batch);
        /* Segmented-V0 B-unlock: B_batch > 1 is supported on the REAL path
         * (interleaved per-column scatter). The complex path keeps the
         * historical B = 1 contract (re/im draw-order convention). */
        if (s_B_batch > 1 && (to_im || !s_is_real))
            mexErrMsgIdAndTxt("clut_block_pg_Ih:setv0",
                "set_v0 with B_batch > 1 supports the REAL path only.");
        float *dst = to_im ? s_d_v_im : s_d_v_re;
        if (dst == NULL)
            mexErrMsgIdAndTxt("clut_block_pg_Ih:setv0",
                "target V0 buffer not allocated (real-path init has no v_im).");
        if (s_B_batch == 1 && !s_fp16) {
            /* Historical byte path (production M=1..5 split-V0): untouched. */
            cudaMemcpy(dst + off, mxGPUGetDataReadOnly(g_c),
                       (size_t)nc * sizeof(float), cudaMemcpyDeviceToDevice);
        } else {
            int blocks = (int)((nc + 255) / 256);
            const float *src = (const float *)mxGPUGetDataReadOnly(g_c);
            if (s_fp16)
                scatter_v0_col<<<blocks, 256>>>((__half *)dst, src, nc,
                                                s_B_batch, col, off,
                                                s_vt_presc);   /* raw draw: 2^-k */
            else
                scatter_v0_col<<<blocks, 256>>>(dst, src, nc,
                                                s_B_batch, col, off, 1.0f);
        }
        mxGPUDestroyGPUArray(g_c);
    }

    /* ============================================================
     * ABI_VERSION (July 2026): version handshake for the 64-bit
     * basis-offset ABI. RUN_FTLM_PG_SECTOR_GPU_IH calls this once per
     * session; a stale pre-audit MEX errors on the unknown mode, which
     * the .m side converts into a clear "run build_all" message --
     * preventing SILENT garbage from int64 rep_offsets being read as
     * int32 by an old binary.
     * ============================================================ */
    else if (strcmp(mode, "abi_version") == 0) {
        plhs[0] = mxCreateDoubleScalar((double)KERNEL_ABI_VERSION);
    }

    /* ============================================================
     * CLEANUP_KEEP_TABLE (keep-table, wave-2 [13])
     * ============================================================ */
    else if (strcmp(mode, "cleanup_keep_table") == 0) {
        /* Like 'cleanup' (no mexUnlock, see below) but PRESERVES the
         * irrep-independent streaming table state for the next same-table
         * init_skel_stream. After a non-streaming init this is a full
         * cleanup (nothing worth keeping). */
        cleanup_core(true);
    }

    else if (strcmp(mode, "keep_stats") == 0) {
        /* [reused; kept; prefix_entries] -- keep-table test observability. */
        plhs[0] = mxCreateDoubleMatrix(3, 1, mxREAL);
        double *o = mxGetPr(plhs[0]);
        o[0] = s_last_init_reused ? 1.0 : 0.0;
        o[1] = s_kept_table ? 1.0 : 0.0;
        o[2] = (double)s_pref_e_count;
    }

    else if (strcmp(mode, "l2_size") == 0) {
        /* Device L2 size in bytes (audit K2d: the driver's gather cap was
         * hard-calibrated to the AD104's 48 MB; B200 has ~126 MB). The
         * driver falls back to 48e6 when this mode is missing (stale MEX). */
        cudaDeviceProp prop;
        int dev = 0; cudaGetDevice(&dev);
        cudaGetDeviceProperties(&prop, dev);
        plhs[0] = mxCreateDoubleScalar((double)prop.l2CacheSize);
    }

    /* ============================================================
     * CLEANUP
     * ============================================================ */
    else if (strcmp(mode, "cleanup") == 0) {
        /* Free all device buffers via cleanup_all() but DO NOT mexUnlock().
         * mexUnlock() permits MATLAB to clear the MEX file from memory
         * between sectors; under memory pressure MATLAB sometimes does
         * exactly that, after which the next sector's
         *     exist('cuda_lanczos_clut_block_pg_Ih', 'file') == 3
         * assertion in RUN_FTLM_PG_SECTOR_GPU_IH fails. The MEX stays
         * locked for the rest of the session; the mexAtExit hook still
         * cleans up at MATLAB shutdown. */
        cleanup_all();
    }

    else {
        mexErrMsgIdAndTxt("clut_block_pg_Ih:mode",
            "Unknown mode '%s'. Expected 'init', 'init_ref', 'init_skel', "
            "'init_skel_ref', 'init_skel_b2', 'init_skel_stream', "
            "'block_lanczos', 'spmv', 'set_v0', 'abi_version', 'cleanup', "
            "'cleanup_keep_table', or 'keep_stats'.",
            mode);
    }
}
