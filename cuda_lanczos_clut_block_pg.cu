/*
 * cuda_lanczos_clut_block_pg.cu
 *
 * Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
 * and Helmholtz-Zentrum Dresden-Rossendorf e.V.
 *
 * Licensed under the Apache License, Version 2.0 (the "License").
 *
 * ================================================================
 * Matrix-free Heisenberg BLOCK Lanczos on GPU (FP32) — PG variant
 *   with COMPRESSED lookup table (CLT) on the orbit-representative basis.
 *
 * Differences vs. release cuda_lanczos_clut_block.cu (k=0 / trivial irrep):
 *
 *   1. The CLT and basis array are built over orbit representatives
 *      (smallest integer label in each C_N orbit of an M sector), not
 *      over all M-sector states.
 *
 *   2. An additional per-rep array `orbit_lens` (int32, length DIM) holds
 *      the orbit length L_r for each representative. It is mapped to a
 *      persistent device buffer at `init` and consumed by the SpMV.
 *
 *   3. The SpMV kernel performs a min-image cyclic-shift search on each
 *      generated source state to find its orbit representative before
 *      the CLT lookup, and applies the symmetry-adapted matrix-element
 *      factor sqrt(L_r / L_{a_R}) on top of the unsymmetrized ladder
 *      coefficient c_a. The diagonal is unchanged because [H_diag, T] = 0.
 *
 * Phase factor for the trivial irrep (p = 0) is identically 1, so the
 * arithmetic remains real FP32, the random vectors are real Gaussian,
 * and the Lanczos coefficients alpha and beta stay real. The complex
 * pipeline for p != 0 will live in a sibling kernel (Milestone C).
 *
 * Block-Lanczos infrastructure (interleaved memory layout, fused dot/
 * orthogonalisation kernels, pointer swap) is identical to the release
 * cuda_lanczos_clut_block.cu.
 *
 * Modes:
 *   cuda_lanczos_clut_block_pg('init', block_base_gpu, block_mask_gpu,
 *                               basis_gpu, orbit_lens_gpu, bonds_flat,
 *                               N, d, s, J, dim, B_batch)
 *   [AL, BE] = cuda_lanczos_clut_block_pg('block_lanczos', V0_gpu, M_lz)
 *   cuda_lanczos_clut_block_pg('cleanup')
 *
 * Compile (from MATLAB):
 *   mexcuda cuda_lanczos_clut_block_pg.cu -lcublas
 * ================================================================
 */

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <string.h>

#define MAX_SITES 32
#define MAX_BONDS 64
#define BLOCK_BITS 5
#define CLT_BS    32
#define MAX_B     16

/* ================================================================
 * Constant memory: model + symmetry parameters
 * ================================================================ */
__constant__ int    c_bonds[2 * MAX_BONDS];
__constant__ int    c_powers[MAX_SITES];
__constant__ int    c_N;
__constant__ int    c_d;
__constant__ int    c_nbonds;
__constant__ float  c_s;
__constant__ float  c_J;
__constant__ float  c_ss1;
__constant__ int    c_d_minus_1;

/* ================================================================
 * Persistent state
 * ================================================================ */
static cublasHandle_t s_blH          = NULL;
static int           *s_d_block_base = NULL;
static unsigned int  *s_d_block_mask = NULL;
static int           *s_d_basis      = NULL;
static int           *s_d_orbit_lens = NULL;   /* NEW: L_r per rep */
static float         *s_d_v          = NULL;
static float         *s_d_vp         = NULL;
static float         *s_d_w          = NULL;
static int            s_dim          = 0;
static int            s_B_batch      = 0;
static bool           s_init         = false;

/* Reduction buffers */
static float *s_d_partial   = NULL;
static float *s_d_alpha     = NULL;
static float *s_d_beta      = NULL;
static float *s_d_beta_prev = NULL;
static float *s_d_tmp_col   = NULL;
static int    s_n_reduce_blocks = 0;

static void cleanup_all(void) {
    if (s_d_block_base) cudaFree(s_d_block_base);
    if (s_d_block_mask) cudaFree(s_d_block_mask);
    if (s_d_basis)      cudaFree(s_d_basis);
    if (s_d_orbit_lens) cudaFree(s_d_orbit_lens);
    if (s_d_v)          cudaFree(s_d_v);
    if (s_d_vp)         cudaFree(s_d_vp);
    if (s_d_w)          cudaFree(s_d_w);
    if (s_d_partial)    cudaFree(s_d_partial);
    if (s_d_alpha)      cudaFree(s_d_alpha);
    if (s_d_beta)       cudaFree(s_d_beta);
    if (s_d_beta_prev)  cudaFree(s_d_beta_prev);
    if (s_d_tmp_col)    cudaFree(s_d_tmp_col);
    if (s_blH)          cublasDestroy(s_blH);
    s_d_block_base = NULL; s_d_block_mask = NULL;
    s_d_basis = NULL; s_d_orbit_lens = NULL;
    s_d_v = NULL; s_d_vp = NULL; s_d_w = NULL;
    s_d_partial = NULL; s_d_alpha = NULL;
    s_d_beta = NULL; s_d_beta_prev = NULL;
    s_d_tmp_col = NULL;
    s_blH = NULL; s_init = false;
}

/* ================================================================
 * Device: compressed lookup (identical to release)
 * ================================================================ */
__device__ __forceinline__ int clut_lookup(
    const int          * __restrict__ block_base,
    const unsigned int * __restrict__ block_mask,
    int state_a)
{
    int blk = state_a >> BLOCK_BITS;
    int bit = state_a & (CLT_BS - 1);
    unsigned int mask = __ldg(&block_mask[blk]);

    if (!(mask & (1u << bit)))
        return -1;

    unsigned int lower = mask & ((1u << bit) - 1u);
    return __ldg(&block_base[blk]) + __popc(lower);
}

/* ================================================================
 * Device: cyclic min-image search for C_N translation symmetry.
 *
 * Treats `state` as an N-digit base-d_loc integer; iterates the cyclic
 * digit shift T(n) = floor(n/d_loc) + (n % d_loc) * d_loc^(N-1).
 * Returns the smallest integer label in the orbit and writes the
 * translation index h_min (smallest g >= 0 with T^h(state) = rep) to
 * *p_h_min. The k=0 SpMV ignores p_h_min; the complex-k SpMV (Milestone
 * C) will consume it.
 *
 * Loop bound c_N is uniform across the warp so this is divergence-free.
 * For states with stabilizer (orbit length L < N) the iteration revisits
 * the same orbit elements N/L times without changing rep or h_min, which
 * is correct but slightly wasteful. Such states are rare and the overhead
 * is negligible relative to the bond loop.
 * ================================================================ */
__device__ __forceinline__ int min_image_C_N(int state, int* p_h_min)
{
    int rep = state;
    int h_min = 0;
    int n_cur = state;
    int d_top = c_powers[c_N - 1];
    for (int g = 1; g < c_N; g++) {
        n_cur = (n_cur / c_d) + (n_cur % c_d) * d_top;
        if (n_cur < rep) {
            rep = n_cur;
            h_min = g;
        }
    }
    *p_h_min = h_min;
    return rep;
}

/* ================================================================
 * CUDA kernel:  W = H_pg * V  for p = 0 (trivial irrep, real arithmetic)
 *
 * Per-output-rep modifications vs. release SpMV:
 *   - load L_r = orbit_lens[t] and precompute sqrt(L_r)
 *   - for each off-diagonal source state state_a:
 *        rep_a = min_image_C_N(state_a)
 *        idx_a = clut_lookup(rep_a)
 *        if idx_a >= 0:
 *           L_a   = orbit_lens[idx_a]
 *           coeff *= sqrt_L_r / sqrtf(L_a)
 *
 * Diagonal contribution is identical to release (H_diag commutes with T,
 * so the matrix element on the rep basis equals the unsymmetrized one).
 * ================================================================ */
__global__ void heisenberg_clut_block_spmv_pg_k0(
    float              * __restrict__ W,
    const float        * __restrict__ V,
    const int          * __restrict__ block_base,
    const unsigned int * __restrict__ block_mask,
    const int          * __restrict__ basis,
    const int          * __restrict__ orbit_lens,
    int dim,
    int B)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= dim) return;

    int state = __ldg(&basis[t]);
    int L_r   = __ldg(&orbit_lens[t]);
    float sqrt_L_r = sqrtf((float)L_r);

    /* --- Digit decomposition (ONCE for all B vectors) --- */
    int digits[MAX_SITES];
    {
        int tmp = state;
        for (int k = 0; k < c_N; k++) {
            digits[k] = tmp % c_d;
            tmp /= c_d;
        }
    }

    /* --- Accumulators for B vectors --- */
    float result[MAX_B];
    for (int b = 0; b < B; b++)
        result[b] = 0.0f;

    /* --- Diagonal: J * sum_<ij> m_i*m_j   (factor 1 on rep basis) --- */
    float diag = 0.0f;
    for (int bd = 0; bd < c_nbonds; bd++) {
        float mi = (float)digits[c_bonds[2*bd]]     - c_s;
        float mj = (float)digits[c_bonds[2*bd + 1]] - c_s;
        diag += mi * mj;
    }
    float diag_coeff = c_J * diag;

    int t_base = t * B;
    for (int b = 0; b < B; b++)
        result[b] = diag_coeff * V[t_base + b];

    /* --- Off-diagonal: ladder terms + min-image + sqrt(L_r/L_a) --- */
    float ss1 = c_ss1;
    for (int bd = 0; bd < c_nbonds; bd++) {
        int si = c_bonds[2*bd];
        int sj = c_bonds[2*bd + 1];
        int di = digits[si];
        int dj = digits[sj];
        float mi = (float)di - c_s;
        float mj = (float)dj - c_s;

        /* S+_i S-_j */
        if (di > 0 && dj < c_d_minus_1) {
            float mi_a = mi - 1.0f;
            float mj_a = mj + 1.0f;
            float coeff = 0.5f * c_J
                * sqrtf(ss1 - mi_a * (mi_a + 1.0f))
                * sqrtf(ss1 - mj_a * (mj_a - 1.0f));
            int state_a = state - c_powers[si] + c_powers[sj];
            int h_unused;
            int rep_a = min_image_C_N(state_a, &h_unused);
            int idx_a = clut_lookup(block_base, block_mask, rep_a);
            if (idx_a >= 0) {
                int L_a = __ldg(&orbit_lens[idx_a]);
                float norm = sqrt_L_r * rsqrtf((float)L_a);
                float total = coeff * norm;
                int a_base = idx_a * B;
                for (int b = 0; b < B; b++)
                    result[b] += total * V[a_base + b];
            }
        }

        /* S-_i S+_j */
        if (di < c_d_minus_1 && dj > 0) {
            float mi_a = mi + 1.0f;
            float mj_a = mj - 1.0f;
            float coeff = 0.5f * c_J
                * sqrtf(ss1 - mi_a * (mi_a - 1.0f))
                * sqrtf(ss1 - mj_a * (mj_a + 1.0f));
            int state_a = state + c_powers[si] - c_powers[sj];
            int h_unused;
            int rep_a = min_image_C_N(state_a, &h_unused);
            int idx_a = clut_lookup(block_base, block_mask, rep_a);
            if (idx_a >= 0) {
                int L_a = __ldg(&orbit_lens[idx_a]);
                float norm = sqrt_L_r * rsqrtf((float)L_a);
                float total = coeff * norm;
                int a_base = idx_a * B;
                for (int b = 0; b < B; b++)
                    result[b] += total * V[a_base + b];
            }
        }
    }

    /* --- Write results (interleaved) --- */
    for (int b = 0; b < B; b++)
        W[t_base + b] = result[b];
}

/* ================================================================
 * The transposition, dot-product, orthogonalisation and norm kernels
 * below are byte-for-byte identical to release/cuda_lanczos_clut_block.cu.
 * They are inlined here so the file is self-contained.
 * ================================================================ */

__global__ void transpose_col2interleaved(
    float       * __restrict__ dst,
    const float * __restrict__ src,
    int dim, int B)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= dim) return;
    for (int b = 0; b < B; b++)
        dst[t * B + b] = src[t + b * dim];
}

__global__ void transpose_interleaved2col(
    float       * __restrict__ dst,
    const float * __restrict__ src,
    int dim, int B)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= dim) return;
    for (int b = 0; b < B; b++)
        dst[t + b * dim] = src[t * B + b];
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

#define FUSED_BS 256

__global__ void fused_dot_partial(
    float       * __restrict__ partial,
    const float * __restrict__ V,
    const float * __restrict__ W,
    int dim, int B)
{
    __shared__ float sdata[FUSED_BS];

    int tid = threadIdx.x;
    int t   = blockIdx.x * blockDim.x + threadIdx.x;

    for (int b = 0; b < B; b++) {
        float sum = 0.0f;
        if (t < dim)
            sum = V[t * B + b] * W[t * B + b];

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
    int b = threadIdx.x;
    if (b >= B) return;
    float sum = 0.0f;
    for (int i = 0; i < n_blocks; i++)
        sum += partial[i * B + b];
    result[b] = sum;
}

__global__ void fused_ortho_norm_partial(
    float       * __restrict__ W,
    const float * __restrict__ V,
    const float * __restrict__ Vp,
    const float * __restrict__ alpha,
    const float * __restrict__ beta_prev,
    float       * __restrict__ partial_nrm,
    int dim, int B, int use_vp)
{
    __shared__ float sdata[FUSED_BS];

    int tid = threadIdx.x;
    int t   = blockIdx.x * blockDim.x + threadIdx.x;

    for (int b = 0; b < B; b++) {
        float w_val = 0.0f;
        if (t < dim) {
            int idx = t * B + b;
            w_val = W[idx] - alpha[b] * V[idx];
            if (use_vp)
                w_val -= beta_prev[b] * Vp[idx];
            W[idx] = w_val;
        }

        sdata[tid] = w_val * w_val;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] += sdata[tid + s];
            __syncthreads();
        }
        if (tid == 0)
            partial_nrm[blockIdx.x * B + b] = sdata[0];
    }
}

__global__ void scale_interleaved(
    float       * __restrict__ W,
    const float * __restrict__ scale,
    int dim, int B)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= dim) return;
    int base = t * B;
    for (int b = 0; b < B; b++)
        W[base + b] *= scale[b];
}

/* ================================================================
 * Shared init helper
 * ================================================================ */
static void init_common(int *h_bonds, int nb, int N, int d,
                        float sv, float J)
{
    int h_pw[MAX_SITES];
    h_pw[0] = 1;
    for (int k = 1; k < N; k++) h_pw[k] = h_pw[k-1] * d;

    cudaMemcpyToSymbol(c_bonds,     h_bonds, 2*nb*sizeof(int));
    cudaMemcpyToSymbol(c_powers,    h_pw,    N*sizeof(int));
    cudaMemcpyToSymbol(c_N,         &N,  sizeof(int));
    cudaMemcpyToSymbol(c_d,         &d,  sizeof(int));
    cudaMemcpyToSymbol(c_nbonds,    &nb, sizeof(int));
    cudaMemcpyToSymbol(c_s,         &sv, sizeof(float));
    cudaMemcpyToSymbol(c_J,         &J,  sizeof(float));
    float ss1 = sv * (sv + 1.0f);
    cudaMemcpyToSymbol(c_ss1,       &ss1, sizeof(float));
    int dm1 = d - 1;
    cudaMemcpyToSymbol(c_d_minus_1, &dm1, sizeof(int));
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
     *
     * cuda_lanczos_clut_block_pg('init',
     *     block_base_gpu, block_mask_gpu, basis_gpu, orbit_lens_gpu,
     *     bonds_flat, N, d, s, J, dim, B_batch)
     * ============================================================ */
    if (strcmp(mode, "init") == 0)
    {
        if (s_init) cleanup_all();

        const mxGPUArray *g_bb = mxGPUCreateFromMxArray(prhs[1]);
        const mxGPUArray *g_bm = mxGPUCreateFromMxArray(prhs[2]);
        const mxGPUArray *g_bs = mxGPUCreateFromMxArray(prhs[3]);
        const mxGPUArray *g_ol = mxGPUCreateFromMxArray(prhs[4]);   /* NEW */

        int *h_bonds  = (int *)mxGetData(prhs[5]);
        int n_entries = (int)mxGetNumberOfElements(prhs[5]);
        int nb        = n_entries / 2;
        int N         = (int)mxGetScalar(prhs[6]);
        int d         = (int)mxGetScalar(prhs[7]);
        float sv      = (float)mxGetScalar(prhs[8]);
        float J       = (float)mxGetScalar(prhs[9]);
        s_dim         = (int)mxGetScalar(prhs[10]);
        s_B_batch     = (int)mxGetScalar(prhs[11]);

        if (s_B_batch > MAX_B) {
            mexErrMsgIdAndTxt("clut_block_pg:B",
                "B_batch = %d exceeds MAX_B = %d!", s_B_batch, MAX_B);
        }

        init_common(h_bonds, nb, N, d, sv, J);

        int bb_n = (int)mxGPUGetNumberOfElements(g_bb);
        int bm_n = (int)mxGPUGetNumberOfElements(g_bm);
        int bs_n = (int)mxGPUGetNumberOfElements(g_bs);
        int ol_n = (int)mxGPUGetNumberOfElements(g_ol);

        if (ol_n != bs_n) {
            mexErrMsgIdAndTxt("clut_block_pg:dim",
                "orbit_lens has %d entries, expected %d (matches basis).",
                ol_n, bs_n);
        }

        cudaMalloc(&s_d_block_base, bb_n * sizeof(int));
        cudaMalloc(&s_d_block_mask, bm_n * sizeof(unsigned int));
        cudaMalloc(&s_d_basis,      bs_n * sizeof(int));
        cudaMalloc(&s_d_orbit_lens, ol_n * sizeof(int));

        cudaMemcpy(s_d_block_base, mxGPUGetDataReadOnly(g_bb),
                   bb_n * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(s_d_block_mask, mxGPUGetDataReadOnly(g_bm),
                   bm_n * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(s_d_basis, mxGPUGetDataReadOnly(g_bs),
                   bs_n * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(s_d_orbit_lens, mxGPUGetDataReadOnly(g_ol),
                   ol_n * sizeof(int), cudaMemcpyDeviceToDevice);

        size_t vec_bytes = (size_t)s_dim * s_B_batch * sizeof(float);
        cudaMalloc(&s_d_v,  vec_bytes);
        cudaMalloc(&s_d_vp, vec_bytes);
        cudaMalloc(&s_d_w,  vec_bytes);

        s_n_reduce_blocks = (s_dim + FUSED_BS - 1) / FUSED_BS;
        cudaMalloc(&s_d_partial, (size_t)s_n_reduce_blocks * s_B_batch * sizeof(float));
        cudaMalloc(&s_d_alpha,      s_B_batch * sizeof(float));
        cudaMalloc(&s_d_beta,       s_B_batch * sizeof(float));
        cudaMalloc(&s_d_beta_prev,  s_B_batch * sizeof(float));
        cudaMalloc(&s_d_tmp_col,    vec_bytes);

        cublasCreate(&s_blH);

        mxGPUDestroyGPUArray(g_bb);
        mxGPUDestroyGPUArray(g_bm);
        mxGPUDestroyGPUArray(g_bs);
        mxGPUDestroyGPUArray(g_ol);

        s_init = true;
        mexLock();
        mexAtExit(cleanup_all);
    }

    /* ============================================================
     * BLOCK_LANCZOS  (identical structure to release; calls PG SpMV)
     * ============================================================ */
    else if (strcmp(mode, "block_lanczos") == 0)
    {
        if (!s_init)
            mexErrMsgIdAndTxt("clut_block_pg:run", "Call 'init' first!");

        const mxGPUArray *g_V0 = mxGPUCreateFromMxArray(prhs[1]);
        const mwSize *dims_V0  = mxGPUGetDimensions(g_V0);
        int n  = (int)dims_V0[0];
        int B  = (mxGPUGetNumberOfDimensions(g_V0) > 1) ? (int)dims_V0[1] : 1;
        int M_lz = (int)mxGetScalar(prhs[2]);

        if (n != s_dim)
            mexErrMsgIdAndTxt("clut_block_pg:dim",
                "V0 has %d rows, expected %d!", n, s_dim);
        if (B > s_B_batch)
            mexErrMsgIdAndTxt("clut_block_pg:B",
                "B = %d exceeds B_batch = %d!", B, s_B_batch);
        if (M_lz > n) M_lz = n;

        int spmv_blocks   = (n + 255) / 256;
        int reduce_blocks = (n + FUSED_BS - 1) / FUSED_BS;

        cudaMemcpy(s_d_tmp_col, mxGPUGetDataReadOnly(g_V0),
                   (size_t)n * B * sizeof(float), cudaMemcpyDeviceToDevice);
        mxGPUDestroyGPUArray(g_V0);

        transpose_col2interleaved<<<spmv_blocks, 256>>>(
            s_d_v, s_d_tmp_col, n, B);

        /* Normalize each of the B vectors. */
        {
            fused_dot_partial<<<reduce_blocks, FUSED_BS>>>(
                s_d_partial, s_d_v, s_d_v, n, B);
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
            scale_interleaved<<<spmv_blocks, 256>>>(
                s_d_v, s_d_alpha, n, B);
        }

        cudaMemset(s_d_vp, 0, (size_t)n * B * sizeof(float));

        double *h_AL = (double *)mxCalloc((size_t)M_lz * B, sizeof(double));
        double *h_BE = (double *)mxCalloc((size_t)M_lz * B, sizeof(double));

        float h_alpha[MAX_B];
        float h_beta[MAX_B];
        float h_beta_prev[MAX_B];

        int nsteps = M_lz;
        memset(h_beta_prev, 0, sizeof(h_beta_prev));

        float *ptr_v  = s_d_v;
        float *ptr_vp = s_d_vp;
        float *ptr_w  = s_d_w;

        for (int j = 0; j < M_lz; j++) {

            /* === W = H_pg * V (PG SpMV; min-image + sqrt(L_r/L_a)) === */
            heisenberg_clut_block_spmv_pg_k0<<<spmv_blocks, 256>>>(
                ptr_w, ptr_v, s_d_block_base, s_d_block_mask,
                s_d_basis, s_d_orbit_lens, n, B);

            /* === alpha[b] = dot(v_b, w_b) === */
            fused_dot_partial<<<reduce_blocks, FUSED_BS>>>(
                s_d_partial, ptr_v, ptr_w, n, B);
            reduce_partial<<<1, B>>>(s_d_alpha, s_d_partial,
                                     reduce_blocks, B);
            cudaMemcpy(h_alpha, s_d_alpha, B * sizeof(float),
                       cudaMemcpyDeviceToHost);
            for (int b = 0; b < B; b++)
                h_AL[j + (size_t)b * M_lz] = (double)h_alpha[b];

            /* === w -= alpha*v - beta_prev*vp, compute ||w||^2 === */
            if (j > 0) {
                cudaMemcpy(s_d_beta_prev, h_beta_prev, B * sizeof(float),
                           cudaMemcpyHostToDevice);
            }
            fused_ortho_norm_partial<<<reduce_blocks, FUSED_BS>>>(
                ptr_w, ptr_v, ptr_vp,
                s_d_alpha, s_d_beta_prev,
                s_d_partial,
                n, B, (j > 0) ? 1 : 0);

            reduce_partial<<<1, B>>>(s_d_beta, s_d_partial,
                                     reduce_blocks, B);

            float h_beta_sq[MAX_B];
            cudaMemcpy(h_beta_sq, s_d_beta, B * sizeof(float),
                       cudaMemcpyDeviceToHost);
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
                scale_interleaved<<<spmv_blocks, 256>>>(
                    ptr_w, s_d_beta, n, B);

                float *tmp = ptr_vp;
                ptr_vp = ptr_v;
                ptr_v  = ptr_w;
                ptr_w  = tmp;

                memcpy(h_beta_prev, h_beta, B * sizeof(float));
            }
        }

        cudaDeviceSynchronize();

        s_d_v  = ptr_v;
        s_d_vp = ptr_vp;
        s_d_w  = ptr_w;

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
     * CLEANUP
     * ============================================================ */
    else if (strcmp(mode, "cleanup") == 0) {
        cleanup_all();
        mexUnlock();
    }

    else {
        mexErrMsgIdAndTxt("clut_block_pg:mode",
            "Unknown mode '%s'. Expected 'init', 'block_lanczos', or 'cleanup'.", mode);
    }
}
