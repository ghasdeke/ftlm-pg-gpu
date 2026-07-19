/*
 * cuda_lanczos_clut_block_pg_cplx.cu
 *
 * Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
 * and Helmholtz-Zentrum Dresden-Rossendorf e.V.
 *
 * Licensed under the Apache License, Version 2.0.
 *
 * ================================================================
 * Matrix-free Heisenberg BLOCK Lanczos on GPU (FP32) — PG variant
 *   for the complex-irrep case (k = 2*pi*p/N with p != 0 in general;
 *   the kernel also works for the real cases p = 0 and 2p = N but the
 *   real-only kernel cuda_lanczos_clut_block_pg.cu is preferred there
 *   because it avoids the imaginary-vector arithmetic).
 *
 * Two real FP32 vectors V_re and V_im represent the complex Krylov
 * state; the imaginary part is treated as a second interleaved buffer
 * with the same memory layout as the real one. The Lanczos coefficients
 * alpha and beta remain real because the symmetry-adapted Hamiltonian
 * is Hermitian.
 *
 * Phase factor handling:
 *   For each generated source state we compute the min-image translation
 *   h_min via cyclic digit shift; cos(k*h_min) and sin(k*h_min) are read
 *   from precomputed tables staged in shared memory (size 2*N floats).
 *   The forward-gather convention used here implements
 *       H[t, idx_a]  =  c_a * sqrt(L_r / L_a) * exp(+i * k * h_min),
 *   which (after derivation from the symmetry-adapted matrix element
 *   M(r, a_R) = sqrt(L_r/L_{a_R}) c_a exp(+i k h_a) plus the Hermiticity
 *   of H) is the conjugate of the build-time convention used in
 *   build_heisenberg_sparse_pg.m. The Python verification
 *   verify_spmv_pg_cplx.py reproduces this exact arithmetic with
 *   machine-precision agreement against the reference sparse H_pg.
 *
 * Modes:
 *   cuda_lanczos_clut_block_pg_cplx('init',
 *                                   block_base_gpu, block_mask_gpu,
 *                                   basis_gpu, orbit_lens_gpu,
 *                                   bonds_flat, N, d, s, J,
 *                                   dim, B_batch, p_irrep)
 *   [AL, BE] = cuda_lanczos_clut_block_pg_cplx('block_lanczos',
 *                                              V0_re_gpu, V0_im_gpu, M_lz)
 *   cuda_lanczos_clut_block_pg_cplx('cleanup')
 *
 * Compile (from MATLAB):
 *   mexcuda cuda_lanczos_clut_block_pg_cplx.cu -lcublas
 * ================================================================
 */

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <math.h>
#include <string.h>

/* M_PI is not provided by MSVC's <math.h> unless _USE_MATH_DEFINES is set
 * before the include; define our own to keep the file portable. */
#ifndef PG_PI
#define PG_PI 3.14159265358979323846
#endif

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
__constant__ float  c_cos_kh[MAX_SITES];   /* cos(k*h) for h = 0..N-1 */
__constant__ float  c_sin_kh[MAX_SITES];   /* sin(k*h) for h = 0..N-1 */

/* ================================================================
 * Persistent state
 * ================================================================ */
static cublasHandle_t s_blH          = NULL;
static int           *s_d_block_base = NULL;
static unsigned int  *s_d_block_mask = NULL;
static int           *s_d_basis      = NULL;
static int           *s_d_orbit_lens = NULL;
static float         *s_d_v_re       = NULL;
static float         *s_d_v_im       = NULL;
static float         *s_d_vp_re      = NULL;
static float         *s_d_vp_im      = NULL;
static float         *s_d_w_re       = NULL;
static float         *s_d_w_im       = NULL;
static int            s_dim          = 0;
static int            s_B_batch      = 0;
static bool           s_init         = false;

/* Reduction buffers (same as real kernel) */
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
    if (s_d_v_re)       cudaFree(s_d_v_re);
    if (s_d_v_im)       cudaFree(s_d_v_im);
    if (s_d_vp_re)      cudaFree(s_d_vp_re);
    if (s_d_vp_im)      cudaFree(s_d_vp_im);
    if (s_d_w_re)       cudaFree(s_d_w_re);
    if (s_d_w_im)       cudaFree(s_d_w_im);
    if (s_d_partial)    cudaFree(s_d_partial);
    if (s_d_alpha)      cudaFree(s_d_alpha);
    if (s_d_beta)       cudaFree(s_d_beta);
    if (s_d_beta_prev)  cudaFree(s_d_beta_prev);
    if (s_d_tmp_col)    cudaFree(s_d_tmp_col);
    if (s_blH)          cublasDestroy(s_blH);
    s_d_block_base = NULL; s_d_block_mask = NULL;
    s_d_basis = NULL; s_d_orbit_lens = NULL;
    s_d_v_re = NULL; s_d_v_im = NULL;
    s_d_vp_re = NULL; s_d_vp_im = NULL;
    s_d_w_re = NULL; s_d_w_im = NULL;
    s_d_partial = NULL; s_d_alpha = NULL;
    s_d_beta = NULL; s_d_beta_prev = NULL;
    s_d_tmp_col = NULL;
    s_blH = NULL; s_init = false;
}

/* ================================================================
 * Device helpers
 * ================================================================ */
__device__ __forceinline__ int clut_lookup(
    const int          * __restrict__ block_base,
    const unsigned int * __restrict__ block_mask,
    int state_a)
{
    int blk = state_a >> BLOCK_BITS;
    int bit = state_a & (CLT_BS - 1);
    unsigned int mask = __ldg(&block_mask[blk]);
    if (!(mask & (1u << bit))) return -1;
    unsigned int lower = mask & ((1u << bit) - 1u);
    return __ldg(&block_base[blk]) + __popc(lower);
}

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
 * CUDA kernel:  W = H_pg * V  for general complex irrep p
 *
 * Per-thread implementation of the complex SpMV. The cos/sin tables
 * c_cos_kh, c_sin_kh are staged into shared memory at the top of the
 * kernel; subsequent thread-divergent indexing by h_min then hits
 * shared memory instead of constant memory (which would serialize
 * divergent accesses).
 * ================================================================ */
__global__ void heisenberg_clut_block_spmv_pg_cplx(
    float              * __restrict__ W_re,
    float              * __restrict__ W_im,
    const float        * __restrict__ V_re,
    const float        * __restrict__ V_im,
    const int          * __restrict__ block_base,
    const unsigned int * __restrict__ block_mask,
    const int          * __restrict__ basis,
    const int          * __restrict__ orbit_lens,
    int dim,
    int B)
{
    __shared__ float s_cos_kh[MAX_SITES];
    __shared__ float s_sin_kh[MAX_SITES];
    if (threadIdx.x < c_N) {
        s_cos_kh[threadIdx.x] = c_cos_kh[threadIdx.x];
        s_sin_kh[threadIdx.x] = c_sin_kh[threadIdx.x];
    }
    __syncthreads();

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= dim) return;

    int state = __ldg(&basis[t]);
    int L_r   = __ldg(&orbit_lens[t]);
    float sqrt_L_r = sqrtf((float)L_r);

    /* Digit decomposition */
    int digits[MAX_SITES];
    {
        int tmp = state;
        for (int k = 0; k < c_N; k++) {
            digits[k] = tmp % c_d;
            tmp /= c_d;
        }
    }

    /* Accumulators (re and im for B vectors) */
    float result_re[MAX_B];
    float result_im[MAX_B];
    for (int b = 0; b < B; b++) {
        result_re[b] = 0.0f;
        result_im[b] = 0.0f;
    }

    /* Diagonal: real, no phase */
    float diag = 0.0f;
    for (int bd = 0; bd < c_nbonds; bd++) {
        float mi = (float)digits[c_bonds[2*bd]]     - c_s;
        float mj = (float)digits[c_bonds[2*bd + 1]] - c_s;
        diag += mi * mj;
    }
    float diag_coeff = c_J * diag;

    int t_base = t * B;
    for (int b = 0; b < B; b++) {
        result_re[b] = diag_coeff * V_re[t_base + b];
        result_im[b] = diag_coeff * V_im[t_base + b];
    }

    /* Off-diagonal with min-image + sqrt(L_r/L_a) + complex phase */
    float ss1 = c_ss1;
    for (int bd = 0; bd < c_nbonds; bd++) {
        int si = c_bonds[2*bd];
        int sj = c_bonds[2*bd + 1];
        int di = digits[si];
        int dj = digits[sj];
        float mi = (float)di - c_s;
        float mj = (float)dj - c_s;

        /* First ladder branch */
        if (di > 0 && dj < c_d_minus_1) {
            float mi_a = mi - 1.0f;
            float mj_a = mj + 1.0f;
            float coeff = 0.5f * c_J
                * sqrtf(ss1 - mi_a * (mi_a + 1.0f))
                * sqrtf(ss1 - mj_a * (mj_a - 1.0f));
            int state_a = state - c_powers[si] + c_powers[sj];
            int h_min;
            int rep_a = min_image_C_N(state_a, &h_min);
            int idx_a = clut_lookup(block_base, block_mask, rep_a);
            if (idx_a >= 0) {
                int L_a = __ldg(&orbit_lens[idx_a]);
                float norm = sqrt_L_r * rsqrtf((float)L_a);
                float cos_phi = s_cos_kh[h_min];
                float sin_phi = s_sin_kh[h_min];
                float alpha_re = coeff * norm * cos_phi;
                float alpha_im = coeff * norm * sin_phi;
                int a_base = idx_a * B;
                for (int b = 0; b < B; b++) {
                    float vr = V_re[a_base + b];
                    float vi = V_im[a_base + b];
                    result_re[b] += alpha_re * vr - alpha_im * vi;
                    result_im[b] += alpha_re * vi + alpha_im * vr;
                }
            }
        }

        /* Second ladder branch */
        if (di < c_d_minus_1 && dj > 0) {
            float mi_a = mi + 1.0f;
            float mj_a = mj - 1.0f;
            float coeff = 0.5f * c_J
                * sqrtf(ss1 - mi_a * (mi_a - 1.0f))
                * sqrtf(ss1 - mj_a * (mj_a + 1.0f));
            int state_a = state + c_powers[si] - c_powers[sj];
            int h_min;
            int rep_a = min_image_C_N(state_a, &h_min);
            int idx_a = clut_lookup(block_base, block_mask, rep_a);
            if (idx_a >= 0) {
                int L_a = __ldg(&orbit_lens[idx_a]);
                float norm = sqrt_L_r * rsqrtf((float)L_a);
                float cos_phi = s_cos_kh[h_min];
                float sin_phi = s_sin_kh[h_min];
                float alpha_re = coeff * norm * cos_phi;
                float alpha_im = coeff * norm * sin_phi;
                int a_base = idx_a * B;
                for (int b = 0; b < B; b++) {
                    float vr = V_re[a_base + b];
                    float vi = V_im[a_base + b];
                    result_re[b] += alpha_re * vr - alpha_im * vi;
                    result_im[b] += alpha_re * vi + alpha_im * vr;
                }
            }
        }
    }

    for (int b = 0; b < B; b++) {
        W_re[t_base + b] = result_re[b];
        W_im[t_base + b] = result_im[b];
    }
}

/* ================================================================
 * Transposition: column-major -> interleaved (same as real kernel)
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

#define FUSED_BS 256

/* ================================================================
 * Complex dot product Re[<V|W>] in INTERLEAVED layout, per chain b:
 *   alpha[b] = sum_t (V_re[t,b]*W_re[t,b] + V_im[t,b]*W_im[t,b])
 *
 * The imaginary part of <V|W> vanishes exactly for Hermitian H modulo
 * roundoff and is discarded here.
 * ================================================================ */
__global__ void fused_dot_partial_cplx(
    float       * __restrict__ partial,
    const float * __restrict__ V_re,
    const float * __restrict__ V_im,
    const float * __restrict__ W_re,
    const float * __restrict__ W_im,
    int dim, int B)
{
    __shared__ float sdata[FUSED_BS];

    int tid = threadIdx.x;
    int t   = blockIdx.x * blockDim.x + threadIdx.x;

    for (int b = 0; b < B; b++) {
        float sum = 0.0f;
        if (t < dim) {
            int idx = t * B + b;
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
    int b = threadIdx.x;
    if (b >= B) return;
    float sum = 0.0f;
    for (int i = 0; i < n_blocks; i++)
        sum += partial[i * B + b];
    result[b] = sum;
}

/* ================================================================
 * Orthogonalisation + norm in complex form.
 *   w -= alpha[b] * v      (alpha real)
 *   w -= beta_prev[b] * vp (if j > 0; beta_prev real)
 *   sum += w_re^2 + w_im^2
 * ================================================================ */
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
    int dim, int B, int use_vp)
{
    __shared__ float sdata[FUSED_BS];

    int tid = threadIdx.x;
    int t   = blockIdx.x * blockDim.x + threadIdx.x;

    for (int b = 0; b < B; b++) {
        float wr = 0.0f, wi = 0.0f;
        if (t < dim) {
            int idx = t * B + b;
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

/* ================================================================
 * Scale w_re, w_im by per-chain real scalar.
 * ================================================================ */
__global__ void scale_interleaved_cplx(
    float       * __restrict__ W_re,
    float       * __restrict__ W_im,
    const float * __restrict__ scale,
    int dim, int B)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= dim) return;
    int base = t * B;
    for (int b = 0; b < B; b++) {
        W_re[base + b] *= scale[b];
        W_im[base + b] *= scale[b];
    }
}

/* ================================================================
 * Shared init helper (extended for cos/sin tables)
 * ================================================================ */
static void init_common(int *h_bonds, int nb, int N, int d,
                        float sv, float J, int p_irrep)
{
    int h_pw[MAX_SITES];
    h_pw[0] = 1;
    for (int k = 1; k < N; k++) h_pw[k] = h_pw[k-1] * d;

    float h_cos[MAX_SITES] = {0};
    float h_sin[MAX_SITES] = {0};
    double k_phase = 2.0 * PG_PI * (double)p_irrep / (double)N;
    for (int h = 0; h < N; h++) {
        h_cos[h] = (float)cos(k_phase * h);
        h_sin[h] = (float)sin(k_phase * h);
    }

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
    cudaMemcpyToSymbol(c_cos_kh,    h_cos, N*sizeof(float));
    cudaMemcpyToSymbol(c_sin_kh,    h_sin, N*sizeof(float));
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
     * cuda_lanczos_clut_block_pg_cplx('init',
     *     block_base_gpu, block_mask_gpu, basis_gpu, orbit_lens_gpu,
     *     bonds_flat, N, d, s, J, dim, B_batch, p_irrep)
     * ============================================================ */
    if (strcmp(mode, "init") == 0)
    {
        if (s_init) cleanup_all();

        const mxGPUArray *g_bb = mxGPUCreateFromMxArray(prhs[1]);
        const mxGPUArray *g_bm = mxGPUCreateFromMxArray(prhs[2]);
        const mxGPUArray *g_bs = mxGPUCreateFromMxArray(prhs[3]);
        const mxGPUArray *g_ol = mxGPUCreateFromMxArray(prhs[4]);

        int *h_bonds  = (int *)mxGetData(prhs[5]);
        int n_entries = (int)mxGetNumberOfElements(prhs[5]);
        int nb        = n_entries / 2;
        int N         = (int)mxGetScalar(prhs[6]);
        int d         = (int)mxGetScalar(prhs[7]);
        float sv      = (float)mxGetScalar(prhs[8]);
        float J       = (float)mxGetScalar(prhs[9]);
        s_dim         = (int)mxGetScalar(prhs[10]);
        s_B_batch     = (int)mxGetScalar(prhs[11]);
        int p_irrep   = (int)mxGetScalar(prhs[12]);

        if (s_B_batch > MAX_B) {
            mexErrMsgIdAndTxt("clut_block_pg_cplx:B",
                "B_batch = %d exceeds MAX_B = %d!", s_B_batch, MAX_B);
        }

        init_common(h_bonds, nb, N, d, sv, J, p_irrep);

        int bb_n = (int)mxGPUGetNumberOfElements(g_bb);
        int bm_n = (int)mxGPUGetNumberOfElements(g_bm);
        int bs_n = (int)mxGPUGetNumberOfElements(g_bs);
        int ol_n = (int)mxGPUGetNumberOfElements(g_ol);

        if (ol_n != bs_n) {
            mexErrMsgIdAndTxt("clut_block_pg_cplx:dim",
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
        cudaMalloc(&s_d_v_re,  vec_bytes);
        cudaMalloc(&s_d_v_im,  vec_bytes);
        cudaMalloc(&s_d_vp_re, vec_bytes);
        cudaMalloc(&s_d_vp_im, vec_bytes);
        cudaMalloc(&s_d_w_re,  vec_bytes);
        cudaMalloc(&s_d_w_im,  vec_bytes);

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
     * BLOCK_LANCZOS
     *
     * [AL, BE] = cuda_lanczos_clut_block_pg_cplx('block_lanczos',
     *                                            V0_re_gpu, V0_im_gpu, M_lz)
     * ============================================================ */
    else if (strcmp(mode, "block_lanczos") == 0)
    {
        if (!s_init)
            mexErrMsgIdAndTxt("clut_block_pg_cplx:run", "Call 'init' first!");

        const mxGPUArray *g_V0r = mxGPUCreateFromMxArray(prhs[1]);
        const mxGPUArray *g_V0i = mxGPUCreateFromMxArray(prhs[2]);
        const mwSize *dims_V0   = mxGPUGetDimensions(g_V0r);
        int n  = (int)dims_V0[0];
        int B  = (mxGPUGetNumberOfDimensions(g_V0r) > 1) ? (int)dims_V0[1] : 1;
        int M_lz = (int)mxGetScalar(prhs[3]);

        if (n != s_dim)
            mexErrMsgIdAndTxt("clut_block_pg_cplx:dim",
                "V0 has %d rows, expected %d!", n, s_dim);
        if (B > s_B_batch)
            mexErrMsgIdAndTxt("clut_block_pg_cplx:B",
                "B = %d exceeds B_batch = %d!", B, s_B_batch);
        if (M_lz > n) M_lz = n;

        int spmv_blocks   = (n + 255) / 256;
        int reduce_blocks = (n + FUSED_BS - 1) / FUSED_BS;

        /* Real part: column-major -> interleaved */
        cudaMemcpy(s_d_tmp_col, mxGPUGetDataReadOnly(g_V0r),
                   (size_t)n * B * sizeof(float), cudaMemcpyDeviceToDevice);
        transpose_col2interleaved<<<spmv_blocks, 256>>>(
            s_d_v_re, s_d_tmp_col, n, B);

        /* Imag part: column-major -> interleaved */
        cudaMemcpy(s_d_tmp_col, mxGPUGetDataReadOnly(g_V0i),
                   (size_t)n * B * sizeof(float), cudaMemcpyDeviceToDevice);
        transpose_col2interleaved<<<spmv_blocks, 256>>>(
            s_d_v_im, s_d_tmp_col, n, B);

        mxGPUDestroyGPUArray(g_V0r);
        mxGPUDestroyGPUArray(g_V0i);

        /* Normalize each vector: |v_b|^2 = sum (v_re^2 + v_im^2) */
        {
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
            scale_interleaved_cplx<<<spmv_blocks, 256>>>(
                s_d_v_re, s_d_v_im, s_d_alpha, n, B);
        }

        cudaMemset(s_d_vp_re, 0, (size_t)n * B * sizeof(float));
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

            /* W = H_pg * V (complex SpMV) */
            heisenberg_clut_block_spmv_pg_cplx<<<spmv_blocks, 256>>>(
                p_w_re, p_w_im,
                p_v_re, p_v_im,
                s_d_block_base, s_d_block_mask,
                s_d_basis, s_d_orbit_lens, n, B);

            /* alpha[b] = Re[<V|W>] = sum (V_re*W_re + V_im*W_im) */
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

            /* w -= alpha*v - beta_prev*vp; ||w||^2 */
            fused_ortho_norm_partial_cplx<<<reduce_blocks, FUSED_BS>>>(
                p_w_re, p_w_im,
                p_v_re, p_v_im,
                p_vp_re, p_vp_im,
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
                scale_interleaved_cplx<<<spmv_blocks, 256>>>(
                    p_w_re, p_w_im, s_d_beta, n, B);

                /* Pointer swap (re and im together) */
                float *t1 = p_vp_re; p_vp_re = p_v_re; p_v_re = p_w_re; p_w_re = t1;
                float *t2 = p_vp_im; p_vp_im = p_v_im; p_v_im = p_w_im; p_w_im = t2;

                memcpy(h_beta_prev, h_beta, B * sizeof(float));
            }
        }

        cudaDeviceSynchronize();

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
     * CLEANUP
     * ============================================================ */
    else if (strcmp(mode, "cleanup") == 0) {
        cleanup_all();
        mexUnlock();
    }

    else {
        mexErrMsgIdAndTxt("clut_block_pg_cplx:mode",
            "Unknown mode '%s'. Expected 'init', 'block_lanczos', or 'cleanup'.", mode);
    }
}
