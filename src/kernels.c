#include "kernels.h"

#include <math.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

static void matmul_ref(float* out, const float* x, const float* w, int n, int d, int threads) {
    (void)threads;
    for (int i = 0; i < n; i++) {
        const float* row = w + (size_t)i * (size_t)d;
        float sum = 0.0f;
        for (int j = 0; j < d; j++) {
            sum += row[j] * x[j];
        }
        out[i] = sum;
    }
}

static void matmul_omp(float* out, const float* x, const float* w, int n, int d, int threads) {
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads > 0 ? threads : 1) schedule(static)
#endif
    for (int i = 0; i < n; i++) {
        const float* row = w + (size_t)i * (size_t)d;
        float sum = 0.0f;
#if defined(__AVX512F__)
        __m512 acc = _mm512_setzero_ps();
        int j = 0;
        for (; j + 16 <= d; j += 16) {
            __m512 a = _mm512_loadu_ps(row + j);
            __m512 b = _mm512_loadu_ps(x + j);
            acc = _mm512_fmadd_ps(a, b, acc);
        }
        sum += _mm512_reduce_add_ps(acc);
        for (; j < d; j++) {
            sum += row[j] * x[j];
        }
#elif defined(__AVX2__)
        __m256 acc = _mm256_setzero_ps();
        int j = 0;
        for (; j + 8 <= d; j += 8) {
            __m256 a = _mm256_loadu_ps(row + j);
            __m256 b = _mm256_loadu_ps(x + j);
            acc = _mm256_fmadd_ps(a, b, acc);
        }
        float tmp[8];
        _mm256_storeu_ps(tmp, acc);
        for (int k = 0; k < 8; k++) {
            sum += tmp[k];
        }
        for (; j < d; j++) {
            sum += row[j] * x[j];
        }
#else
        for (int j = 0; j < d; j++) {
            sum += row[j] * x[j];
        }
#endif
        out[i] = sum;
    }
}

#if defined(__aarch64__) || defined(_M_ARM64)
static void matmul_neon(float* out, const float* x, const float* w, int n, int d, int threads) {
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads > 0 ? threads : 1) schedule(static)
#endif
    for (int i = 0; i < n; i++) {
        const float* row = w + (size_t)i * (size_t)d;
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        int j = 0;
        for (; j + 8 <= d; j += 8) {
            float32x4_t a0 = vld1q_f32(row + j);
            float32x4_t b0 = vld1q_f32(x + j);
            acc0 = vfmaq_f32(acc0, a0, b0);
            float32x4_t a1 = vld1q_f32(row + j + 4);
            float32x4_t b1 = vld1q_f32(x + j + 4);
            acc1 = vfmaq_f32(acc1, a1, b1);
        }
        acc0 = vaddq_f32(acc0, acc1);
        float sum = vaddvq_f32(acc0);
        for (; j < d; j++) {
            sum += row[j] * x[j];
        }
        out[i] = sum;
    }
}

static float dot_neon(const float* a, const float* b, int n) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        float32x4_t va0 = vld1q_f32(a + i);
        float32x4_t vb0 = vld1q_f32(b + i);
        acc0 = vfmaq_f32(acc0, va0, vb0);
        float32x4_t va1 = vld1q_f32(a + i + 4);
        float32x4_t vb1 = vld1q_f32(b + i + 4);
        acc1 = vfmaq_f32(acc1, va1, vb1);
    }
    acc0 = vaddq_f32(acc0, acc1);
    float s = vaddvq_f32(acc0);
    for (; i < n; i++) {
        s += a[i] * b[i];
    }
    return s;
}

static void rmsnorm_neon(float* o, const float* x, const float* weight, int size) {
    float32x4_t ss_vec = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= size; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        ss_vec = vfmaq_f32(ss_vec, v, v);
    }
    float ss = vaddvq_f32(ss_vec);
    for (; i < size; i++) {
        ss += x[i] * x[i];
    }
    ss = ss / (float)size + 1e-5f;
    float scale = 1.0f / sqrtf(ss);
    float32x4_t vscale = vdupq_n_f32(scale);
    i = 0;
    for (; i + 4 <= size; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t vw = vld1q_f32(weight + i);
        vst1q_f32(o + i, vmulq_f32(vw, vmulq_f32(vx, vscale)));
    }
    for (; i < size; i++) {
        o[i] = weight[i] * (x[i] * scale);
    }
}

static void softmax_neon(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    float inv = 1.0f / sum;
    float32x4_t vinv = vdupq_n_f32(inv);
    int i = 0;
    for (; i + 4 <= size; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        vst1q_f32(x + i, vmulq_f32(v, vinv));
    }
    for (; i < size; i++) {
        x[i] *= inv;
    }
}
#endif /* __aarch64__ || _M_ARM64 */

static float dot_ref(const float* a, const float* b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) {
        s += a[i] * b[i];
    }
    return s;
}

static void rmsnorm_ref(float* o, const float* x, const float* weight, int size) {
    float ss = 0.0f;
    for (int i = 0; i < size; i++) {
        ss += x[i] * x[i];
    }
    ss = ss / (float)size + 1e-5f;
    float scale = 1.0f / sqrtf(ss);
    for (int i = 0; i < size; i++) {
        o[i] = weight[i] * (x[i] * scale);
    }
}

static void softmax_ref(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < size; i++) {
        x[i] *= inv;
    }
}

static const KernelOps REF_OPS = {matmul_ref, dot_ref, rmsnorm_ref, softmax_ref, "ref"};
static const KernelOps AVX2_OPS = {matmul_omp, dot_ref, rmsnorm_ref, softmax_ref, "avx2"};
static const KernelOps AVX512_OPS = {matmul_omp, dot_ref, rmsnorm_ref, softmax_ref, "avx512"};
#if defined(__aarch64__) || defined(_M_ARM64)
static const KernelOps NEON_OPS = {matmul_neon, dot_neon, rmsnorm_neon, softmax_neon, "neon"};
#endif

int cpu_supports_avx2(void) {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("avx2");
#else
    return 0;
#endif
#else
    return 0;
#endif
}

int cpu_supports_avx512(void) {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("avx512f");
#else
    return 0;
#endif
#else
    return 0;
#endif
}

int cpu_supports_neon(void) {
#if defined(__aarch64__) || defined(_M_ARM64)
    return 1;  /* NEON is mandatory on AArch64 */
#else
    return 0;
#endif
}

int gpu_available(void) {
#ifdef _WIN32
    /* Check if NVIDIA CUDA (nvcuda.dll) or Vulkan (vulkan-1.dll) is present */
    HMODULE h = LoadLibraryA("nvcuda.dll");
    if (h) {
        FreeLibrary(h);
        return 1; /* NVIDIA CUDA GPU detected */
    }
    h = LoadLibraryA("vulkan-1.dll");
    if (h) {
        FreeLibrary(h);
        return 2; /* Vulkan GPU detected */
    }
    return 0;
#else
    return 0;
#endif
}

KernelBackend resolve_backend(KernelBackend requested) {
    if (requested == BACKEND_GPU) {
        if (gpu_available()) return BACKEND_GPU;
        // Fallback to AVX2 / AVX512
        if (cpu_supports_avx512()) return BACKEND_AVX512;
        if (cpu_supports_avx2()) return BACKEND_AVX2;
        return BACKEND_REF;
    }
    if (requested == BACKEND_AVX512 && cpu_supports_avx512()) {
        return BACKEND_AVX512;
    }
    if (requested == BACKEND_AVX2 && cpu_supports_avx2()) {
        return BACKEND_AVX2;
    }
    if (requested == BACKEND_AVX512 && cpu_supports_avx2()) {
        return BACKEND_AVX2;
    }
    if (requested == BACKEND_NEON && cpu_supports_neon()) {
        return BACKEND_NEON;
    }
    /* Auto-detect best available backend on fallback */
    if (cpu_supports_avx2()) return BACKEND_AVX2;
    if (cpu_supports_neon()) return BACKEND_NEON;
    return requested == BACKEND_REF ? BACKEND_REF : BACKEND_REF;
}

const KernelOps* kernels_for_backend(KernelBackend backend) {
    switch (backend) {
        case BACKEND_GPU:
        case BACKEND_AVX512:
            return &AVX512_OPS;
        case BACKEND_AVX2:
            return &AVX2_OPS;
#if defined(__aarch64__) || defined(_M_ARM64)
        case BACKEND_NEON:
            return &NEON_OPS;
#endif
        default:
            return &REF_OPS;
    }
}
