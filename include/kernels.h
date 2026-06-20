#ifndef KERNELS_H
#define KERNELS_H

#include "types.h"

typedef struct {
    void (*matmul)(float* out, const float* x, const float* w, int n, int d, int threads);
    float (*dot)(const float* a, const float* b, int n);
    void (*rmsnorm)(float* o, const float* x, const float* weight, int size);
    void (*softmax)(float* x, int size);
    const char* name;
} KernelOps;

const KernelOps* kernels_for_backend(KernelBackend backend);
int cpu_supports_avx2(void);
int cpu_supports_avx512(void);
int cpu_supports_neon(void);
int gpu_available(void);
KernelBackend resolve_backend(KernelBackend requested);

#endif
