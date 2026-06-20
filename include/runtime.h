#ifndef RUNTIME_H
#define RUNTIME_H

#include "kernels.h"
#include "types.h"

typedef struct {
    RuntimeOptions options;
    const KernelOps* kernels;
} RuntimeContext;

void runtime_init(RuntimeContext* rt, RuntimeOptions options);
float* forward_runtime(RuntimeContext* rt, Transformer* t, int token, int pos);
void prefill_runtime(RuntimeContext* rt, Transformer* t, const int* tokens, int n_tokens);

#endif
