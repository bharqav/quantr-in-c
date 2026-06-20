/**
 * @file fuzz_gguf.c
 * @brief Continuous Fuzzing Target for GGUF Binary Parser & Tokenizer.
 *
 * Can be compiled with libFuzzer (`clang -fsanitize=fuzzer,address`)
 * or run as a standalone fuzz harness for regression testing.
 */

#include "gguf.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* LLVMFuzzerTestOneInput entrypoint for libFuzzer / AFL++ */
int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (size < 4) return 0;

    /* Copy to mutable buffer */
    void* buf = malloc(size);
    if (!buf) return 0;
    memcpy(buf, data, size);

    /* Test GGUF initialization */
    gguf_context* ctx = gguf_init_from_buffer(buf, size);
    if (ctx) {
        /* Exercise querying methods on parsed context */
        (void)gguf_find_tensor(ctx, "token_embd.weight");
        (void)gguf_get_val_u32(ctx, "general.alignment", 32);
        (void)gguf_get_val_f32(ctx, "llama.rope.freq_base", 10000.0f);
        (void)gguf_get_val_str(ctx, "general.architecture");

        char** arr = NULL;
        uint64_t arr_len = 0;
        (void)gguf_get_val_str_array(ctx, "tokenizer.ggml.tokens", &arr, &arr_len);

        gguf_free(ctx);
    }

    free(buf);
    return 0;
}

#ifndef LIBFUZZER
/* Standalone fuzz testing harness with synthetic corrupted payloads */
int main(void) {
    printf("[FUZZ] Running standalone GGUF mutation fuzz testing...\n");

    /* 1. Test empty & tiny inputs */
    uint8_t tiny[8] = { 'G', 'G', 'U', 'F', 0x03, 0x00, 0x00, 0x00 };
    LLVMFuzzerTestOneInput(tiny, sizeof(tiny));

    /* 2. Test corrupted header lengths */
    uint8_t corrupt_header[64];
    memset(corrupt_header, 0xFF, sizeof(corrupt_header));
    memcpy(corrupt_header, "GGUF", 4);
    corrupt_header[4] = 0x03; /* Version 3 */
    LLVMFuzzerTestOneInput(corrupt_header, sizeof(corrupt_header));

    /* 3. Run pseudo-random mutated payloads */
    srand(42);
    for (int iter = 0; iter < 1000; iter++) {
        size_t sz = 16 + (size_t)(rand() % 4096);
        uint8_t* payload = (uint8_t*)malloc(sz);
        for (size_t i = 0; i < sz; i++) {
            payload[i] = (uint8_t)(rand() % 256);
        }
        /* Occasionally inject valid magic */
        if (iter % 3 == 0) {
            memcpy(payload, "GGUF", 4);
        }
        LLVMFuzzerTestOneInput(payload, sz);
        free(payload);
    }

    printf("[FUZZ] PASSED: 1,000 synthetic mutated inputs parsed without memory fault.\n");
    return 0;
}
#endif
