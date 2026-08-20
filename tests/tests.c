/**
 * @file tests.c
 * @brief Comprehensive unit and integration test harness for Baremetal AI Engine.
 * Tests SIMD kernels, Quantization math (Q4_K, Q5_K, Q6_K, Q8_0), Tokenizer, and Engine E2E.
 */
#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kernels.h"
#include "model.h"
#include "runtime.h"
#include "quant.h"
#include "tokenizer.h"
#include "gguf.h"

static int test_softmax_sum(void) {
    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const KernelOps* k = kernels_for_backend(BACKEND_REF);
    k->softmax(x, 4);
    float sum = x[0] + x[1] + x[2] + x[3];
    return fabsf(sum - 1.0f) < 1e-5f ? 0 : -1;
}

static int test_rmsnorm_basic(void) {
    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float w[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float o[4];
    const KernelOps* k = kernels_for_backend(BACKEND_REF);
    k->rmsnorm(o, x, w, 4);
    return isfinite(o[0]) && isfinite(o[1]) && isfinite(o[2]) && isfinite(o[3]) ? 0 : -1;
}

static int test_fp16_conversion(void) {
    float vals[5] = {0.0f, 1.0f, -1.0f, 3.14159f, 0.0001f};
    for (int i = 0; i < 5; i++) {
        uint16_t h = fp32_to_fp16(vals[i]);
        float f = fp16_to_fp32(h);
        if (fabsf(vals[i] - f) > (fabsf(vals[i]) * 0.01f + 1e-4f)) {
            return -1;
        }
    }
    return 0;
}

static int test_q8_0_quant_dequant(void) {
    float in[32];
    float out[32];
    block_q8_0 b;
    for (int i = 0; i < 32; i++) {
        in[i] = ((float)i - 16.0f) * 0.125f;
    }
    quantize_row_q8_0(in, &b, 32);
    dequantize_row(out, &b, 8, 32);
    
    float max_err = 0.0f;
    for (int i = 0; i < 32; i++) {
        float err = fabsf(in[i] - out[i]);
        if (err > max_err) max_err = err;
    }
    return max_err < 0.05f ? 0 : -1;
}

static int test_q4_K_dequant(void) {
    block_q4_K b;
    memset(&b, 0, sizeof(b));
    b.d = fp32_to_fp16(0.5f);
    b.dmin = fp32_to_fp16(0.1f);
    for (int i = 0; i < K_SCALE_SIZE; i++) b.scales[i] = 1;
    for (int i = 0; i < QK_K/2; i++) b.qs[i] = 0x21; // low=1, high=2

    float out[256];
    dequantize_row(out, &b, 12, 256);
    
    for (int i = 0; i < 256; i++) {
        if (!isfinite(out[i])) return -1;
    }
    return 0;
}

static int test_q5_K_dequant(void) {
    block_q5_K b;
    memset(&b, 0, sizeof(b));
    b.d = fp32_to_fp16(0.25f);
    b.dmin = fp32_to_fp16(0.05f);
    for (int i = 0; i < K_SCALE_SIZE; i++) b.scales[i] = 1;
    for (int i = 0; i < QK_K/2; i++) b.qs[i] = 0x43; // low=3, high=4
    for (int i = 0; i < QK_K/8; i++) b.qh[i] = 0x01; // high bit

    float out[256];
    dequantize_row(out, &b, 13, 256);

    for (int i = 0; i < 256; i++) {
        if (!isfinite(out[i])) return -1;
    }
    return 0;
}

static int test_q6_K_dequant(void) {
    uint8_t raw[210];
    memset(raw, 0, sizeof(raw));
    // Set scale = 1, d = 0.5
    for (int i = 0; i < 8; i++) raw[192 + i] = 1;
    uint16_t d_raw = fp32_to_fp16(0.5f);
    memcpy(raw + 208, &d_raw, 2);
    for (int i = 0; i < 128; i++) raw[i] = 0x55;

    float out[256];
    dequantize_row(out, raw, 14, 256);

    for (int i = 0; i < 256; i++) {
        if (!isfinite(out[i])) return -1;
    }
    return 0;
}

static int test_vec_dot_q8_0(void) {
    float a_in[32], b_in[32];
    block_q8_0 a, b;
    float ref_dot = 0.0f;
    for (int i = 0; i < 32; i++) {
        a_in[i] = (float)i * 0.1f;
        b_in[i] = (float)(32 - i) * 0.1f;
        ref_dot += a_in[i] * b_in[i];
    }
    quantize_row_q8_0(a_in, &a, 32);
    quantize_row_q8_0(b_in, &b, 32);

    float simd_dot = vec_dot_q8_0_q8_0(&a, &b, 32);
    float err = fabsf(ref_dot - simd_dot);
    return err < (ref_dot * 0.05f) ? 0 : -1;
}

static int test_tokenizer_bpe_hash_lookup(void) {
    Tokenizer tok;
    memset(&tok, 0, sizeof(tok));
    tok.vocab_size = 4;
    tok.type = TOKENIZER_BPE;
    tok.vocab = (char**)malloc(4 * sizeof(char*));
    tok.bucket_next = (int*)malloc(4 * sizeof(int));
    for (int i = 0; i < 256; i++) tok.bucket_head[i] = -1;
    for (int i = 0; i < 4; i++) tok.bucket_next[i] = -1;

    tok.vocab[0] = strdup("<unk>");
    tok.vocab[1] = strdup("He");
    tok.vocab[2] = strdup("llo");
    tok.vocab[3] = strdup("Hello");

    for (int i = 0; i < 4; i++) {
        uint8_t fb = (uint8_t)tok.vocab[i][0];
        tok.bucket_next[i] = tok.bucket_head[fb];
        tok.bucket_head[fb] = i;
    }

    int tokens[16];
    int n = tokenizer_encode(&tok, "Hello", tokens, 16);
    int passed = (n == 1 && tokens[0] == 3);
    
    tokenizer_free(&tok);
    return passed ? 0 : -1;
}

static int test_deterministic_forward(void) {
    const char* model = "test_dummy.bin";
    write_dummy_model(model);
    Transformer t;
    load_transformer(&t, model);
    RuntimeContext rt;
    memset(&rt, 0, sizeof(rt));
    RuntimeOptions opt;
    memset(&opt, 0, sizeof(opt));
    opt.backend = BACKEND_REF;
    opt.num_threads = 1;
    opt.temperature = 0.0f;
    runtime_init(&rt, opt);
    float* a = forward_runtime(&rt, &t, 1, 0);
    float first = a[0];
    float* b = forward_runtime(&rt, &t, 1, 0);
    float second = b[0];
    free_transformer(&t);
    remove(model);
    return fabsf(first - second) < 1e-6f ? 0 : -1;
}

#include "context_window.h"

static int test_context_window_accounting(void) {
    ContextWindow cw;
    if (context_window_init(&cw, 512, 1024 * 1024) != 0) return -1;
    
    int sample_tokens[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    context_window_push_batch(&cw, sample_tokens, 10);
    if (cw.current_pos != 10 || cw.history_length != 10) {
        context_window_free(&cw);
        return -1;
    }

    float pct = 0.0f;
    int free_tok = 0;
    float kv_mb = 0.0f;
    context_window_get_stats(&cw, &pct, &free_tok, &kv_mb);
    if (free_tok != 502 || pct < 1.0f) {
        context_window_free(&cw);
        return -1;
    }

    context_window_free(&cw);
    return 0;
}

static int test_tokenizer_eos_decode(void) {
    Tokenizer tok;
    memset(&tok, 0, sizeof(tok));
    tok.vocab_size = 3;
    char* vocab[3] = {"<s>", "</s>", "\xe2\x96\x81hello"};
    tok.vocab = vocab;

    if (!tokenizer_is_eos(&tok, 1)) return -1;
    if (tokenizer_is_eos(&tok, 2)) return -1;

    char out[64] = {0};
    tokenizer_decode_append(&tok, 2, out, sizeof(out));
    if (strcmp(out, " hello") != 0) return -1;

    return 0;
}

#include "speculative.h"

static int test_cross_backend_parity(void) {
    const int N = 64;
    const int D = 128;
    float x[128];
    float w[64 * 128];
    float out_ref[64];
    float out_other[64];

    for (int i = 0; i < D; i++) x[i] = sinf((float)i * 0.1f);
    for (int i = 0; i < N * D; i++) w[i] = cosf((float)i * 0.05f) * 0.1f;

    const KernelOps* ops_ref = kernels_for_backend(BACKEND_REF);
    const KernelOps* ops_other = kernels_for_backend(resolve_backend(BACKEND_AVX2));

    ops_ref->matmul(out_ref, x, w, N, D, 1);
    ops_other->matmul(out_other, x, w, N, D, 1);

    for (int i = 0; i < N; i++) {
        if (fabsf(out_ref[i] - out_other[i]) > 1e-3f) {
            return -1;
        }
    }
    return 0;
}

static int test_speculative_verification(void) {
    const int vocab = 8;
    float target_logits[8] = {1.0f, 2.0f, 0.5f, 10.0f, 3.0f, 0.2f, 1.1f, 0.0f};
    float draft_logits[8]  = {1.1f, 1.9f, 0.4f, 9.8f,  2.9f, 0.3f, 1.0f, 0.1f};

    /* Greedy: draft_token 3 is the argmax (10.0), must be accepted */
    if (speculative_verify_token(target_logits, draft_logits, vocab, 3, 0.0f) != 1) return -1;

    /* Greedy: draft_token 1 is not argmax, must be rejected */
    if (speculative_verify_token(target_logits, draft_logits, vocab, 1, 0.0f) != 0) return -1;

    /* Stochastic: verify bounds */
    int acc_count = 0;
    for (int i = 0; i < 100; i++) {
        acc_count += speculative_verify_token(target_logits, draft_logits, vocab, 3, 0.7f);
    }
    if (acc_count < 90) return -1; // Top candidate should have high acceptance probability

    return 0;
}

static int test_kv_cache_quantization_parity(void) {
    const int N = 256;
    float k_fp32[256];
    float q_fp32[256];
    block_q8_0 k_q8[256 / QK8_0];
    block_q8_0 q_q8[256 / QK8_0];

    for (int i = 0; i < N; i++) {
        k_fp32[i] = sinf((float)i * 0.05f);
        q_fp32[i] = cosf((float)i * 0.03f);
    }

    quantize_row_q8_0(k_fp32, k_q8, N);
    quantize_row_q8_0(q_fp32, q_q8, N);

    float dot_fp32 = 0.0f;
    for (int i = 0; i < N; i++) dot_fp32 += k_fp32[i] * q_fp32[i];

    float dot_q8 = vec_dot_q8_0_q8_0(k_q8, q_q8, N);

    /* Quantized dot product relative error should be < 2% */
    float rel_err = fabsf(dot_fp32 - dot_q8) / (fabsf(dot_fp32) + 1e-6f);
    return (rel_err < 0.02f) ? 0 : -1;
}

int main(void) {
    int passed = 0;
    int total = 0;

#define RUN_TEST(fn) do { \
    total++; \
    printf("[TEST] %-35s ", #fn); \
    if (fn() == 0) { \
        printf("PASSED\n"); \
        passed++; \
    } else { \
        printf("FAILED\n"); \
    } \
} while(0)

    RUN_TEST(test_softmax_sum);
    RUN_TEST(test_rmsnorm_basic);
    RUN_TEST(test_fp16_conversion);
    RUN_TEST(test_q8_0_quant_dequant);
    RUN_TEST(test_q4_K_dequant);
    RUN_TEST(test_q5_K_dequant);
    RUN_TEST(test_q6_K_dequant);
    RUN_TEST(test_vec_dot_q8_0);
    RUN_TEST(test_tokenizer_bpe_hash_lookup);
    RUN_TEST(test_deterministic_forward);
    RUN_TEST(test_context_window_accounting);
    RUN_TEST(test_tokenizer_eos_decode);
    RUN_TEST(test_cross_backend_parity);
    RUN_TEST(test_speculative_verification);
    RUN_TEST(test_kv_cache_quantization_parity);

    printf("\nTest Summary: %d/%d tests passed.\n", passed, total);
    return passed == total ? 0 : 1;
}
