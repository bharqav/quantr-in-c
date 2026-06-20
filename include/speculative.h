/**
 * @file speculative.h
 * @brief Speculative Decoding Engine for Baremetal AI.
 *
 * Employs a small, fast draft model (e.g. 0.5B) to speculate K tokens,
 * followed by a batched verification pass on the large target model (e.g. 30B),
 * accelerating token generation throughput on CPU.
 */

#ifndef SPECULATIVE_H
#define SPECULATIVE_H

#include "model.h"
#include "runtime.h"
#include "tokenizer.h"
#include "sampling.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    Transformer target;
    RuntimeContext target_rt;
    Transformer draft;
    RuntimeContext draft_rt;
    Tokenizer tokenizer;
    int has_tokenizer;
    int spec_k;
    RuntimeOptions options;
} SpeculativeEngine;

/**
 * @brief Initialize speculative decoding engine with target and draft models.
 */
int speculative_init(SpeculativeEngine* eng, const char* target_path, const char* draft_path, int spec_k, RuntimeOptions opt);

/**
 * @brief Free speculative engine resources.
 */
void speculative_free(SpeculativeEngine* eng);

/**
 * @brief Run speculative generation loop.
 */
int speculative_generate(SpeculativeEngine* eng, const char* prompt, int max_tokens, void (*on_token)(const char* token_str, int token_id, void* user_data), void* user_data);

/**
 * @brief Verify a draft token against target model distribution (speculative acceptance contract).
 * @return 1 if token is accepted, 0 if rejected.
 */
int speculative_verify_token(const float* target_logits, const float* draft_logits, int vocab_size, int draft_token, float temperature);

#ifdef __cplusplus
}
#endif

#endif /* SPECULATIVE_H */
