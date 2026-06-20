/**
 * @file speculative.c
 * @brief Speculative Decoding Engine implementation.
 */

#include "speculative.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int speculative_init(SpeculativeEngine* eng, const char* target_path, const char* draft_path, int spec_k, RuntimeOptions opt) {
    if (!eng || !target_path || !draft_path) return -1;
    memset(eng, 0, sizeof(*eng));

    eng->spec_k = (spec_k > 0 && spec_k <= 16) ? spec_k : 4;
    eng->options = opt;

    printf("[SPEC] Loading Target Model: %s\n", target_path);
    load_transformer(&eng->target, target_path);
    if (!eng->target.mapped_data) {
        fprintf(stderr, "[SPEC] Failed to load target model.\n");
        return -1;
    }
    runtime_init(&eng->target_rt, opt);

    printf("[SPEC] Loading Draft Model: %s\n", draft_path);
    load_transformer(&eng->draft, draft_path);
    if (!eng->draft.mapped_data) {
        fprintf(stderr, "[SPEC] Failed to load draft model.\n");
        free_transformer(&eng->target);
        return -1;
    }
    runtime_init(&eng->draft_rt, opt);

    if (eng->target.gguf_ctx && tokenizer_load_gguf(&eng->tokenizer, (gguf_context*)eng->target.gguf_ctx) == 0) {
        eng->has_tokenizer = 1;
    } else {
        eng->has_tokenizer = 0;
    }

    printf("[SPEC] Initialized Speculative Engine (Speculative K=%d)\n", eng->spec_k);
    return 0;
}

void speculative_free(SpeculativeEngine* eng) {
    if (!eng) return;
    free_transformer(&eng->target);
    free_transformer(&eng->draft);
    if (eng->has_tokenizer) {
        tokenizer_free(&eng->tokenizer);
    }
    memset(eng, 0, sizeof(*eng));
}

int speculative_generate(SpeculativeEngine* eng, const char* prompt, int max_tokens,
                         void (*on_token)(const char* token_str, int token_id, void* user_data),
                         void* user_data) {
    if (!eng || !prompt) return -1;

    SamplingState target_samp, draft_samp;
    sampling_state_init(&target_samp, 128);
    sampling_state_init(&draft_samp, 128);

    int prompt_tokens[1024];
    int n_prompt = 0;
    int token = 1;
    int pos = 0;

    if (eng->has_tokenizer) {
        n_prompt = tokenizer_encode(&eng->tokenizer, prompt, prompt_tokens, 1024);
    }

    if (n_prompt > 0) {
        prefill_runtime(&eng->target_rt, &eng->target, prompt_tokens, n_prompt);
        prefill_runtime(&eng->draft_rt, &eng->draft, prompt_tokens, n_prompt);
        for (int i = 0; i < n_prompt; i++) {
            sampling_state_push(&target_samp, prompt_tokens[i]);
            sampling_state_push(&draft_samp, prompt_tokens[i]);
        }
        token = prompt_tokens[n_prompt - 1];
        pos = n_prompt;
    }

    int emitted = 0;
    int K = eng->spec_k;
    int draft_tokens[16];

    while (emitted < max_tokens) {
        /* 1. Draft Phase: generate K speculative candidate tokens with fast draft model */
        int draft_curr_token = token;
        int draft_curr_pos = pos;

        for (int k = 0; k < K; k++) {
            float* draft_logits = forward_runtime(&eng->draft_rt, &eng->draft, draft_curr_token, draft_curr_pos % eng->draft.config.seq_len);
            int next_draft = sample_next(draft_logits, eng->draft.config.vocab_size, &eng->options, &draft_samp);
            draft_tokens[k] = next_draft;
            sampling_state_push(&draft_samp, next_draft);
            draft_curr_token = next_draft;
            draft_curr_pos++;
        }

        /* 2. Verification Phase: run target model on current token and draft candidates */
        int accepted = 0;
        int curr_token = token;
        int curr_pos = pos;

        for (int k = 0; k < K && emitted + accepted < max_tokens; k++) {
            float* target_logits = forward_runtime(&eng->target_rt, &eng->target, curr_token, curr_pos % eng->target.config.seq_len);
            int target_pred = sample_next(target_logits, eng->target.config.vocab_size, &eng->options, &target_samp);

            if (target_pred == draft_tokens[k]) {
                /* Exact match! Accept speculative token */
                accepted++;
                sampling_state_push(&target_samp, target_pred);

                if (on_token && eng->has_tokenizer) {
                    char piece[256] = {0};
                    tokenizer_decode_append(&eng->tokenizer, target_pred, piece, sizeof(piece));
                    on_token(piece, target_pred, user_data);
                }

                curr_token = target_pred;
                curr_pos++;
            } else {
                /* Mismatch: reject subsequent draft tokens, accept target's ground-truth prediction */
                sampling_state_push(&target_samp, target_pred);

                if (on_token && eng->has_tokenizer) {
                    char piece[256] = {0};
                    tokenizer_decode_append(&eng->tokenizer, target_pred, piece, sizeof(piece));
                    on_token(piece, target_pred, user_data);
                }

                curr_token = target_pred;
                curr_pos++;
                accepted++;
                break;
            }
        }

        if (accepted == 0) {
            /* Fallback single step */
            float* target_logits = forward_runtime(&eng->target_rt, &eng->target, curr_token, curr_pos % eng->target.config.seq_len);
            int target_pred = sample_next(target_logits, eng->target.config.vocab_size, &eng->options, &target_samp);
            sampling_state_push(&target_samp, target_pred);

            if (on_token && eng->has_tokenizer) {
                char piece[256] = {0};
                tokenizer_decode_append(&eng->tokenizer, target_pred, piece, sizeof(piece));
                on_token(piece, target_pred, user_data);
            }
            curr_token = target_pred;
            curr_pos++;
            accepted = 1;
        }

        token = curr_token;
        pos = curr_pos;
        emitted += accepted;
    }

    sampling_state_free(&target_samp);
    sampling_state_free(&draft_samp);

    return emitted;
}

int speculative_verify_token(const float* target_logits, const float* draft_logits, int vocab_size, int draft_token, float temperature) {
    if (!target_logits || vocab_size <= 0 || draft_token < 0 || draft_token >= vocab_size) return 0;

    if (temperature <= 0.0f) {
        /* Greedy verification: accept if draft token is the exact argmax of target model */
        int target_argmax = 0;
        float max_val = target_logits[0];
        for (int i = 1; i < vocab_size; i++) {
            if (target_logits[i] > max_val) {
                max_val = target_logits[i];
                target_argmax = i;
            }
        }
        return (target_argmax == draft_token) ? 1 : 0;
    }

    /* Stochastic verification via Leviathan et al. rejection sampling */
    if (!draft_logits) {
        /* If draft logits not provided, check argmax */
        int target_argmax = 0;
        float max_val = target_logits[0];
        for (int i = 1; i < vocab_size; i++) {
            if (target_logits[i] > max_val) {
                max_val = target_logits[i];
                target_argmax = i;
            }
        }
        return (target_argmax == draft_token) ? 1 : 0;
    }

    /* Compute max values for numerical stability */
    float target_max = target_logits[0], draft_max = draft_logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (target_logits[i] > target_max) target_max = target_logits[i];
        if (draft_logits[i] > draft_max) draft_max = draft_logits[i];
    }

    double sum_p = 0.0, sum_q = 0.0;
    for (int i = 0; i < vocab_size; i++) {
        sum_p += exp((double)(target_logits[i] - target_max) / (double)temperature);
        sum_q += exp((double)(draft_logits[i] - draft_max) / (double)temperature);
    }

    double p_target = exp((double)(target_logits[draft_token] - target_max) / (double)temperature) / sum_p;
    double q_draft = exp((double)(draft_logits[draft_token] - draft_max) / (double)temperature) / sum_q;

    if (q_draft <= 0.0) return 0;
    double ratio = p_target / q_draft;
    if (ratio >= 1.0) return 1;

    /* Sample uniform random number in [0, 1) */
    double u = (double)rand() / ((double)RAND_MAX + 1.0);
    return (u < ratio) ? 1 : 0;
}
