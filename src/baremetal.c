#define BAREMETAL_BUILD_DLL
#include "baremetal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"
#include "runtime.h"
#include "sampling.h"
#include "tokenizer.h"
#include "benchmark.h"

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#endif

struct bm_engine {
    Transformer transformer;
    RuntimeContext runtime;
    Tokenizer tokenizer;
    int has_tokenizer;
    bm_stats_t last_stats;
};

bm_engine_t* bm_create(const char* model_path, const bm_config_t* config) {
    if (model_path == NULL) return NULL;

    bm_engine_t* e = (bm_engine_t*)calloc(1, sizeof(bm_engine_t));
    if (e == NULL) return NULL;

    load_transformer(&e->transformer, model_path);

    RuntimeOptions opt;
    memset(&opt, 0, sizeof(opt));
    opt.backend = config ? (KernelBackend)config->backend : BACKEND_AVX2;
    opt.num_threads = (config && config->num_threads > 0) ? config->num_threads : 8;
    opt.temperature = config ? config->temperature : 0.8f;
    opt.top_k = config ? config->top_k : 40;
    opt.top_p = config ? config->top_p : 0.9f;
    opt.repetition_penalty = config ? config->repetition_penalty : 1.05f;

    runtime_init(&e->runtime, opt);

    if (e->transformer.gguf_ctx != NULL) {
        if (tokenizer_load_gguf(&e->tokenizer, (gguf_context*)e->transformer.gguf_ctx) == 0) {
            e->has_tokenizer = 1;
        }
    }

    return e;
}

int bm_generate(bm_engine_t* e,
                const char* prompt,
                int max_tokens,
                bm_token_callback cb,
                void* user_data) {
    if (e == NULL || max_tokens <= 0) return -1;

    SamplingState s;
    sampling_state_init(&s, 128);

    int prompt_tokens[1024];
    int n_prompt = 0;
    int token = 1;
    int pos = 0;

    if (e->has_tokenizer && prompt != NULL) {
        n_prompt = tokenizer_encode(&e->tokenizer, prompt, prompt_tokens, 1024);
        prefill_runtime(&e->runtime, &e->transformer, prompt_tokens, n_prompt);
        for (int i = 0; i < n_prompt; i++) {
            sampling_state_push(&s, prompt_tokens[i]);
        }
        if (n_prompt > 0) {
            token = prompt_tokens[n_prompt - 1];
            pos = n_prompt;
        }
    }

    for (int i = 0; i < max_tokens; i++) {
        float* logits = forward_runtime(&e->runtime, &e->transformer, token, pos++ % e->transformer.config.seq_len);
        int next = sample_next(logits, e->transformer.config.vocab_size, &e->runtime.options, &s);
        sampling_state_push(&s, next);
        token = next;

        if (e->has_tokenizer) {
            char out[256] = {0};
            tokenizer_decode_append(&e->tokenizer, next, out, (int)sizeof(out));
            if (cb) cb(out, next, user_data);
        } else {
            char buf[32];
            snprintf(buf, sizeof(buf), "%d ", next);
            if (cb) cb(buf, next, user_data);
        }

#if defined(_WIN32)
        SetProcessWorkingSetSize(GetCurrentProcess(), (SIZE_T)-1, (SIZE_T)-1);
#endif
    }

    sampling_state_free(&s);
    return 0;
}

bm_stats_t bm_get_stats(const bm_engine_t* engine) {
    if (engine == NULL) {
        bm_stats_t empty = {0};
        return empty;
    }
    return engine->last_stats;
}

void bm_destroy(bm_engine_t* engine) {
    if (engine == NULL) return;
    if (engine->has_tokenizer) tokenizer_free(&engine->tokenizer);
    free_transformer(&engine->transformer);
    free(engine);
}
