#include "sampling.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static void softmax_local(float* x, int n) {
    float m = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > m) m = x[i];
    }
    float s = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - m);
        s += x[i];
    }
    float inv = 1.0f / s;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

void json_grammar_init(JsonGrammarState* g) {
    if (!g) return;
    memset(g, 0, sizeof(*g));
}

void json_grammar_update(JsonGrammarState* g, const char* token_piece) {
    if (!g || !token_piece) return;
    g->token_count++;
    for (const char* p = token_piece; *p != '\0'; p++) {
        char c = *p;
        if (g->in_string) {
            if (g->escape) {
                g->escape = 0;
            } else if (c == '\\') {
                g->escape = 1;
            } else if (c == '"') {
                g->in_string = 0;
            }
        } else {
            if (c == '"') {
                g->in_string = 1;
            } else if (c == '{') {
                g->depth++;
            } else if (c == '}') {
                if (g->depth > 0) g->depth--;
                if (g->depth == 0 && g->array_depth == 0) {
                    g->complete = 1;
                }
            } else if (c == '[') {
                g->array_depth++;
            } else if (c == ']') {
                if (g->array_depth > 0) g->array_depth--;
                if (g->depth == 0 && g->array_depth == 0) {
                    g->complete = 1;
                }
            }
        }
    }
}

void json_grammar_filter_logits(float* logits, int vocab_size, const JsonGrammarState* g) {
    if (!g || vocab_size <= 0) return;

    /* If JSON is starting (token_count == 0), strongly bias toward starting '{' or '[' */
    if (g->token_count == 0 && g->depth == 0 && g->array_depth == 0) {
        /* Boost early token candidates that begin JSON objects */
        for (int i = 0; i < vocab_size; i++) {
            /* Keep existing logit, but penalize tokens heavily if not typical JSON starters */
            if (i > 256) {
                logits[i] -= 2.0f;
            }
        }
    } else if (g->complete) {
        /* Once JSON top-level object is complete, bias heavily toward EOS / stop token (token 0/1/2) */
        if (vocab_size > 0) logits[0] += 5.0f;
        if (vocab_size > 1) logits[1] += 5.0f;
        if (vocab_size > 2) logits[2] += 5.0f;
    }
}

void sampling_state_init(SamplingState* s, int capacity) {
    memset(s, 0, sizeof(*s));
    if (capacity > 0) {
        s->recent_tokens = (float*)calloc((size_t)capacity, sizeof(float));
        s->recent_capacity = capacity;
    }
}

void sampling_state_free(SamplingState* s) {
    free(s->recent_tokens);
    memset(s, 0, sizeof(*s));
}

void sampling_state_push(SamplingState* s, int token) {
    if (s->recent_capacity <= 0) return;
    if (s->recent_count < s->recent_capacity) {
        s->recent_tokens[s->recent_count++] = (float)token;
        return;
    }
    memmove(s->recent_tokens, s->recent_tokens + 1, sizeof(float) * (size_t)(s->recent_capacity - 1));
    s->recent_tokens[s->recent_capacity - 1] = (float)token;
}

static void apply_penalties(float* logits, int vocab_size, const RuntimeOptions* opt, const SamplingState* state) {
    if (state->recent_count <= 0) return;
    for (int i = 0; i < state->recent_count; i++) {
        int tok = (int)state->recent_tokens[i];
        if (tok >= 0 && tok < vocab_size) {
            logits[tok] /= opt->repetition_penalty > 0.0f ? opt->repetition_penalty : 1.0f;
            logits[tok] -= opt->frequency_penalty;
        }
    }
}

int sample_next(float* logits, int vocab_size, const RuntimeOptions* opt, SamplingState* state) {
    apply_penalties(logits, vocab_size, opt, state);

    if (opt->json_mode) {
        JsonGrammarState g;
        json_grammar_init(&g);
        g.token_count = state->recent_count;
        json_grammar_filter_logits(logits, vocab_size, &g);
    }

    if (opt->temperature <= 0.0f) {
        int best = 0;
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > logits[best]) best = i;
        }
        return best;
    }

    for (int i = 0; i < vocab_size; i++) logits[i] /= opt->temperature;
    softmax_local(logits, vocab_size);

    int k = opt->top_k > 0 ? opt->top_k : vocab_size;
    if (k > vocab_size) k = vocab_size;
    if (k < vocab_size) {
        for (int t = 0; t < vocab_size - k; t++) {
            int min_i = 0;
            for (int i = 1; i < vocab_size; i++) if (logits[i] < logits[min_i]) min_i = i;
            logits[min_i] = 0.0f;
        }
    }

    if (opt->top_p > 0.0f && opt->top_p < 1.0f) {
        float cum = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            cum += logits[i];
            if (cum > opt->top_p) {
                for (int j = i + 1; j < vocab_size; j++) logits[j] = 0.0f;
                break;
            }
        }
    }

    float norm = 0.0f;
    for (int i = 0; i < vocab_size; i++) norm += logits[i];
    if (norm <= 0.0f) return rand() % vocab_size;
    for (int i = 0; i < vocab_size; i++) logits[i] /= norm;

    float r = (float)rand() / ((float)RAND_MAX + 1.0f);
    float cdf = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cdf += logits[i];
        if (r < cdf) return i;
    }
    return vocab_size - 1;
}
