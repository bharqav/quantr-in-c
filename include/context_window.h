#ifndef CONTEXT_WINDOW_H
#define CONTEXT_WINDOW_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int max_seq_len;          /* Model context limit (e.g., 2048, 4096) */
    int current_pos;          /* Current KV cache position / active tokens */
    int system_tokens_count;  /* Number of fixed system prompt tokens */
    int user_turns_count;     /* Number of user turns in current session */
    int* token_history;       /* Dynamic array storing full token history */
    int history_capacity;     /* Capacity of history array */
    int history_length;       /* Total count of tokens accumulated in history */
    size_t kv_bytes_used;     /* Memory footprint of active KV cache in bytes */
    size_t kv_bytes_total;    /* Total memory allocated for KV cache */
} ContextWindow;

/* Initialize context window with maximum sequence length */
int context_window_init(ContextWindow* cw, int max_seq_len, size_t kv_bytes_total);

/* Free resources allocated for context window */
void context_window_free(ContextWindow* cw);

/* Reset context window back to beginning (or keeping system prompt) */
void context_window_reset(ContextWindow* cw, int keep_system_prompt);

/* Push a single token to context window */
int context_window_push(ContextWindow* cw, int token);

/* Push an array of tokens to context window */
int context_window_push_batch(ContextWindow* cw, const int* tokens, int count);

/* Mark the start of a user turn */
void context_window_start_turn(ContextWindow* cw);

/* Set the fixed system prompt token count */
void context_window_set_system_tokens(ContextWindow* cw, int count);

/* Calculate context window usage statistics */
void context_window_get_stats(const ContextWindow* cw, float* pct_used, int* free_tokens, float* kv_mb_used);

/* Render visual ANSI context window HUD bar to console */
void context_window_render_hud(const ContextWindow* cw);

/* Sliding window eviction: slide history if context limit is approaching */
int context_window_slide(ContextWindow* cw, int reserve_tokens);

#ifdef __cplusplus
}
#endif

#endif /* CONTEXT_WINDOW_H */
