#include "context_window.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int context_window_init(ContextWindow* cw, int max_seq_len, size_t kv_bytes_total) {
    if (cw == NULL || max_seq_len <= 0) return -1;
    memset(cw, 0, sizeof(*cw));
    cw->max_seq_len = max_seq_len;
    cw->kv_bytes_total = kv_bytes_total;
    cw->history_capacity = max_seq_len > 1024 ? max_seq_len * 2 : 2048;
    cw->token_history = (int*)malloc(sizeof(int) * (size_t)cw->history_capacity);
    if (cw->token_history == NULL) {
        return -1;
    }
    cw->current_pos = 0;
    cw->history_length = 0;
    cw->system_tokens_count = 0;
    cw->user_turns_count = 0;
    cw->kv_bytes_used = 0;
    return 0;
}

void context_window_free(ContextWindow* cw) {
    if (cw == NULL) return;
    if (cw->token_history != NULL) {
        free(cw->token_history);
        cw->token_history = NULL;
    }
    memset(cw, 0, sizeof(*cw));
}

void context_window_reset(ContextWindow* cw, int keep_system_prompt) {
    if (cw == NULL) return;
    if (keep_system_prompt && cw->system_tokens_count > 0 && cw->system_tokens_count <= cw->history_length) {
        cw->current_pos = cw->system_tokens_count;
        cw->history_length = cw->system_tokens_count;
        cw->user_turns_count = 0;
    } else {
        cw->current_pos = 0;
        cw->history_length = 0;
        cw->system_tokens_count = 0;
        cw->user_turns_count = 0;
    }
    if (cw->max_seq_len > 0) {
        cw->kv_bytes_used = (size_t)(((double)cw->current_pos / (double)cw->max_seq_len) * (double)cw->kv_bytes_total);
    }
}

int context_window_push(ContextWindow* cw, int token) {
    if (cw == NULL) return -1;
    if (cw->history_length >= cw->history_capacity) {
        int new_cap = cw->history_capacity * 2;
        int* new_hist = (int*)realloc(cw->token_history, sizeof(int) * (size_t)new_cap);
        if (new_hist == NULL) return -1;
        cw->token_history = new_hist;
        cw->history_capacity = new_cap;
    }
    cw->token_history[cw->history_length++] = token;
    cw->current_pos++;
    if (cw->max_seq_len > 0) {
        cw->kv_bytes_used = (size_t)(((double)cw->current_pos / (double)cw->max_seq_len) * (double)cw->kv_bytes_total);
    }
    return cw->current_pos;
}

int context_window_push_batch(ContextWindow* cw, const int* tokens, int count) {
    if (cw == NULL || tokens == NULL || count <= 0) return -1;
    for (int i = 0; i < count; i++) {
        if (context_window_push(cw, tokens[i]) < 0) {
            return -1;
        }
    }
    return cw->current_pos;
}

void context_window_start_turn(ContextWindow* cw) {
    if (cw == NULL) return;
    cw->user_turns_count++;
}

void context_window_set_system_tokens(ContextWindow* cw, int count) {
    if (cw == NULL || count < 0) return;
    cw->system_tokens_count = count;
}

void context_window_get_stats(const ContextWindow* cw, float* pct_used, int* free_tokens, float* kv_mb_used) {
    if (cw == NULL) return;
    int cur = cw->current_pos;
    int max_len = cw->max_seq_len > 0 ? cw->max_seq_len : 1;
    if (pct_used) *pct_used = ((float)cur / (float)max_len) * 100.0f;
    if (free_tokens) *free_tokens = max_len > cur ? max_len - cur : 0;
    if (kv_mb_used) {
        if (cw->kv_bytes_total > 0) {
            *kv_mb_used = (float)((double)cw->kv_bytes_used / (1024.0 * 1024.0));
        } else {
            *kv_mb_used = (float)((double)cur * 2.0 / 1024.0); /* rough estimate */
        }
    }
}

void context_window_render_hud(const ContextWindow* cw) {
    if (cw == NULL) return;
    float pct = 0.0f;
    int free_tok = 0;
    float kv_mb = 0.0f;
    context_window_get_stats(cw, &pct, &free_tok, &kv_mb);

    int bar_width = 24;
    int filled = (int)((pct / 100.0f) * (float)bar_width);
    if (filled > bar_width) filled = bar_width;

    /* ANSI Colors: Green < 60%, Yellow 60-85%, Red > 85% */
    const char* color = "\033[32m"; /* green */
    if (pct >= 85.0f) color = "\033[31m"; /* red */
    else if (pct >= 60.0f) color = "\033[33m"; /* yellow */

    printf("\033[90m[Context: \033[0m%s%d\033[0m/\033[90m%d tok (%.1f%%) \033[0m",
           color, cw->current_pos, cw->max_seq_len, pct);
    
    printf("\033[90m[\033[0m%s", color);
    for (int i = 0; i < bar_width; i++) {
        if (i < filled) putchar('#');
        else putchar('-');
    }
    printf("\033[0m\033[90m] Turns:%d Free:%d KV:%.1fMB]\033[0m\n",
           cw->user_turns_count, free_tok, kv_mb);
    fflush(stdout);
}

int context_window_slide(ContextWindow* cw, int reserve_tokens) {
    if (cw == NULL || cw->max_seq_len <= 0) return 0;
    if (cw->current_pos + reserve_tokens <= cw->max_seq_len) {
        return 0; /* No sliding needed */
    }

    /* We need to slide window to make space.
     * Keep system tokens at the front, drop older user/assistant turns */
    int keep_sys = cw->system_tokens_count;
    int target_keep = (cw->max_seq_len / 2); /* keep newest half */
    if (target_keep <= keep_sys) target_keep = keep_sys + 100;

    int drop_count = cw->history_length - target_keep;
    if (drop_count <= 0) return 0;

    /* Reconstruct history: [System tokens] + [Newer tokens] */
    int new_hist_len = keep_sys + (cw->history_length - (keep_sys + drop_count));
    if (new_hist_len <= 0) return 0;

    int* temp = (int*)malloc(sizeof(int) * (size_t)new_hist_len);
    if (temp == NULL) return -1;

    for (int i = 0; i < keep_sys; i++) {
        temp[i] = cw->token_history[i];
    }
    int src_start = keep_sys + drop_count;
    for (int i = keep_sys; i < new_hist_len; i++) {
        temp[i] = cw->token_history[src_start + (i - keep_sys)];
    }

    memcpy(cw->token_history, temp, sizeof(int) * (size_t)new_hist_len);
    free(temp);

    cw->history_length = new_hist_len;
    cw->current_pos = new_hist_len;
    if (cw->max_seq_len > 0) {
        cw->kv_bytes_used = (size_t)(((double)cw->current_pos / (double)cw->max_seq_len) * (double)cw->kv_bytes_total);
    }
    return drop_count;
}
