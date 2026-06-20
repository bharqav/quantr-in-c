#include "vm_terminal.h"
#include "sampling.h"
#include "benchmark.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

static double get_time_sec(void) {
#ifdef _WIN32
    static LARGE_INTEGER freq;
    static int initialized = 0;
    if (!initialized) {
        QueryPerformanceFrequency(&freq);
        initialized = 1;
    }
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return (double)now.QuadPart / (double)freq.QuadPart;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
#endif
}

static void print_banner(const VMTerminalConfig* cfg) {
    const Transformer* t = cfg->transformer;
    const RuntimeContext* rt = cfg->runtime;
    
    printf("\033[1;36m");
    printf("╔══════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  ⚡ QUANTR BAREMETAL VIRTUAL MACHINE TERMINAL v1.0                           ║\n");
    printf("║  Zero-Dependency Quantized Contextual AI Engine                             ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════════════╝\n");
    printf("\033[0m");

    printf("\033[90m┌─ \033[1;37mVM Environment Details\033[0m\033[90m ──────────────────────────────────────────────────┐\033[0m\n");
    printf("\033[90m│\033[0m  \033[1;33mModel Path:\033[0m      %-58s\033[90m│\033[0m\n", cfg->model_path ? cfg->model_path : "embedded");
    printf("\033[90m│\033[0m  \033[1;33mArchitecture:\033[0m    Layers: %-3d | Dim: %-5d | Heads: %-2d (Q) / %-2d (KV)     \033[90m│\033[0m\n",
           t->config.n_layers, t->config.dim, t->config.n_heads, t->config.n_kv_heads);
    printf("\033[90m│\033[0m  \033[1;33mContext Window:\033[0m  %-5d tokens | Vocab: %-6d tokens                    \033[90m│\033[0m\n",
           t->config.seq_len, t->config.vocab_size);
    printf("\033[90m│\033[0m  \033[1;33mCompute Target:\033[0m  Backend: %-8s | Threads: %-2d | Offload: %-13s\033[90m│\033[0m\n",
           rt->options.backend == BACKEND_GPU ? "GPU (Offload)" :
           rt->options.backend == BACKEND_AVX512 ? "AVX-512" :
           rt->options.backend == BACKEND_AVX2 ? "AVX2" :
           rt->options.backend == BACKEND_NEON ? "ARM NEON" : "Scalar Ref",
           rt->options.num_threads,
           rt->options.use_gpu ? "GPU Active" : "Pure CPU");
    printf("\033[90m└──────────────────────────────────────────────────────────────────────────────┘\033[0m\n\n");
    
    printf("\033[32m[Ready]\033[0m Type your message and press \033[1mEnter\033[0m. Type \033[1;33m/help\033[0m for VM commands.\n\n");
    fflush(stdout);
}

static void print_help(void) {
    printf("\n\033[1;33m=== Quantr VM Terminal Commands ===\033[0m\n");
    printf("  \033[1;36m/help\033[0m               - Display this help message\n");
    printf("  \033[1;36m/context\033[0m            - Show context window token statistics and memory usage\n");
    printf("  \033[1;36m/clear\033[0m or \033[1;36m/reset\033[0m     - Flush KV cache and reset conversation history\n");
    printf("  \033[1;36m/params\033[0m             - Show or adjust runtime sampling hyperparameters\n");
    printf("  \033[1;36m/temp <val>\033[0m         - Set sampling temperature (e.g., /temp 0.7)\n");
    printf("  \033[1;36m/topk <val>\033[0m         - Set top-k sampling threshold (e.g., /topk 40)\n");
    printf("  \033[1;36m/topp <val>\033[0m         - Set top-p nucleus threshold (e.g., /topp 0.9)\n");
    printf("  \033[1;36m/vm\033[0m or \033[1;36m/sysinfo\033[0m     - Inspect CPU vector registers & DRAM bandwidth\n");
    printf("  \033[1;36m/exit\033[0m or \033[1;36m/quit\033[0m      - Gracefully terminate the Virtual Machine session\n\n");
    fflush(stdout);
}

int vm_terminal_run(VMTerminalConfig* cfg) {
    if (cfg == NULL || cfg->transformer == NULL || cfg->runtime == NULL) {
        return -1;
    }

#ifdef _WIN32
    // Enable ANSI escape sequence colors in Windows Console
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut != INVALID_HANDLE_VALUE) {
        DWORD dwMode = 0;
        if (GetConsoleMode(hOut, &dwMode)) {
            dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
            SetConsoleMode(hOut, dwMode);
        }
    }
#endif

    Transformer* t = cfg->transformer;
    RuntimeContext* rt = cfg->runtime;
    Tokenizer* tok = cfg->tokenizer;
    int max_steps = cfg->max_steps_per_turn > 0 ? cfg->max_steps_per_turn : 256;

    /* Initialize Context Window */
    ContextWindow cw;
    size_t kv_total_bytes = (size_t)t->config.n_layers * (size_t)t->config.seq_len * (size_t)t->config.dim * sizeof(float) * 2;
    if (context_window_init(&cw, t->config.seq_len, kv_total_bytes) != 0) {
        fprintf(stderr, "[ERROR] Failed to initialize context window manager.\n");
        return -1;
    }

    SamplingState samp;
    sampling_state_init(&samp, 256);

    print_banner(cfg);

    char input_line[4096];
    int prompt_tokens[2048];

    while (1) {
        context_window_render_hud(&cw);
        printf("\033[1;32mUser>\033[0m ");
        fflush(stdout);

        if (fgets(input_line, sizeof(input_line), stdin) == NULL) {
            break; /* EOF or exit */
        }

        /* Trim trailing newlines and whitespace */
        size_t len = strlen(input_line);
        while (len > 0 && (input_line[len - 1] == '\n' || input_line[len - 1] == '\r' || input_line[len - 1] == ' ')) {
            input_line[--len] = '\0';
        }
        if (len == 0) continue;

        /* Handle VM Shell Commands */
        if (input_line[0] == '/') {
            if (strcmp(input_line, "/help") == 0) {
                print_help();
                continue;
            }
            if (strcmp(input_line, "/clear") == 0 || strcmp(input_line, "/reset") == 0) {
                context_window_reset(&cw, 0);
                free_run_state(&t->state);
                malloc_run_state(&t->state, &t->config);
                printf("\033[1;33m[VM] Context window and KV cache cleared.\033[0m\n\n");
                continue;
            }
            if (strcmp(input_line, "/context") == 0) {
                float pct = 0.0f;
                int free_tok = 0;
                float kv_mb = 0.0f;
                context_window_get_stats(&cw, &pct, &free_tok, &kv_mb);
                printf("\n\033[1;36m=== Context Window Diagnostic ===\033[0m\n");
                printf("  Total Capacity:      %d tokens\n", cw.max_seq_len);
                printf("  Active Position:     %d tokens\n", cw.current_pos);
                printf("  Remaining Headroom:  %d tokens\n", free_tok);
                printf("  Utilization:         %.2f%%\n", pct);
                printf("  Conversation Turns:  %d\n", cw.user_turns_count);
                printf("  KV Cache Allocation: %.2f MB\n\n", kv_mb);
                continue;
            }
            if (strcmp(input_line, "/vm") == 0 || strcmp(input_line, "/sysinfo") == 0) {
                print_sysinfo();
                continue;
            }
            if (strcmp(input_line, "/params") == 0) {
                printf("\n\033[1;36m=== Runtime Sampling Parameters ===\033[0m\n");
                printf("  Temperature:   %.2f (set via /temp <val>)\n", rt->options.temperature);
                printf("  Top-K:         %d   (set via /topk <val>)\n", rt->options.top_k);
                printf("  Top-P:         %.2f (set via /topp <val>)\n", rt->options.top_p);
                printf("  Rep. Penalty:  %.2f\n", rt->options.repetition_penalty);
                printf("  Max Steps:     %d tokens\n\n", max_steps);
                continue;
            }
            if (strncmp(input_line, "/temp ", 6) == 0) {
                float val = strtof(input_line + 6, NULL);
                if (val >= 0.0f && val <= 2.0f) {
                    rt->options.temperature = val;
                    printf("\033[1;33m[VM] Temperature set to %.2f\033[0m\n\n", val);
                } else {
                    printf("\033[31m[ERROR] Temperature must be between 0.0 and 2.0\033[0m\n\n");
                }
                continue;
            }
            if (strncmp(input_line, "/topk ", 6) == 0) {
                int val = atoi(input_line + 6);
                if (val >= 1 && val <= 500) {
                    rt->options.top_k = val;
                    printf("\033[1;33m[VM] Top-K set to %d\033[0m\n\n", val);
                } else {
                    printf("\033[31m[ERROR] Top-K must be between 1 and 500\033[0m\n\n");
                }
                continue;
            }
            if (strncmp(input_line, "/topp ", 6) == 0) {
                float val = strtof(input_line + 6, NULL);
                if (val > 0.0f && val <= 1.0f) {
                    rt->options.top_p = val;
                    printf("\033[1;33m[VM] Top-P set to %.2f\033[0m\n\n", val);
                } else {
                    printf("\033[31m[ERROR] Top-P must be between 0.01 and 1.0\033[0m\n\n");
                }
                continue;
            }
            if (strcmp(input_line, "/exit") == 0 || strcmp(input_line, "/quit") == 0) {
                printf("\n\033[1;33m[VM] Shutting down Virtual Machine session. Goodbye!\033[0m\n");
                break;
            }
            printf("\033[31m[Unknown Command: %s. Type /help for available commands.]\033[0m\n\n", input_line);
            continue;
        }

        /* Format Turn with ChatML or standard format */
        char formatted_prompt[4096];
        if (cw.user_turns_count == 0) {
            snprintf(formatted_prompt, sizeof(formatted_prompt),
                     "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", input_line);
        } else {
            snprintf(formatted_prompt, sizeof(formatted_prompt),
                     "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", input_line);
        }

        int n_tok = 0;
        if (tok != NULL && tok->vocab_size > 0) {
            n_tok = tokenizer_encode(tok, formatted_prompt, prompt_tokens, 2048);
        } else {
            prompt_tokens[0] = 1;
            n_tok = 1;
        }

        /* Check for context overflow and slide window if necessary */
        if (cw.current_pos + n_tok + max_steps >= cw.max_seq_len) {
            int dropped = context_window_slide(&cw, max_steps + n_tok);
            printf("\033[33m[Context Window Managed: Evicted %d oldest tokens to preserve conversational budget]\033[0m\n", dropped);
        }

        context_window_start_turn(&cw);

        printf("\033[1;34mQuantr>\033[0m ");
        fflush(stdout);

        double t0 = get_time_sec();
        double t_first = 0.0;
        int tokens_gen = 0;

        /* Prefill prompt tokens */
        int cur_token = 1;
        for (int i = 0; i < n_tok; i++) {
            cur_token = prompt_tokens[i];
            context_window_push(&cw, cur_token);
            sampling_state_push(&samp, cur_token);
            (void)forward_runtime(rt, t, cur_token, (cw.current_pos - 1) % t->config.seq_len);
        }

        t_first = get_time_sec();

        /* Autoregressive generation loop */
        for (int step = 0; step < max_steps; step++) {
            float* logits = forward_runtime(rt, t, cur_token, (cw.current_pos - 1) % t->config.seq_len);
            int next_tok = sample_next(logits, t->config.vocab_size, &rt->options, &samp);

            if (tok != NULL && tokenizer_is_eos(tok, next_tok)) {
                break;
            }

            context_window_push(&cw, next_tok);
            sampling_state_push(&samp, next_tok);
            cur_token = next_tok;
            tokens_gen++;

            if (tok != NULL) {
                char out_buf[256] = {0};
                tokenizer_decode_append(tok, next_tok, out_buf, (int)sizeof(out_buf));
                printf("%s", out_buf);
                fflush(stdout);
            } else {
                printf("%d ", next_tok);
                fflush(stdout);
            }

            if (cw.current_pos >= cw.max_seq_len - 1) {
                printf("\n\033[33m[Context Limit Reached]\033[0m");
                break;
            }
        }

        double t_end = get_time_sec();
        double total_dur = t_end - t0;
        double gen_dur = t_end - t_first;
        double tok_per_sec = gen_dur > 0.001 ? (double)tokens_gen / gen_dur : 0.0;

        printf("\n\033[90m[Generated %d tokens in %.2fs (%.1f tok/s) | Latency to first token: %.1fms]\033[0m\n\n",
               tokens_gen, total_dur, tok_per_sec, (t_first - t0) * 1000.0);
        fflush(stdout);
    }

    sampling_state_free(&samp);
    context_window_free(&cw);
    return 0;
}
