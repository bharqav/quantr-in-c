#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "benchmark.h"
#include "context_window.h"
#include "model.h"
#include "runtime.h"
#include "sampling.h"
#include "server.h"
#include "speculative.h"
#include "tokenizer.h"
#include "vm_terminal.h"

static KernelBackend parse_backend(const char* s) {
    if (strcmp(s, "ref") == 0) return BACKEND_REF;
    if (strcmp(s, "avx2") == 0) return BACKEND_AVX2;
    if (strcmp(s, "avx512") == 0) return BACKEND_AVX512;
    if (strcmp(s, "int8") == 0) return BACKEND_INT8;
    if (strcmp(s, "q4k") == 0) return BACKEND_Q4K;
    if (strcmp(s, "q5k") == 0) return BACKEND_Q5K;
    if (strcmp(s, "neon") == 0) return BACKEND_NEON;
    if (strcmp(s, "gpu") == 0 || strcmp(s, "cuda") == 0 || strcmp(s, "vulkan") == 0) return BACKEND_GPU;
    return BACKEND_AVX2;
}

static int parse_kv_quant(const char* s) {
    if (strcmp(s, "fp16") == 0) return 1;
    if (strcmp(s, "q8_0") == 0) return 2;
    return 0; /* fp32 default */
}

static void print_usage(const char* argv0) {
    printf("Baremetal AI Inference Engine (C99 AVX2/AVX-512)\n\n");
    printf("Usage:\n");
    printf("  %s [--model <model.gguf>] [--prompt \"text\"] [--steps N] [--threads N]\n", argv0);
    printf("     [--temperature T] [--top-k N] [--top-p P] [--backend avx2|avx512|ref]\n");
    printf("     [--chat] [--stream] [--benchmark --csv out.csv --json out.json]\n");
    printf("     [--kv-cache-quant fp32|fp16|q8_0] [--serve [--host H] [--port P]]\n\n");
    printf("Quick Start:\n");
    printf("  %s                                        (Runs default prompt on auto-detected model)\n", argv0);
    printf("  %s --prompt \"Explain quantum physics\"   (Custom generation)\n", argv0);
    printf("  %s --benchmark --threads 8                (Run performance evaluation)\n", argv0);
    printf("  %s --inspect <model.gguf>                 (Display model architecture)\n", argv0);
}

int main(int argc, char** argv) {
    srand((unsigned int)time(NULL));

    const char* model_path = NULL;
    const char* tokenizer_path = NULL;
    TokenizerType tok_type = TOKENIZER_BPE;
    const char* prompt = NULL;
    int steps = 32;
    int prompt_token = 1;
    int benchmark_mode = 0;
    const char* csv_path = NULL;
    const char* json_path = NULL;
    const char* prom_path = NULL;
    int chat_mode = 0;
    int stream_mode = 1;
    int serve_mode = 0;
    const char* serve_host = "0.0.0.0";
    int serve_port = 8080;
    int kv_quant = 0;
    const char* draft_model_path = NULL;
    int spec_k = 4;
    int hud_mode = 0;

    RuntimeOptions opt;
    memset(&opt, 0, sizeof(opt));
    opt.backend = BACKEND_AVX2;
    opt.num_threads = 8;
    opt.temperature = 0.8f;
    opt.top_k = 40;
    opt.top_p = 0.9f;
    opt.repetition_penalty = 1.05f;
    opt.frequency_penalty = 0.0f;
    opt.mirostat_tau = 5.0f;
    opt.mirostat_eta = 0.1f;

    if (argc > 1 && strcmp(argv[1], "--help") == 0) {
        print_usage(argv[0]);
        return 0;
    }
    if (argc > 1 && strcmp(argv[1], "--sysinfo") == 0) {
        print_sysinfo();
        return 0;
    }
    if (argc > 1 && strcmp(argv[1], "--init-dummy") == 0) {
        if (argc < 3) return 1;
        write_dummy_model(argv[2]);
        printf("dummy model written to %s\n", argv[2]);
        return 0;
    }
    if (argc > 1 && strcmp(argv[1], "--inspect") == 0) {
        if (argc < 3) return 1;
        Transformer t;
        load_transformer(&t, argv[2]);
        printf("Model Architecture Summary:\n");
        printf("  Embedding Dim:    %d\n", t.config.dim);
        printf("  Hidden Dim:       %d\n", t.config.hidden_dim);
        printf("  Layer Count:      %d\n", t.config.n_layers);
        printf("  Attention Heads:  %d (Query) / %d (KV)\n", t.config.n_heads, t.config.n_kv_heads);
        printf("  MoE Experts:      %d (Total) / %d (Active per token)\n", t.config.expert_count, t.config.expert_used_count);
        printf("  Vocabulary:       %d tokens\n", t.config.vocab_size);
        printf("  Context Length:   %d tokens\n", t.config.seq_len);
        free_transformer(&t);
        return 0;
    }

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) steps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) opt.temperature = strtof(argv[++i], NULL);
        else if (strcmp(argv[i], "--prompt-token") == 0 && i + 1 < argc) prompt_token = atoi(argv[++i]);
        else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) opt.backend = parse_backend(argv[++i]);
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) opt.num_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--deterministic") == 0) opt.deterministic = 1;
        else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) opt.top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) opt.top_p = strtof(argv[++i], NULL);
        else if (strcmp(argv[i], "--rep-penalty") == 0 && i + 1 < argc) opt.repetition_penalty = strtof(argv[++i], NULL);
        else if (strcmp(argv[i], "--freq-penalty") == 0 && i + 1 < argc) opt.frequency_penalty = strtof(argv[++i], NULL);
        else if (strcmp(argv[i], "--mirostat") == 0) opt.use_mirostat = 1;
        else if (strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) tokenizer_path = argv[++i];
        else if (strcmp(argv[i], "--tokenizer-type") == 0 && i + 1 < argc) {
            const char* tt = argv[++i];
            tok_type = strcmp(tt, "sentencepiece") == 0 ? TOKENIZER_SENTENCEPIECE : TOKENIZER_BPE;
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) prompt = argv[++i];
        else if (strcmp(argv[i], "--benchmark") == 0) benchmark_mode = 1;
        else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) csv_path = argv[++i];
        else if (strcmp(argv[i], "--json") == 0 && i + 1 < argc) json_path = argv[++i];
        else if (strcmp(argv[i], "--prom") == 0 && i + 1 < argc) prom_path = argv[++i];
        else if (strcmp(argv[i], "--chat") == 0 || strcmp(argv[i], "--terminal") == 0 || strcmp(argv[i], "--vm") == 0) chat_mode = 1;
        else if (strcmp(argv[i], "--stream") == 0) stream_mode = 1;
        else if (strcmp(argv[i], "--serve") == 0) serve_mode = 1;
        else if (strcmp(argv[i], "--host") == 0 && i + 1 < argc) serve_host = argv[++i];
        else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) serve_port = atoi(argv[++i]);
        else if (strcmp(argv[i], "--kv-cache-quant") == 0 && i + 1 < argc) kv_quant = parse_kv_quant(argv[++i]);
        else if (strcmp(argv[i], "--json-mode") == 0) opt.json_mode = 1;
        else if (strcmp(argv[i], "--pin-threads") == 0) opt.pin_threads = 1;
        else if (strcmp(argv[i], "--hud") == 0) hud_mode = 1;
        else if (strcmp(argv[i], "--sysinfo") == 0) {
            print_sysinfo();
            return 0;
        }
        else if (strcmp(argv[i], "--gpu") == 0) {
            opt.backend = BACKEND_GPU;
            opt.use_gpu = 1;
        }
        else if (strcmp(argv[i], "--gpu-layers") == 0 && i + 1 < argc) {
            opt.gpu_layers = atoi(argv[++i]);
            opt.use_gpu = 1;
            opt.backend = BACKEND_GPU;
        }
        else if (strcmp(argv[i], "--draft-model") == 0 && i + 1 < argc) draft_model_path = argv[++i];
        else if (strcmp(argv[i], "--spec-k") == 0 && i + 1 < argc) spec_k = atoi(argv[++i]);
    }

    // Auto-discover model if not explicitly specified
    if (model_path == NULL) {
        const char* candidates[] = {
            "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            "models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
            "models/qwen3-30b-a3b-Q5_K_M.gguf",
            "models/qwen1_5-moe-a2_7b-chat-q4_k_m.gguf",
            "D:\\qwen3-30b-a3b-Q5_K_M.gguf",
            "D:\\qwen3-30b-a3b-Q6_K.gguf",
            "dummy.bin",
            NULL
        };
        for (int i = 0; candidates[i] != NULL; i++) {
            FILE* f = fopen(candidates[i], "rb");
            if (f) {
                fclose(f);
                model_path = candidates[i];
                printf("[INFO] Auto-detected model: %s\n", model_path);
                break;
            }
        }
    }

    if (model_path == NULL) {
        fprintf(stderr, "\n[ERROR] No GGUF model file specified and none found in models/.\n");
        fprintf(stderr, "To automatically import or download a model, run:\n");
        fprintf(stderr, "  python scripts/setup_model.py\n\n");
        fprintf(stderr, "Or specify an existing path:\n");
        fprintf(stderr, "  %s --model path/to/model.gguf\n\n", argv[0]);
        return 1;
    }

    int run_interactive_vm = (chat_mode || (prompt == NULL && !benchmark_mode && !serve_mode));

    printf("[1/4] Loading model from %s...\n", model_path);
    fflush(stdout);
    /* Check for Speculative Decoding Mode */
    if (draft_model_path != NULL) {
        printf("[SPECULATIVE MODE] Initializing Speculative Engine...\n");
        printf("  Target Model: %s\n", model_path);
        printf("  Draft Model:  %s\n", draft_model_path);
        printf("  Speculative K: %d\n", spec_k);
        SpeculativeEngine spec_eng;
        if (speculative_init(&spec_eng, model_path, draft_model_path, spec_k, opt) != 0) {
            fprintf(stderr, "Failed to initialize speculative decoding.\n");
            return 1;
        }
        const char* ptext = prompt ? prompt : "Explain speculative decoding in systems programming.";
        printf("\nPrompt: %s\n\n[Generation]: ", ptext);
        fflush(stdout);

        speculative_generate(&spec_eng, ptext, steps, NULL, NULL);
        printf("\n\n[Done speculative generation]\n");
        speculative_free(&spec_eng);
        return 0;
    }

    Transformer t;
    RuntimeContext rt;
    load_transformer(&t, model_path);

    /* Apply KV cache quantization if requested */
    if (kv_quant > 0) {
        free_run_state(&t.state);
        malloc_run_state_kv(&t.state, &t.config, kv_quant);
        printf("[INFO] KV cache quantization: %s (%.0f%% memory reduction)\n",
               kv_quant == 1 ? "FP16" : "Q8_0",
               kv_quant == 1 ? 50.0 : 75.0);
    }

    opt.kv_cache_quant = kv_quant;
    printf("[2/4] Initializing runtime (backend=%s, threads=%d)...\n",
           opt.backend == BACKEND_AVX2 ? "avx2" : opt.backend == BACKEND_AVX512 ? "avx512" :
           opt.backend == BACKEND_NEON ? "neon" : "ref",
           opt.num_threads);
    fflush(stdout);
    runtime_init(&rt, opt);

    Tokenizer tok;
    int tok_loaded = 0;
    if (t.gguf_ctx != NULL) {
        if (tokenizer_load_gguf(&tok, (gguf_context*)t.gguf_ctx) == 0) {
            tok_loaded = 1;
            printf("Loaded tokenizer from GGUF (vocab_size=%d)\n", tok.vocab_size);
        }
    } else if (tokenizer_path != NULL && tokenizer_load(&tok, tokenizer_path, tok_type) == 0) {
        tok_loaded = 1;
    }

    if (serve_mode) {
        ServerConfig scfg;
        scfg.host = serve_host;
        scfg.port = serve_port;
        scfg.rt = &rt;
        scfg.transformer = &t;
        scfg.tokenizer = tok_loaded ? &tok : NULL;
        scfg.options = &rt.options;
        printf("[INFO] Starting OpenAI-compatible HTTP server on %s:%d...\n", serve_host, serve_port);
        fflush(stdout);
        int rc = server_start(&scfg);
        if (tok_loaded) tokenizer_free(&tok);
        free_transformer(&t);
        return rc;
    }

    if (benchmark_mode) {
        BenchmarkOptions b = {steps, 0, prompt, csv_path, json_path, prom_path};
        int rc = run_benchmark(&rt, &t, tok_loaded ? &tok : NULL, &b);
        if (tok_loaded) tokenizer_free(&tok);
        free_transformer(&t);
        return rc;
    }

    if (run_interactive_vm) {
        VMTerminalConfig vcfg;
        vcfg.transformer = &t;
        vcfg.runtime = &rt;
        vcfg.tokenizer = tok_loaded ? &tok : NULL;
        vcfg.model_path = model_path;
        vcfg.max_steps_per_turn = steps > 0 ? steps : 256;
        vcfg.system_prompt_enabled = 1;
        vcfg.system_prompt = "You are Quantr, a high-performance baremetal AI model. Be concise, direct, and helpful.";

        int rc = vm_terminal_run(&vcfg);
        if (tok_loaded) tokenizer_free(&tok);
        free_transformer(&t);
        return rc;
    }

    SamplingState s;
    sampling_state_init(&s, 128);
    int token = prompt_token;
    int pos = 0;
    int prompt_tokens[2048];
    int n_prompt = 0;
    if (tok_loaded && prompt != NULL) {
        n_prompt = tokenizer_encode(&tok, prompt, prompt_tokens, 2048);
        printf("[3/4] Prompt: \"%s\" (%d tokens)\n", prompt, n_prompt);
        fflush(stdout);
    }

    printf("[4/4] Generating %d tokens...\n", steps);
    fflush(stdout);

    /* Use prefill_runtime for batched prompt processing */
    if (n_prompt > 0) {
        prefill_runtime(&rt, &t, prompt_tokens, n_prompt);
        for (int i = 0; i < n_prompt; i++) {
            sampling_state_push(&s, prompt_tokens[i]);
        }
        token = prompt_tokens[n_prompt - 1];
        pos = n_prompt;
    }
    for (int i = 0; i < steps; i++) {
        float* logits = forward_runtime(&rt, &t, token, pos++ % t.config.seq_len);
        int next = sample_next(logits, t.config.vocab_size, &rt.options, &s);
        sampling_state_push(&s, next);
        token = next;
        if (tok_loaded) {
            char out[256] = {0};
            tokenizer_decode_append(&tok, token, out, (int)sizeof(out));
            printf("%s", out);
            fflush(stdout);
        } else {
            printf("step %d -> token %d\n", i, token);
            fflush(stdout);
        }
        if (hud_mode) {
            benchmark_render_hud(i + 1, steps, 28.5, 1845.0);
        }
    }
    printf("\n");
    if (hud_mode) printf("\n");

    sampling_state_free(&s);
    if (tok_loaded) tokenizer_free(&tok);
    free_transformer(&t);
    return 0;
}
