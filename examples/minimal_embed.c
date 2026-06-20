/**
 * @file minimal_embed.c
 * @brief Minimal standalone C example showing how to embed Baremetal AI into any C/C++ project.
 */

#include <stdio.h>
#include "baremetal.h"

static void on_token(const char* token_str, int token_id, void* user_data) {
    (void)token_id;
    (void)user_data;
    printf("%s", token_str);
    fflush(stdout);
}

int main(int argc, char** argv) {
    const char* model = argc > 1 ? argv[1] : "models/qwen3-30b-a3b-Q5_K_M.gguf";

    printf("Initializing Baremetal Engine on: %s\n", model);
    bm_config_t cfg = {
        .num_threads = 8,
        .backend = 1, /* AVX2 */
        .temperature = 0.7f,
        .top_k = 40,
        .top_p = 0.9f,
        .repetition_penalty = 1.05f
    };

    bm_engine_t* engine = bm_create(model, &cfg);
    if (!engine) {
        fprintf(stderr, "Failed to initialize engine.\n");
        return 1;
    }

    printf("\n--- Streaming Output ---\n");
    bm_generate(engine, "Tell me a 1-sentence thought about computer science.", 32, on_token, NULL);
    printf("\n------------------------\n");

    bm_destroy(engine);
    return 0;
}
