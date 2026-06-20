/**
 * @file baremetal.h
 * @brief Embeddable C API for Baremetal AI Engine (IoT, Edge, Microservices, Python/Rust bindings).
 * 
 * Example Usage:
 * @code
 *   bm_engine_t* engine = bm_create("models/qwen3-30b-a3b-Q5_K_M.gguf", 8);
 *   bm_generate(engine, "Hello, world!", 32, my_token_callback, NULL);
 *   bm_destroy(engine);
 * @endcode
 */

#ifndef BAREMETAL_H
#define BAREMETAL_H

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
  #if defined(BAREMETAL_BUILD_DLL)
    #define BM_API __declspec(dllexport)
  #else
    #define BM_API
  #endif
#else
  #define BM_API __attribute__((visibility("default")))
#endif

typedef struct bm_engine bm_engine_t;

typedef void (*bm_token_callback)(const char* token_str, int token_id, void* user_data);

typedef struct {
    int num_threads;
    int backend;            /* 0=ref, 1=avx2, 2=avx512 */
    float temperature;
    int top_k;
    float top_p;
    float repetition_penalty;
} bm_config_t;

typedef struct {
    double ttft_ms;
    double mean_latency_ms;
    double tokens_per_sec;
    double peak_rss_mb;
    int total_tokens;
} bm_stats_t;

/**
 * @brief Initialize engine and map GGUF model into memory.
 */
BM_API bm_engine_t* bm_create(const char* model_path, const bm_config_t* config);

/**
 * @brief Stream prompt response token by token via callback.
 */
BM_API int bm_generate(bm_engine_t* engine,
                       const char* prompt,
                       int max_tokens,
                       bm_token_callback cb,
                       void* user_data);

/**
 * @brief Retrieve performance statistics from the last generation run.
 */
BM_API bm_stats_t bm_get_stats(const bm_engine_t* engine);

/**
 * @brief Free engine runtime buffers and unmap model memory.
 */
BM_API void bm_destroy(bm_engine_t* engine);

#ifdef __cplusplus
}
#endif

#endif /* BAREMETAL_H */
