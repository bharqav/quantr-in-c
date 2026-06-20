/**
 * @file baremetal.h
 * @brief Amalgamated single-header for Baremetal AI Inference Engine.
 * Zero external dependencies. C99 Standard.
 */

#ifndef BAREMETAL_AMALGAMATION_H
#define BAREMETAL_AMALGAMATION_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


/* ─── Header: include/types.h ─── */
/**
 * @file types.h
 * @brief Core data structures and configuration for Baremetal AI Inference Engine.
 */

#ifndef TYPES_H
#define TYPES_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Model architecture hyperparameters.
 */
typedef struct {
    int dim;                   /**< Model embedding dimension (e.g. 2048) */
    int hidden_dim;            /**< Intermediate FFN/expert dimension (e.g. 768 or 6144) */
    int n_layers;              /**< Total transformer layer count (e.g. 48) */
    int n_heads;               /**< Number of query attention heads (e.g. 32) */
    int n_kv_heads;            /**< Number of key/value heads for GQA (e.g. 4) */
    int vocab_size;            /**< Tokenizer vocabulary size (e.g. 151936) */
    int seq_len;               /**< Maximum allocated KV cache context length */
    float rope_freq_base;      /**< RoPE rotary base frequency (e.g. 1000000.0) */
    int expert_count;          /**< Total MoE sparse experts per layer (e.g. 128) */
    int expert_used_count;     /**< Top-K active experts selected per token (e.g. 6) */
} Config;

/**
 * @brief Memory-mapped weights and per-layer quantization types.
 */
typedef struct {
    float* token_embedding_table;  /**< Token embedding table (quantized or FP32) */
    uint32_t token_embd_type;      /**< GGUF type ID for token embeddings */
    size_t token_embd_row_bytes;   /**< Byte stride per token row */

    float** rms_att_weight;        /**< Layer attention RMSNorm weights [n_layers][dim] */
    float** rms_ffn_weight;        /**< Layer feed-forward RMSNorm weights [n_layers][dim] */
    float** wq;                    /**< Attention Query projection weights */
    float** wk;                    /**< Attention Key projection weights */
    float** wv;                    /**< Attention Value projection weights */
    float** wo;                    /**< Attention Output projection weights */
    float** w1;                    /**< Dense FFN Gate weights (if non-MoE) */
    float** w2;                    /**< Dense FFN Down weights (if non-MoE) */
    float** w3;                    /**< Dense FFN Up weights (if non-MoE) */
    
    float* rms_final_weight;       /**< Final RMSNorm weight [dim] */
    float* wcls;                   /**< Output classifier / lm_head weights */
    uint32_t wcls_type;            /**< GGUF type ID for output classifier */

    /* Per-layer dynamic quantization types (supports mixed precision Q5_K_M) */
    uint32_t* wq_type;             /**< Per-layer type for Wq */
    uint32_t* wk_type;             /**< Per-layer type for Wk */
    uint32_t* wv_type;             /**< Per-layer type for Wv */
    uint32_t* wo_type;             /**< Per-layer type for Wo */
    uint32_t* exp_w1_type;         /**< Per-layer type for Expert Up */
    uint32_t* exp_w2_type;         /**< Per-layer type for Expert Down */
    uint32_t* exp_w3_type;         /**< Per-layer type for Expert Gate */
    uint32_t* w1_type;             /**< Per-layer type for Dense W1 */
    uint32_t* w2_type;             /**< Per-layer type for Dense W2 */
    uint32_t* w3_type;             /**< Per-layer type for Dense W3 */
    
    /* Sparse Mixture-of-Experts (MoE) pointers */
    float** ffn_gate_inp;          /**< Router gating weights [n_layers][expert_count * dim] */
    float*** expert_w1;            /**< Expert Up weights [n_layers][expert_count] */
    float*** expert_w2;            /**< Expert Down weights [n_layers][expert_count] */
    float*** expert_w3;            /**< Expert Gate weights [n_layers][expert_count] */
} TransformerWeights;

/**
 * @brief Zero-allocation contiguous runtime state buffer.
 */
typedef struct {
    float* x;                      /**< Current token activation vector [dim] */
    float* xb;                     /**< Normalized activation vector [dim] */
    float* xb2;                    /**< Secondary workspace buffer [dim] */
    float* hb;                     /**< Intermediate FFN/expert activation [hidden_dim] */
    float* hb2;                    /**< Secondary FFN/expert activation [hidden_dim] */
    float* q;                      /**< Query vector [n_heads * head_dim] */
    float* k;                      /**< Key vector [n_kv_heads * head_dim] */
    float* v;                      /**< Value vector [n_kv_heads * head_dim] */
    float* att;                    /**< Attention scores matrix [n_heads * seq_len] */
    float* logits;                 /**< Output vocabulary logits [vocab_size] */
    void* key_cache;               /**< KV Key cache (FP32, FP16, or Q8_0) */
    void* value_cache;             /**< KV Value cache (FP32, FP16, or Q8_0) */
    int kv_cache_type;             /**< 0=FP32, 1=FP16, 2=Q8_0 */
    float* memory;                 /**< Base pointer to contiguous allocation */
    size_t memory_floats;          /**< Total allocated floats in memory */
    void* kv_memory;               /**< Separate allocation for KV cache (if quantized) */
    size_t kv_memory_bytes;        /**< Total KV cache allocation in bytes */
} RunState;

/**
 * @brief Execution kernel acceleration backends.
 */
typedef enum {
    BACKEND_REF = 0,               /**< Pure scalar C reference backend */
    BACKEND_AVX2 = 1,              /**< AVX2 + FMA SIMD acceleration (256-bit) */
    BACKEND_AVX512 = 2,            /**< AVX-512 vector acceleration (512-bit) */
    BACKEND_INT8 = 3,              /**< Int8 quantized kernel backend */
    BACKEND_Q4K = 4,               /**< Block Q4_K specialized backend */
    BACKEND_Q5K = 5,               /**< Block Q5_K specialized backend */
    BACKEND_NEON = 6               /**< ARM NEON SIMD acceleration (128-bit) */
} KernelBackend;

/**
 * @brief User-configurable runtime and generation options.
 */
typedef struct {
    KernelBackend backend;         /**< Selected SIMD backend */
    int num_threads;               /**< OpenMP worker thread count */
    int deterministic;            /**< Force greedy argmax sampling (temperature=0) */
    float temperature;             /**< Sampling temperature (0.0 to 2.0) */
    int top_k;                     /**< Top-K truncation filter */
    float top_p;                   /**< Top-P (Nucleus) cumulative threshold */
    float repetition_penalty;      /**< Penalty applied to previously generated tokens */
    float frequency_penalty;       /**< Penalty proportional to token occurrence */
    int use_mirostat;              /**< Enable Mirostat v2 adaptive entropy sampling */
    float mirostat_tau;            /**< Mirostat target entropy */
    float mirostat_eta;            /**< Mirostat learning rate */
    int kv_cache_quant;            /**< KV cache quantization: 0=FP32, 1=FP16, 2=Q8_0 */
    int json_mode;                 /**< Enable grammar-constrained JSON mode decoding */
    int pin_threads;               /**< Enable hardware thread affinity / CPU core pinning */
} RuntimeOptions;

/**
 * @brief Top-level transformer model instance.
 */
typedef struct {
    Config config;                 /**< Architectural configuration */
    TransformerWeights weights;    /**< Sliced weight pointers */
    RunState state;                /**< Contiguous execution buffers */
    void* mapped_data;             /**< Base address of memory-mapped model file */
    size_t mapped_size;            /**< Total mapped file byte size */
#ifdef _WIN32
    void* map_handle;              /**< Win32 file mapping handle */
#endif
    int mapped_fd;                 /**< File descriptor / handle */
    void* gguf_ctx;                /**< Opaque pointer to GGUF parser context */
} Transformer;

#ifdef __cplusplus
}
#endif

#endif /* TYPES_H */

/* ─── Header: include/gguf.h ─── */
#ifndef GGUF_H
#define GGUF_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define GGUF_MAGIC 0x46554747 // "GGUF"
#define GGUF_VERSION 3

enum gguf_type {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

// These match llama.cpp GGML types
enum ggml_type {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    // ... we care mostly about Q4_K and Q8_0
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_I8   = 16,
    GGML_TYPE_I16  = 17,
    GGML_TYPE_I32  = 18,
    GGML_TYPE_I64  = 19,
};

typedef struct {
    char* name;
    uint32_t type;
    uint32_t n_dims;
    uint64_t ne[4]; // dimensions
    uint64_t offset;
    void* data; // pointer into memory mapped file
} gguf_tensor;

typedef struct {
    char* name;
    uint32_t type;
    void* value; // raw value pointer
    uint64_t len; // for strings and arrays
} gguf_kv;

typedef struct {
    void* data;
    size_t size;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t kv_count;
    gguf_kv* kv;
    gguf_tensor* tensors;
    uint64_t alignment;
    uint64_t data_offset;
} gguf_context;

// Parse a GGUF file from a memory mapped buffer
gguf_context* gguf_init_from_buffer(void* buffer, size_t size);
void gguf_free(gguf_context* ctx);

// Lookups
gguf_tensor* gguf_find_tensor(gguf_context* ctx, const char* name);
uint32_t gguf_get_val_u32(gguf_context* ctx, const char* key, uint32_t def);
float gguf_get_val_f32(gguf_context* ctx, const char* key, float def);
const char* gguf_get_val_str(gguf_context* ctx, const char* key);
int gguf_get_val_str_array(gguf_context* ctx, const char* key, char*** arr_out, uint64_t* len_out);

#endif // GGUF_H

/* ─── Header: include/quant.h ─── */
#ifndef QUANT_H
#define QUANT_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>

typedef struct {
    int group_size;
    int rows;
    int cols;
    int8_t* data;
    float* scales;
    float* zeros;
} QuantInt8Matrix;

typedef struct {
    int rows;
    int cols;
    int group_size;
    int bits;
    uint8_t* data;
    float* scales;
    float* mins;
} QuantKMatrix;

#define QK8_0 32
#pragma pack(push, 1)
typedef struct {
    uint16_t d;          // fp16 delta
    int8_t   qs[QK8_0];  // quants
} block_q8_0;

#define QK_K 256
#define K_SCALE_SIZE 12
typedef struct {
    uint16_t d;             // fp16 delta
    uint16_t dmin;          // fp16 min
    uint8_t  scales[K_SCALE_SIZE];
    uint8_t  qs[QK_K/2];
} block_q4_K;

typedef struct {
    uint16_t d;             // fp16 delta
    uint16_t dmin;          // fp16 min
    uint8_t  scales[K_SCALE_SIZE];
    uint8_t  qh[QK_K/8];
    uint8_t  qs[QK_K/2];
} block_q5_K;
#pragma pack(pop)

static inline float fp16_to_fp32(uint16_t h) {
    uint32_t w = (uint32_t)(h & 0x7FFF) << 13;
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    if (exp == 0x1F) {
        w |= 0x7F800000;
    } else if (exp != 0) {
        w += 0x38000000;
    }
    w |= sign;
    float f;
    memcpy(&f, &w, 4);
    return f;
}

static inline uint16_t fp32_to_fp16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint16_t sign = (x >> 16) & 0x8000;
    int exp = ((x >> 23) & 0xFF) - 127 + 15;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7C00;
    uint16_t mant = (x >> 13) & 0x03FF;
    return sign | (exp << 10) | mant;
}

int quantize_int8_groupwise(QuantInt8Matrix* q, const float* w, int rows, int cols, int group_size);
int quantize_qk_groupwise(QuantKMatrix* q, const float* w, int rows, int cols, int group_size, int bits);
void free_quant_int8(QuantInt8Matrix* q);
void free_quant_k(QuantKMatrix* q);

void quantize_row_q8_0(const float* x, block_q8_0* y, int k);
void dequantize_row(float* out, const void* src, uint32_t type, int n);
float vec_dot_q4K_q8_0(const block_q4_K* a, const block_q8_0* b, int k);
float vec_dot_q5K_q8_0(const block_q5_K* a, const block_q8_0* b, int k);
float vec_dot_q6K_q8_0(const void* a, const block_q8_0* b, int k);
float vec_dot_q8_0_q8_0(const block_q8_0* a, const block_q8_0* b, int k);

void matmul_q_fused(float* out, const float* x, const void* w, uint32_t type,
                    int n, int d, int threads);

void matmul_int8_groupwise(float* out, const float* x, const QuantInt8Matrix* q);
void matmul_qk_groupwise(float* out, const float* x, const QuantKMatrix* q);

#endif

/* ─── Header: include/kernels.h ─── */
#ifndef KERNELS_H
#define KERNELS_H


typedef struct {
    void (*matmul)(float* out, const float* x, const float* w, int n, int d, int threads);
    float (*dot)(const float* a, const float* b, int n);
    void (*rmsnorm)(float* o, const float* x, const float* weight, int size);
    void (*softmax)(float* x, int size);
    const char* name;
} KernelOps;

const KernelOps* kernels_for_backend(KernelBackend backend);
int cpu_supports_avx2(void);
int cpu_supports_avx512(void);
int cpu_supports_neon(void);
KernelBackend resolve_backend(KernelBackend requested);

#endif

/* ─── Header: include/model.h ─── */
#ifndef MODEL_H
#define MODEL_H


size_t checked_mul(size_t a, size_t b, const char* label);
size_t config_weight_floats(const Config* p);
size_t run_state_floats(const Config* p);

void malloc_run_state(RunState* s, const Config* p);
void malloc_run_state_kv(RunState* s, const Config* p, int kv_type);
void free_run_state(RunState* s);

void load_transformer(Transformer* t, const char* model_path);
void free_transformer(Transformer* t);
void write_dummy_model(const char* path);

#endif

/* ─── Header: include/threadpool.h ─── */
#ifndef THREADPOOL_H
#define THREADPOOL_H

typedef struct {
    int threads;
    int deterministic;
} ThreadPool;

void threadpool_init(ThreadPool* p, int threads, int deterministic);
void threadpool_destroy(ThreadPool* p);
void threadpool_bind_affinity(ThreadPool* p);
void threadpool_parallel_for(ThreadPool* p, int start, int end, void (*fn)(int, void*), void* ctx);

#endif

/* ─── Header: include/sampling.h ─── */
#ifndef SAMPLING_H
#define SAMPLING_H


typedef struct {
    float* recent_tokens;
    int recent_capacity;
    int recent_count;
} SamplingState;

typedef struct {
    int depth;            /**< Object {} nesting depth */
    int array_depth;      /**< Array [] nesting depth */
    int in_string;        /**< Inside string literal */
    int escape;           /**< Character escaping active */
    int token_count;      /**< Tokens emitted in JSON generation */
    int complete;         /**< Whether top-level JSON is fully closed */
} JsonGrammarState;

void json_grammar_init(JsonGrammarState* g);
void json_grammar_update(JsonGrammarState* g, const char* token_piece);
void json_grammar_filter_logits(float* logits, int vocab_size, const JsonGrammarState* g);

void sampling_state_init(SamplingState* s, int capacity);
void sampling_state_free(SamplingState* s);
void sampling_state_push(SamplingState* s, int token);

int sample_next(float* logits, int vocab_size, const RuntimeOptions* opt, SamplingState* state);

#endif

/* ─── Header: include/tokenizer.h ─── */
#ifndef TOKENIZER_H
#define TOKENIZER_H


typedef enum {
    TOKENIZER_NONE = 0,
    TOKENIZER_BPE = 1,
    TOKENIZER_SENTENCEPIECE = 2
} TokenizerType;

typedef struct {
    TokenizerType type;
    int vocab_size;
    char** vocab;
    int merges_size;
    char** merges;
    int bucket_head[256];
    int* bucket_next;
} Tokenizer;

int tokenizer_load(Tokenizer* t, const char* path, TokenizerType type_hint);
int tokenizer_load_gguf(Tokenizer* t, gguf_context* ctx);
void tokenizer_free(Tokenizer* t);
int tokenizer_encode(const Tokenizer* t, const char* text, int* out_tokens, int max_tokens);
void tokenizer_decode_append(const Tokenizer* t, int token, char* out_text, int out_cap);
int tokenizer_is_eos(const Tokenizer* t, int token);

#endif

/* ─── Header: include/runtime.h ─── */
#ifndef RUNTIME_H
#define RUNTIME_H


typedef struct {
    RuntimeOptions options;
    const KernelOps* kernels;
} RuntimeContext;

void runtime_init(RuntimeContext* rt, RuntimeOptions options);
float* forward_runtime(RuntimeContext* rt, Transformer* t, int token, int pos);
void prefill_runtime(RuntimeContext* rt, Transformer* t, const int* tokens, int n_tokens);

#endif

/* ─── Header: include/benchmark.h ─── */
/**
 * @file benchmark.h
 * @brief High-precision performance profiler, memory governor, and latency metrics.
 */

#ifndef BENCHMARK_H
#define BENCHMARK_H


#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int steps;
    int warmup;
    const char* prompt;
    const char* csv_path;
    const char* json_path;
    const char* prom_path;     /**< Prometheus metrics export path */
} BenchmarkOptions;

typedef struct {
    double ttft_ms;            /**< Time to First Token (ms) */
    double mean_latency_ms;    /**< Mean inter-token latency (ms) */
    double p50_latency_ms;     /**< 50th percentile (median) latency (ms) */
    double p95_latency_ms;     /**< 95th percentile latency (ms) */
    double p99_latency_ms;     /**< 99th percentile latency (ms) */
    double tokens_per_sec;     /**< Generation throughput (tok/s) */
    double forward_time_sec;   /**< Total tensor forward pass time */
    double sample_time_sec;    /**< Total probability sampling time */
    double elapsed_sec;        /**< Total wall-clock time */
    double peak_rss_mb;        /**< Peak physical resident RAM (MB) */
    double virtual_mem_mb;     /**< Virtual address space allocation (MB) */
    double effective_bw_gbps;  /**< Effective memory bandwidth (GB/s) */
    double expert_sparsity_pct;/**< MoE parameter sparsity percentage */
    double active_params_b;    /**< Active parameter count per token (Billions) */
    double total_params_b;     /**< Total model parameter count (Billions) */
    double compression_ratio;  /**< Model disk compression ratio vs FP32 */
    int total_tokens;          /**< Total generated tokens */
} BenchmarkMetrics;

int run_benchmark(RuntimeContext* rt,
                  Transformer* t,
                  const Tokenizer* tok,
                  const BenchmarkOptions* bopt);

void print_metrics_dashboard(const BenchmarkMetrics* m, const char* model_name, const char* backend_name, int threads);
void print_sysinfo(void);
void benchmark_render_hud(int token_idx, int total_tokens, double tok_per_sec, double rss_mb);

#ifdef __cplusplus
}
#endif

#endif /* BENCHMARK_H */

/* ─── Header: include/server.h ─── */
/**
 * @file server.h
 * @brief Embedded OpenAI-compatible HTTP server for Baremetal AI Engine.
 *
 * Exposes POST /v1/chat/completions with SSE streaming and GET /v1/models.
 */

#ifndef SERVER_H
#define SERVER_H


#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char* host;              /**< Bind address (e.g. "0.0.0.0") */
    int port;                      /**< Listen port (e.g. 8080) */
    RuntimeContext* rt;            /**< Initialized runtime context */
    Transformer* transformer;      /**< Loaded transformer model */
    Tokenizer* tokenizer;          /**< Loaded tokenizer (may be NULL) */
    RuntimeOptions* options;       /**< Runtime options for sampling */
} ServerConfig;

/**
 * @brief Start the HTTP server (blocking). Returns 0 on clean shutdown, -1 on error.
 */
int server_start(const ServerConfig* config);

#ifdef __cplusplus
}
#endif

#endif /* SERVER_H */

/* ─── Header: include/speculative.h ─── */
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

#ifdef __cplusplus
}
#endif

#endif /* SPECULATIVE_H */

/* ─── Header: include/baremetal.h ─── */
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

#endif /* BAREMETAL_AMALGAMATION_H */
