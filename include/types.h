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
    BACKEND_NEON = 6,              /**< ARM NEON SIMD acceleration (128-bit) */
    BACKEND_GPU = 7                /**< GPU acceleration / offload backend */
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
    int use_gpu;                   /**< Enable GPU compute offloading (1=enabled, 0=CPU) */
    int gpu_layers;                /**< Number of transformer layers offloaded to GPU */
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
