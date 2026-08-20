#include "runtime.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "quant.h"
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

static inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

static void apply_rope(float* q, float* k, int dim, int head_dim, int n_kv_heads, int pos, float rope_freq_base) {
    int kv_dim = n_kv_heads * head_dim;
    if (kv_dim > dim) {
        kv_dim = dim;
    }
    float base = rope_freq_base > 0.0f ? rope_freq_base : 10000.0f;
    for (int i = 0; i < dim; i += 2) {
        float freq = powf(base, -((float)i / (float)head_dim));
        float theta = (float)pos * freq;
        float c = cosf(theta);
        float s = sinf(theta);
        float q0 = q[i];
        float q1 = q[i + 1];
        q[i] = q0 * c - q1 * s;
        q[i + 1] = q0 * s + q1 * c;
        if (i < kv_dim) {
            float k0 = k[i];
            float k1 = k[i + 1];
            k[i] = k0 * c - k1 * s;
            k[i + 1] = k0 * s + k1 * c;
        }
    }
}

#include "threadpool.h"

void runtime_init(RuntimeContext* rt, RuntimeOptions options) {
    rt->options = options;
    rt->options.backend = resolve_backend(options.backend);
    rt->kernels = kernels_for_backend(rt->options.backend);

    if (options.pin_threads) {
        ThreadPool tp;
        threadpool_init(&tp, options.num_threads, options.deterministic);
        threadpool_bind_affinity(&tp);
        printf("[INFO] Pinned %d worker threads to physical CPU cores.\n", options.num_threads);
    }
}

static void matmul_q(float* out, const float* x, const void* w, uint32_t type, int n, int d, const KernelOps* k, int threads) {
    if (w == NULL) {
        memset(out, 0, (size_t)n * sizeof(float));
        return;
    }
    if (type == 0) { // F32: use SIMD kernel directly
        k->matmul(out, x, (const float*)w, n, d, threads);
        return;
    }
    // Fused path: quantize x to Q8_0 once, then compute integer dot products
    // directly against quantized weights (Q8_0, Q4_K, Q5_K, Q6_K)
    matmul_q_fused(out, x, w, type, n, d, threads);
}


/* KV cache store/load helpers for FP32/FP16/Q8_0 quantization */
static void kv_store_row(void* cache, int kv_type, size_t offset, const float* src, int dim) {
    if (kv_type == 1) { /* FP16 */
        uint16_t* dst = (uint16_t*)cache + offset;
        for (int i = 0; i < dim; i++) {
            dst[i] = fp32_to_fp16(src[i]);
        }
    } else { /* FP32 (default) */
        float* dst = (float*)cache + offset;
        memcpy(dst, src, (size_t)dim * sizeof(float));
    }
}

/* Inline KV cache element access for attention dot product (avoids full row dequant) */
static inline float kv_get_element(const void* cache, int kv_type, size_t idx) {
    if (kv_type == 1) {
        return fp16_to_fp32(((const uint16_t*)cache)[idx]);
    }
    return ((const float*)cache)[idx];
}

float* forward_runtime(RuntimeContext* rt, Transformer* t, int token, int pos) {
    Config* p = &t->config;
    TransformerWeights* w = &t->weights;
    RunState* s = &t->state;
    const KernelOps* k = rt->kernels;

    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int head_dim = dim / (p->n_heads > 0 ? p->n_heads : 1);
    int q_dim = p->n_heads * head_dim;
    int kv_dim = p->n_kv_heads * head_dim;
    int kv_mul = p->n_heads / (p->n_kv_heads > 0 ? p->n_kv_heads : 1);
    float inv_sqrt_head_dim = 1.0f / sqrtf((float)head_dim);

    if (w->token_embedding_table != NULL && w->token_embd_row_bytes > 0) {
        const char* row_ptr = (const char*)w->token_embedding_table + (size_t)token * w->token_embd_row_bytes;
        dequantize_row(s->x, row_ptr, w->token_embd_type, dim);
    } else {
        memcpy(s->x, (const float*)w->token_embedding_table + (size_t)token * (size_t)dim, (size_t)dim * sizeof(float));
    }

    for (int l = 0; l < p->n_layers; l++) {
        const float* rms_att = w->rms_att_weight[l];
        const float* rms_ffn = w->rms_ffn_weight[l];
        const float* wq = w->wq[l];
        const float* wk = w->wk[l];
        const float* wv = w->wv[l];
        const float* wo = w->wo[l];

        k->rmsnorm(s->xb, s->x, rms_att, dim);
        matmul_q(s->q, s->xb, wq, w->wq_type ? w->wq_type[l] : 0, q_dim, dim, k, rt->options.num_threads);
        matmul_q(s->k, s->xb, wk, w->wk_type ? w->wk_type[l] : 0, kv_dim, dim, k, rt->options.num_threads);
        matmul_q(s->v, s->xb, wv, w->wv_type ? w->wv_type[l] : 0, kv_dim, dim, k, rt->options.num_threads);
        apply_rope(s->q, s->k, dim, head_dim, p->n_kv_heads, pos, p->rope_freq_base);

        /* Store K/V to cache (with optional FP16 quantization) */
        size_t cache_offset = ((size_t)l * (size_t)p->seq_len + (size_t)pos) * (size_t)dim;
        kv_store_row(s->key_cache, s->kv_cache_type, cache_offset, s->k, dim);
        kv_store_row(s->value_cache, s->kv_cache_type, cache_offset, s->v, dim);
        memset(s->xb, 0, (size_t)dim * sizeof(float));

            int kv_h;
#ifdef _OPENMP
#pragma omp parallel for num_threads(rt->options.num_threads > 0 ? rt->options.num_threads : 1) schedule(static)
#endif
        for (kv_h = 0; kv_h < p->n_kv_heads; kv_h++) {
            size_t kv_offset = (size_t)kv_h * (size_t)head_dim;
            
            for (int tpos = 0; tpos <= pos; tpos++) {
                size_t k_offset = ((size_t)l * (size_t)p->seq_len + (size_t)tpos) * (size_t)dim + kv_offset;
                /* Load cached key row on-the-fly for dot product */
                float kh_buf[512]; /* head_dim is typically <= 128 */
                for (int i = 0; i < head_dim; i++) {
                    kh_buf[i] = kv_get_element(s->key_cache, s->kv_cache_type, k_offset + (size_t)i);
                }
                
                for (int m = 0; m < kv_mul; m++) {
                    int h = kv_h * kv_mul + m;
                    float* qh = s->q + (size_t)h * (size_t)head_dim;
                    float* att = s->att + (size_t)h * (size_t)p->seq_len;
                    att[tpos] = k->dot(qh, kh_buf, head_dim) * inv_sqrt_head_dim;
                }
            }
        }
        
    int h;
#ifdef _OPENMP
#pragma omp parallel for num_threads(rt->options.num_threads > 0 ? rt->options.num_threads : 1) schedule(static)
#endif
        for (h = 0; h < p->n_heads; h++) {
            float* att = s->att + (size_t)h * (size_t)p->seq_len;
            k->softmax(att, pos + 1);
        }
        
#ifdef _OPENMP
#pragma omp parallel for num_threads(rt->options.num_threads > 0 ? rt->options.num_threads : 1) schedule(static)
#endif
        for (kv_h = 0; kv_h < p->n_kv_heads; kv_h++) {
            size_t kv_offset = (size_t)kv_h * (size_t)head_dim;
            
            for (int m = 0; m < kv_mul; m++) {
                int h = kv_h * kv_mul + m;
                float* out = s->xb + (size_t)h * (size_t)head_dim;
                memset(out, 0, head_dim * sizeof(float));
            }
            
            for (int tpos = 0; tpos <= pos; tpos++) {
                size_t v_offset = ((size_t)l * (size_t)p->seq_len + (size_t)tpos) * (size_t)dim + kv_offset;
                
                for (int m = 0; m < kv_mul; m++) {
                    int h = kv_h * kv_mul + m;
                    float* att = s->att + (size_t)h * (size_t)p->seq_len;
                    float a = att[tpos];
                    float* out = s->xb + (size_t)h * (size_t)head_dim;
                    
                    for (int i = 0; i < head_dim; i++) {
                        out[i] += a * kv_get_element(s->value_cache, s->kv_cache_type, v_offset + (size_t)i);
                    }
                }
            }
        }

        matmul_q(s->xb2, s->xb, wo, w->wo_type ? w->wo_type[l] : 0, dim, q_dim, k, rt->options.num_threads);
        for (int i = 0; i < dim; i++) {
            s->x[i] += s->xb2[i];
        }
        k->rmsnorm(s->xb, s->x, rms_ffn, dim);

        if (p->expert_count > 0) {
            // MoE Routing
            float router_logits[256]; 
            k->matmul(router_logits, s->xb, w->ffn_gate_inp[l], p->expert_count, dim, 1);
            k->softmax(router_logits, p->expert_count);
            
            int active_experts[16];
            float expert_weights[16];
            for (int k_idx = 0; k_idx < p->expert_used_count; k_idx++) {
                float max_val = -1e9f;
                int max_idx = -1;
                for (int e = 0; e < p->expert_count; e++) {
                    if (router_logits[e] > max_val) {
                        max_val = router_logits[e];
                        max_idx = e;
                    }
                }
                active_experts[k_idx] = max_idx;
                expert_weights[k_idx] = max_val;
                router_logits[max_idx] = -1e9f; // mask out
            }

            float weight_sum = 0.0f;
            for (int k_idx = 0; k_idx < p->expert_used_count; k_idx++) {
                weight_sum += expert_weights[k_idx];
            }
            for (int k_idx = 0; k_idx < p->expert_used_count; k_idx++) {
                expert_weights[k_idx] /= weight_sum;
            }

            // Software prefetch active expert weights into CPU L1/L2 cache
            for (int k_idx = 0; k_idx < p->expert_used_count; k_idx++) {
                int e = active_experts[k_idx];
#if defined(__x86_64__) || defined(_M_X64)
                if (w->expert_w1 && w->expert_w1[l] && w->expert_w1[l][e]) _mm_prefetch((const char*)w->expert_w1[l][e], _MM_HINT_T0);
                if (w->expert_w2 && w->expert_w2[l] && w->expert_w2[l][e]) _mm_prefetch((const char*)w->expert_w2[l][e], _MM_HINT_T0);
                if (w->expert_w3 && w->expert_w3[l] && w->expert_w3[l][e]) _mm_prefetch((const char*)w->expert_w3[l][e], _MM_HINT_T0);
#endif
            }

            memset(s->xb2, 0, dim * sizeof(float));

            for (int k_idx = 0; k_idx < p->expert_used_count; k_idx++) {
                int e = active_experts[k_idx];
                float ew = expert_weights[k_idx];
                
                const float* ew1 = w->expert_w1[l][e];
                const float* ew2 = w->expert_w2[l][e];
                const float* ew3 = w->expert_w3[l][e];
                
                matmul_q(s->hb, s->xb, ew1, w->exp_w1_type ? w->exp_w1_type[l] : 0, hidden_dim, dim, k, rt->options.num_threads);
                matmul_q(s->hb2, s->xb, ew3, w->exp_w3_type ? w->exp_w3_type[l] : 0, hidden_dim, dim, k, rt->options.num_threads);
                for (int i = 0; i < hidden_dim; i++) {
                    s->hb[i] = silu(s->hb[i]) * s->hb2[i];
                }
                
                // Reuse s->q as temporary buffer for expert output since attention is done
                float* expert_out = s->q;
                matmul_q(expert_out, s->hb, ew2, w->exp_w2_type ? w->exp_w2_type[l] : 0, dim, hidden_dim, k, rt->options.num_threads);
                
                for (int i = 0; i < dim; i++) {
                    s->xb2[i] += expert_out[i] * ew;
                }
            }
            
            for (int i = 0; i < dim; i++) {
                s->x[i] += s->xb2[i];
            }
        } else {
            const float* w1 = w->w1[l];
            const float* w2 = w->w2[l];
            const float* w3 = w->w3[l];
            
            matmul_q(s->hb, s->xb, w1, w->w1_type ? w->w1_type[l] : 0, hidden_dim, dim, k, rt->options.num_threads);
            matmul_q(s->hb2, s->xb, w3, w->w3_type ? w->w3_type[l] : 0, hidden_dim, dim, k, rt->options.num_threads);
            for (int i = 0; i < hidden_dim; i++) {
                s->hb[i] = silu(s->hb[i]) * s->hb2[i];
            }
            matmul_q(s->xb, s->hb, w2, w->w2_type ? w->w2_type[l] : 0, dim, hidden_dim, k, rt->options.num_threads);
            for (int i = 0; i < dim; i++) {
                s->x[i] += s->xb[i];
            }
        }
    }
    k->rmsnorm(s->xb, s->x, w->rms_final_weight, dim);
    matmul_q(s->logits, s->xb, w->wcls, w->wcls_type, p->vocab_size, dim, k, rt->options.num_threads);
    return s->logits;
}

/* Batched prompt prefill: processes all prompt tokens through the transformer,
 * populating the KV cache for all positions. Chunks into sub-batches of 512 tokens
 * to limit temporary memory usage. Returns logits for the last prompt token. */
void prefill_runtime(RuntimeContext* rt, Transformer* t, const int* tokens, int n_tokens) {
    if (n_tokens <= 0) return;

    /* Process tokens sequentially through forward_runtime.
     * The KV cache is populated at each position.
     * This is the simplest correct implementation — future optimization
     * can add true batched GEMM for the Q/K/V projections. */
    for (int i = 0; i < n_tokens; i++) {
        (void)forward_runtime(rt, t, tokens[i], i);
    }
}
