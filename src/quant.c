#include "quant.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

static int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

int quantize_int8_groupwise(QuantInt8Matrix* q, const float* w, int rows, int cols, int group_size) {
    memset(q, 0, sizeof(*q));
    q->rows = rows;
    q->cols = cols;
    q->group_size = group_size;
    int groups_per_row = ceil_div(cols, group_size);
    q->data = (int8_t*)malloc((size_t)rows * (size_t)cols);
    q->scales = (float*)malloc(sizeof(float) * (size_t)rows * (size_t)groups_per_row);
    q->zeros = (float*)malloc(sizeof(float) * (size_t)rows * (size_t)groups_per_row);
    if (q->data == NULL || q->scales == NULL || q->zeros == NULL) {
        free_quant_int8(q);
        return -1;
    }
    for (int r = 0; r < rows; r++) {
        for (int g = 0; g < groups_per_row; g++) {
            int c0 = g * group_size;
            int c1 = c0 + group_size;
            if (c1 > cols) c1 = cols;
            float mn = FLT_MAX;
            float mx = -FLT_MAX;
            for (int c = c0; c < c1; c++) {
                float v = w[(size_t)r * (size_t)cols + (size_t)c];
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            float scale = (mx - mn) / 255.0f;
            if (scale < 1e-8f) scale = 1e-8f;
            q->scales[(size_t)r * (size_t)groups_per_row + (size_t)g] = scale;
            q->zeros[(size_t)r * (size_t)groups_per_row + (size_t)g] = mn;
            for (int c = c0; c < c1; c++) {
                float v = w[(size_t)r * (size_t)cols + (size_t)c];
                int qv = (int)lrintf((v - mn) / scale) - 128;
                if (qv < -128) qv = -128;
                if (qv > 127) qv = 127;
                q->data[(size_t)r * (size_t)cols + (size_t)c] = (int8_t)qv;
            }
        }
    }
    return 0;
}

int quantize_qk_groupwise(QuantKMatrix* q, const float* w, int rows, int cols, int group_size, int bits) {
    memset(q, 0, sizeof(*q));
    q->rows = rows;
    q->cols = cols;
    q->group_size = group_size;
    q->bits = bits;
    int levels = (1 << bits) - 1;
    int groups_per_row = ceil_div(cols, group_size);
    q->data = (uint8_t*)malloc((size_t)rows * (size_t)cols);
    q->scales = (float*)malloc(sizeof(float) * (size_t)rows * (size_t)groups_per_row);
    q->mins = (float*)malloc(sizeof(float) * (size_t)rows * (size_t)groups_per_row);
    if (q->data == NULL || q->scales == NULL || q->mins == NULL) {
        free_quant_k(q);
        return -1;
    }
    for (int r = 0; r < rows; r++) {
        for (int g = 0; g < groups_per_row; g++) {
            int c0 = g * group_size;
            int c1 = c0 + group_size;
            if (c1 > cols) c1 = cols;
            float mn = FLT_MAX;
            float mx = -FLT_MAX;
            for (int c = c0; c < c1; c++) {
                float v = w[(size_t)r * (size_t)cols + (size_t)c];
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            float scale = (mx - mn) / (float)levels;
            if (scale < 1e-8f) scale = 1e-8f;
            q->scales[(size_t)r * (size_t)groups_per_row + (size_t)g] = scale;
            q->mins[(size_t)r * (size_t)groups_per_row + (size_t)g] = mn;
            for (int c = c0; c < c1; c++) {
                float v = w[(size_t)r * (size_t)cols + (size_t)c];
                int qv = (int)lrintf((v - mn) / scale);
                if (qv < 0) qv = 0;
                if (qv > levels) qv = levels;
                q->data[(size_t)r * (size_t)cols + (size_t)c] = (uint8_t)qv;
            }
        }
    }
    return 0;
}

void matmul_int8_groupwise(float* out, const float* x, const QuantInt8Matrix* q) {
    int groups_per_row = ceil_div(q->cols, q->group_size);
    for (int r = 0; r < q->rows; r++) {
        float sum = 0.0f;
        for (int c = 0; c < q->cols; c++) {
            int g = c / q->group_size;
            float scale = q->scales[(size_t)r * (size_t)groups_per_row + (size_t)g];
            float zero = q->zeros[(size_t)r * (size_t)groups_per_row + (size_t)g];
            float w = ((float)q->data[(size_t)r * (size_t)q->cols + (size_t)c] + 128.0f) * scale + zero;
            sum += w * x[c];
        }
        out[r] = sum;
    }
}

void matmul_qk_groupwise(float* out, const float* x, const QuantKMatrix* q) {
    int groups_per_row = ceil_div(q->cols, q->group_size);
    for (int r = 0; r < q->rows; r++) {
        float sum = 0.0f;
        for (int c = 0; c < q->cols; c++) {
            int g = c / q->group_size;
            float scale = q->scales[(size_t)r * (size_t)groups_per_row + (size_t)g];
            float mn = q->mins[(size_t)r * (size_t)groups_per_row + (size_t)g];
            float w = ((float)q->data[(size_t)r * (size_t)q->cols + (size_t)c] * scale) + mn;
            sum += w * x[c];
        }
        out[r] = sum;
    }
}

void free_quant_int8(QuantInt8Matrix* q) {
    free(q->data);
    free(q->scales);
    free(q->zeros);
    memset(q, 0, sizeof(*q));
}

void free_quant_k(QuantKMatrix* q) {
    free(q->data);
    free(q->scales);
    free(q->mins);
    memset(q, 0, sizeof(*q));
}

void quantize_row_q8_0(const float* x, block_q8_0* y, int k) {
    int nb = k / QK8_0;
    for (int i = 0; i < nb; i++) {
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; j++) {
            float v = fabsf(x[i * QK8_0 + j]);
            if (v > amax) amax = v;
        }
        float d = amax / 127.0f;
        float id = d ? 1.0f / d : 0.0f;
        y[i].d = fp32_to_fp16(d);
        for (int j = 0; j < QK8_0; j++) {
            y[i].qs[j] = (int8_t)lrintf(x[i * QK8_0 + j] * id);
        }
    }
}

void dequantize_row(float* out, const void* src, uint32_t type, int n) {
    if (src == NULL) {
        memset(out, 0, (size_t)n * sizeof(float));
        return;
    }
    if (type == 0) { // F32
        memcpy(out, src, (size_t)n * sizeof(float));
        return;
    }
    if (type == 8) { // Q8_0
        const block_q8_0* b = (const block_q8_0*)src;
        int nb = n / QK8_0;
        for (int i = 0; i < nb; i++) {
            float d = fp16_to_fp32(b[i].d);
            for (int j = 0; j < QK8_0; j++) {
                out[i * QK8_0 + j] = (float)b[i].qs[j] * d;
            }
        }
        return;
    }
    if (type == 12) { // Q4_K
        const block_q4_K* b = (const block_q4_K*)src;
        int nb = n / QK_K;
        for (int i = 0; i < nb; i++) {
            const float d = fp16_to_fp32(b[i].d);
            const float dmin = fp16_to_fp32(b[i].dmin);
            for (int j = 0; j < QK_K/32; j++) {
                float sc = (float)(b[i].scales[j%K_SCALE_SIZE] & 63) * d;
                float m  = (float)((b[i].scales[j%K_SCALE_SIZE] >> 6)) * dmin;
                for (int l = 0; l < 32; l++) {
                    int q = (b[i].qs[(j*32 + l)/2] >> ((l%2)*4)) & 0xF;
                    out[i * QK_K + j * 32 + l] = (float)q * sc - m;
                }
            }
        }
        return;
    }
    if (type == 13) { // Q5_K
        const block_q5_K* b = (const block_q5_K*)src;
        int nb = n / QK_K;
        for (int i = 0; i < nb; i++) {
            const float d = fp16_to_fp32(b[i].d);
            const float dmin = fp16_to_fp32(b[i].dmin);
            for (int j = 0; j < QK_K/32; j++) {
                float sc = (float)(b[i].scales[j%K_SCALE_SIZE] & 63) * d;
                float m  = (float)((b[i].scales[j%K_SCALE_SIZE] >> 6)) * dmin;
                for (int l = 0; l < 32; l++) {
                    int q4 = (b[i].qs[(j*32 + l)/2] >> ((l%2)*4)) & 0xF;
                    int qh = (b[i].qh[(j*32 + l)/8] >> ((j*32 + l)%8)) & 1;
                    int q = q4 | (qh << 4);
                    out[i * QK_K + j * 32 + l] = (float)q * sc - m;
                }
            }
        }
        return;
    }
    if (type == 14) { // Q6_K
        const uint8_t* raw = (const uint8_t*)src;
        int nb = n / QK_K;
        for (int i = 0; i < nb; i++) {
            const uint8_t* block = raw + i * 210;
            const uint8_t* ql = block;
            const uint8_t* qh = block + 128;
            const int8_t* scales = (const int8_t*)(block + 192);
            uint16_t d_raw;
            memcpy(&d_raw, block + 208, 2);
            float d = fp16_to_fp32(d_raw);
            for (int j = 0; j < QK_K/32; j++) {
                float sc = (float)scales[j] * d;
                for (int l = 0; l < 32; l++) {
                    int ql_val = (ql[(j*32 + l)/2] >> ((l%2)*4)) & 0xF;
                    int qh_val = (qh[(j*32 + l)/4] >> (((j*32 + l)%4)*2)) & 3;
                    int q = (ql_val | (qh_val << 4)) - 32;
                    out[i * QK_K + j * 32 + l] = (float)q * sc;
                }
            }
        }
        return;
    }
    memset(out, 0, (size_t)n * sizeof(float));
}

float vec_dot_q8_0_q8_0(const block_q8_0* a, const block_q8_0* b, int k) {
    int nb = k / QK8_0;
    float sum = 0.0f;
#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < nb; i++) {
        __m256i q1 = _mm256_loadu_si256((const __m256i*)a[i].qs);
        __m256i q2 = _mm256_loadu_si256((const __m256i*)b[i].qs);
        
        __m128i q1_lo = _mm256_castsi256_si128(q1);
        __m128i q1_hi = _mm256_extracti128_si256(q1, 1);
        __m128i q2_lo = _mm256_castsi256_si128(q2);
        __m128i q2_hi = _mm256_extracti128_si256(q2, 1);
        
        __m256i v1_lo = _mm256_cvtepi8_epi16(q1_lo);
        __m256i v1_hi = _mm256_cvtepi8_epi16(q1_hi);
        __m256i v2_lo = _mm256_cvtepi8_epi16(q2_lo);
        __m256i v2_hi = _mm256_cvtepi8_epi16(q2_hi);
        
        __m256i mul_lo = _mm256_madd_epi16(v1_lo, v2_lo);
        __m256i mul_hi = _mm256_madd_epi16(v1_hi, v2_hi);
        
        __m256i s = _mm256_add_epi32(mul_lo, mul_hi);
        __m256 fs = _mm256_cvtepi32_ps(s);
        
        float d = fp16_to_fp32(a[i].d) * fp16_to_fp32(b[i].d);
        __m256 vd = _mm256_set1_ps(d);
#if defined(__AVX512VNNI__) || defined(__AVX512F__)
        acc = _mm256_fmadd_ps(fs, vd, acc);
#else
        acc = _mm256_add_ps(_mm256_mul_ps(fs, vd), acc);
#endif
    }
    float f[8];
    _mm256_storeu_ps(f, acc);
    for(int i = 0; i < 8; i++) sum += f[i];
#else
    for (int i = 0; i < nb; i++) {
        int isum = 0;
        for (int j = 0; j < QK8_0; j++) {
            isum += a[i].qs[j] * b[i].qs[j];
        }
        sum += (float)isum * fp16_to_fp32(a[i].d) * fp16_to_fp32(b[i].d);
    }
#endif
    return sum;
}

// Minimal scalar fallback for Q4_K dot Q8_0 (llama.cpp compatible unpacking)
float vec_dot_q4K_q8_0(const block_q4_K* a, const block_q8_0* b, int k) {
    int nb = k / QK_K;
    float sumf = 0.0f;
#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    __m128i m15 = _mm_set1_epi8(0xF);
    __m256i ones = _mm256_set1_epi16(1);

    for (int i = 0; i < nb; i++) {
        const uint8_t* q4 = a[i].qs;
        const uint8_t* scales = a[i].scales;
        const float d = a[i].d;
        const float dmin = a[i].dmin;
        
        const block_q8_0* b8 = &b[i * (QK_K/QK8_0)];
        
        for (int j = 0; j < QK_K/32; j++) {
            float sc = (float)(scales[j%K_SCALE_SIZE] & 63) * d * b8[j].d;
            float m  = (float)((scales[j%K_SCALE_SIZE] >> 6)) * dmin * b8[j].d;
            
            __m128i q4_128 = _mm_loadu_si128((const __m128i*)(q4 + j * 16));
            __m128i lo = _mm_and_si128(q4_128, m15);
            __m128i hi = _mm_and_si128(_mm_srli_epi16(q4_128, 4), m15);
            
            __m128i lo_hi_0 = _mm_unpacklo_epi8(lo, hi);
            __m128i lo_hi_1 = _mm_unpackhi_epi8(lo, hi);
            
            __m256i q16_0 = _mm256_cvtepu8_epi16(lo_hi_0);
            __m256i q16_1 = _mm256_cvtepu8_epi16(lo_hi_1);
            
            __m256i b8_val = _mm256_loadu_si256((const __m256i*)b8[j].qs);
            __m256i b16_0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b8_val));
            __m256i b16_1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b8_val, 1));
            
            __m256i mul0 = _mm256_madd_epi16(q16_0, b16_0);
            __m256i mul1 = _mm256_madd_epi16(q16_1, b16_1);
            __m256i sum_int = _mm256_add_epi32(mul0, mul1);
            
            __m256i b_sum0 = _mm256_madd_epi16(b16_0, ones);
            __m256i b_sum1 = _mm256_madd_epi16(b16_1, ones);
            __m256i b_sum_int = _mm256_add_epi32(b_sum0, b_sum1);
            
            __m256 f_sum = _mm256_cvtepi32_ps(sum_int);
            __m256 f_bsum = _mm256_cvtepi32_ps(b_sum_int);
            
            __m256 f_sc = _mm256_set1_ps(sc);
            __m256 f_m = _mm256_set1_ps(m);
            
            acc = _mm256_add_ps(acc, _mm256_sub_ps(_mm256_mul_ps(f_sum, f_sc), _mm256_mul_ps(f_bsum, f_m)));
        }
    }
    float f[8];
    _mm256_storeu_ps(f, acc);
    for(int i = 0; i < 8; i++) sumf += f[i];
#else
    for (int i = 0; i < nb; i++) {
        const uint8_t* q4 = a[i].qs;
        const uint8_t* scales = a[i].scales;
        const float d = a[i].d;
        const float dmin = a[i].dmin;
        
        const block_q8_0* b8 = &b[i * (QK_K/QK8_0)];
        
        int step = 32;
        for (int j = 0; j < QK_K/step; j++) {
            float sc = (float)(scales[j%K_SCALE_SIZE] & 63) * d;
            float m  = (float)((scales[j%K_SCALE_SIZE] >> 6)) * dmin;
            
            int isum = 0;
            for (int l = 0; l < step; l++) {
                int q = (q4[(j*step + l)/2] >> ((l%2)*4)) & 0xF;
                isum += (q * sc - m) * b8[j].qs[l];
            }
            sumf += isum * b8[j].d;
        }
    }
#endif
    return sumf;
}

// Fused Q5_K x Q8_0 dot product (AVX2 integer SIMD with scalar fallback)
float vec_dot_q5K_q8_0(const block_q5_K* a, const block_q8_0* b, int k) {
    int nb = k / QK_K;
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        const uint8_t* q5 = a[i].qs;
        const uint8_t* qh = a[i].qh;
        const uint8_t* scales = a[i].scales;
        const float d = fp16_to_fp32(a[i].d);
        const float dmin = fp16_to_fp32(a[i].dmin);

        const block_q8_0* b8 = &b[i * (QK_K/QK8_0)];

        for (int j = 0; j < QK_K/32; j++) {
            float sc = (float)(scales[j%K_SCALE_SIZE] & 63) * d;
            float m  = (float)((scales[j%K_SCALE_SIZE] >> 6)) * dmin;

            int isum = 0;
            int bsum = 0;
            for (int l = 0; l < 32; l++) {
                int q4 = (q5[(j*32 + l)/2] >> ((l%2)*4)) & 0xF;
                int qh_bit = (qh[(j*32 + l)/8] >> ((j*32 + l)%8)) & 1;
                int q = q4 | (qh_bit << 4);
                int8_t bval = b8[j].qs[l];
                isum += q * bval;
                bsum += bval;
            }
            float bd = fp16_to_fp32(b8[j].d);
            sumf += (sc * (float)isum - m * (float)bsum) * bd;
        }
    }
    return sumf;
}

// Fused Q6_K x Q8_0 dot product (scalar implementation)
float vec_dot_q6K_q8_0(const void* a, const block_q8_0* b, int k) {
    int nb = k / QK_K;
    float sumf = 0.0f;
    const uint8_t* raw = (const uint8_t*)a;

    for (int i = 0; i < nb; i++) {
        const uint8_t* block = raw + i * 210;
        const uint8_t* ql = block;
        const uint8_t* qh_data = block + 128;
        const int8_t* scales = (const int8_t*)(block + 192);
        uint16_t d_raw;
        memcpy(&d_raw, block + 208, 2);
        float d = fp16_to_fp32(d_raw);

        const block_q8_0* b8 = &b[i * (QK_K/QK8_0)];

        for (int j = 0; j < QK_K/32; j++) {
            float sc = (float)scales[j] * d;

            int isum = 0;
            for (int l = 0; l < 32; l++) {
                int ql_val = (ql[(j*32 + l)/2] >> ((l%2)*4)) & 0xF;
                int qh_val = (qh_data[(j*32 + l)/4] >> (((j*32 + l)%4)*2)) & 3;
                int q = (ql_val | (qh_val << 4)) - 32;
                isum += q * (int)b8[j].qs[l];
            }
            float bd = fp16_to_fp32(b8[j].d);
            sumf += sc * (float)isum * bd;
        }
    }
    return sumf;
}

// Fused quantized matmul: quantize x to Q8_0 once, then use fused dot products
// Eliminates per-row FP32 dequantization buffers entirely
void matmul_q_fused(float* out, const float* x, const void* w, uint32_t type,
                    int n, int d, int threads) {
    if (w == NULL) {
        memset(out, 0, (size_t)n * sizeof(float));
        return;
    }
    // F32 weights: plain FP32 matmul (no quantization involved)
    if (type == 0) {
        const float* wf = (const float*)w;
#ifdef _OPENMP
        int nth = threads > 0 ? threads : 1;
#pragma omp parallel for num_threads(nth) schedule(static)
#endif
        for (int i = 0; i < n; i++) {
            const float* row = wf + (size_t)i * (size_t)d;
            float sum = 0.0f;
            for (int j = 0; j < d; j++) {
                sum += row[j] * x[j];
            }
            out[i] = sum;
        }
        return;
    }

    // Quantize input activation vector x to Q8_0 (once per matmul call)
    int nb_x = d / QK8_0;
    if (nb_x <= 0) nb_x = 1;
    block_q8_0* x_q8 = (block_q8_0*)malloc((size_t)nb_x * sizeof(block_q8_0));
    if (!x_q8) {
        memset(out, 0, (size_t)n * sizeof(float));
        return;
    }
    quantize_row_q8_0(x, x_q8, d);

    // Compute row byte sizes for each quantization type
    size_t row_bytes;
    switch (type) {
        case 8:  row_bytes = (size_t)(d / 32) * sizeof(block_q8_0); break;  // Q8_0
        case 12: row_bytes = (size_t)(d / 256) * sizeof(block_q4_K); break; // Q4_K
        case 13: row_bytes = (size_t)(d / 256) * 176; break;               // Q5_K
        case 14: row_bytes = (size_t)(d / 256) * 210; break;               // Q6_K
        default: row_bytes = (size_t)d * sizeof(float); break;
    }

#ifdef _OPENMP
    int nth = threads > 0 ? threads : 1;
#pragma omp parallel for num_threads(nth) schedule(static)
#endif
    for (int i = 0; i < n; i++) {
        const char* row_ptr = (const char*)w + (size_t)i * row_bytes;
        switch (type) {
            case 8:  // Q8_0: fused int8 dot product
                out[i] = vec_dot_q8_0_q8_0((const block_q8_0*)row_ptr, x_q8, d);
                break;
            case 12: // Q4_K: fused mixed-precision dot product
                out[i] = vec_dot_q4K_q8_0((const block_q4_K*)row_ptr, x_q8, d);
                break;
            case 13: // Q5_K: fused mixed-precision dot product
                out[i] = vec_dot_q5K_q8_0((const block_q5_K*)row_ptr, x_q8, d);
                break;
            case 14: // Q6_K: fused mixed-precision dot product
                out[i] = vec_dot_q6K_q8_0(row_ptr, x_q8, d);
                break;
            default: {
                // Fallback: dequantize row to FP32 then dot
                float* buf = (float*)malloc((size_t)d * sizeof(float));
                if (buf) {
                    dequantize_row(buf, row_ptr, type, d);
                    float sum = 0.0f;
                    for (int j = 0; j < d; j++) sum += buf[j] * x[j];
                    out[i] = sum;
                    free(buf);
                } else {
                    out[i] = 0.0f;
                }
                break;
            }
        }
    }
    free(x_q8);
}
