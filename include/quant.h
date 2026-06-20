#ifndef QUANT_H
#define QUANT_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "types.h"

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
