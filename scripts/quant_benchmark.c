/**
 * @file quant_benchmark.c
 * @brief Multi-quantization comparative benchmark for Q4_K, Q5_K, Q6_K, and Q8_0.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "quant.h"

#ifdef _WIN32
#include <windows.h>
static double get_time_sec(void) {
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
}
#else
static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
#endif

#define N_ELEMS (2048 * 2048)
#define ITERS 20

int main(void) {
    printf("=== Baremetal Engine Quantization Micro-Benchmark ===\n");
    printf("Elements per run: %d (%.2f MB FP32)\n", N_ELEMS, (double)(N_ELEMS * 4) / (1024.0 * 1024.0));
    printf("Iterations: %d\n\n", ITERS);

    float* in = (float*)malloc(N_ELEMS * sizeof(float));
    float* out = (float*)malloc(N_ELEMS * sizeof(float));
    for (int i = 0; i < N_ELEMS; i++) in[i] = ((float)(i % 100) - 50.0f) * 0.05f;

    // 1. Q8_0 Benchmark
    int n_q8_blocks = N_ELEMS / 32;
    block_q8_0* q8_data = (block_q8_0*)malloc(n_q8_blocks * sizeof(block_q8_0));
    for (int i = 0; i < n_q8_blocks; i++) quantize_row_q8_0(&in[i * 32], &q8_data[i], 32);

    double t0 = get_time_sec();
    for (int it = 0; it < ITERS; it++) {
        dequantize_row(out, q8_data, 8, N_ELEMS);
    }
    double t1 = get_time_sec();
    double q8_time = (t1 - t0) / ITERS;
    double q8_bw = ((double)N_ELEMS * sizeof(float)) / (q8_time * 1e9);

    // 2. Q4_K Benchmark
    int n_q4_blocks = N_ELEMS / 256;
    block_q4_K* q4_data = (block_q4_K*)malloc(n_q4_blocks * sizeof(block_q4_K));
    for (int i = 0; i < n_q4_blocks; i++) {
        q4_data[i].d = fp32_to_fp16(0.1f);
        q4_data[i].dmin = fp32_to_fp16(0.01f);
        memset(q4_data[i].scales, 1, K_SCALE_SIZE);
        memset(q4_data[i].qs, 0x22, QK_K/2);
    }
    t0 = get_time_sec();
    for (int it = 0; it < ITERS; it++) {
        dequantize_row(out, q4_data, 12, N_ELEMS);
    }
    t1 = get_time_sec();
    double q4_time = (t1 - t0) / ITERS;
    double q4_bw = ((double)N_ELEMS * sizeof(float)) / (q4_time * 1e9);

    // 3. Q5_K Benchmark
    int n_q5_blocks = N_ELEMS / 256;
    block_q5_K* q5_data = (block_q5_K*)malloc(n_q5_blocks * sizeof(block_q5_K));
    for (int i = 0; i < n_q5_blocks; i++) {
        q5_data[i].d = fp32_to_fp16(0.1f);
        q5_data[i].dmin = fp32_to_fp16(0.01f);
        memset(q5_data[i].scales, 1, K_SCALE_SIZE);
        memset(q5_data[i].qs, 0x33, QK_K/2);
        memset(q5_data[i].qh, 0x01, QK_K/8);
    }
    t0 = get_time_sec();
    for (int it = 0; it < ITERS; it++) {
        dequantize_row(out, q5_data, 13, N_ELEMS);
    }
    t1 = get_time_sec();
    double q5_time = (t1 - t0) / ITERS;
    double q5_bw = ((double)N_ELEMS * sizeof(float)) / (q5_time * 1e9);

    // 4. Q6_K Benchmark
    int n_q6_blocks = N_ELEMS / 256;
    uint8_t* q6_data = (uint8_t*)malloc(n_q6_blocks * 210);
    for (int i = 0; i < n_q6_blocks; i++) {
        uint8_t* block = q6_data + i * 210;
        memset(block, 0x44, 128);
        memset(block + 128, 0x02, 64);
        memset(block + 192, 1, 16);
        uint16_t d = fp32_to_fp16(0.1f);
        memcpy(block + 208, &d, 2);
    }
    t0 = get_time_sec();
    for (int it = 0; it < ITERS; it++) {
        dequantize_row(out, q6_data, 14, N_ELEMS);
    }
    t1 = get_time_sec();
    double q6_time = (t1 - t0) / ITERS;
    double q6_bw = ((double)N_ELEMS * sizeof(float)) / (q6_time * 1e9);

    printf("| Quantization Format | Bits / Weight | Block Size (Bytes) | Dequant Latency (ms) | SIMD Throughput (GB/s) | Theoretical 30B Memory |\n");
    printf("| :--- | :---: | :---: | :---: | :---: | :---: |\n");
    printf("| **Q4_K_M** | 4.50 bpw | 144 B / 256 | %8.2f ms | %8.2f GB/s | **~17.2 GB** |\n", q4_time * 1000.0, q4_bw);
    printf("| **Q5_K_M** | 5.50 bpw | 176 B / 256 | %8.2f ms | %8.2f GB/s | **~21.7 GB** |\n", q5_time * 1000.0, q5_bw);
    printf("| **Q6_K**   | 6.56 bpw | 210 B / 256 | %8.2f ms | %8.2f GB/s | **~25.1 GB** |\n", q6_time * 1000.0, q6_bw);
    printf("| **Q8_0**   | 8.50 bpw |  34 B / 32  | %8.2f ms | %8.2f GB/s | **~32.5 GB** |\n", q8_time * 1000.0, q8_bw);

    free(in); free(out); free(q8_data); free(q4_data); free(q5_data); free(q6_data);
    return 0;
}
