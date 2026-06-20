/**
 * @file benchmark.h
 * @brief High-precision performance profiler, memory governor, and latency metrics.
 */

#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "model.h"
#include "runtime.h"
#include "tokenizer.h"

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
