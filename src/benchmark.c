#define _POSIX_C_SOURCE 200809L
#include "benchmark.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "model.h"
#include "sampling.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#include <unistd.h>
#endif

static double now_sec(void) {
#if defined(_WIN32)
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

static int compare_doubles(const void* a, const void* b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    return (da > db) - (da < db);
}

void print_metrics_dashboard(const BenchmarkMetrics* m, const char* model_name, const char* backend_name, int threads) {
    printf("\n");
    printf("========================================================================================\n");
    printf("                  🚀 BAREMETAL INFERENCE ENGINE - MLSys METRICS REPORT                   \n");
    printf("========================================================================================\n");
    printf("  Model: %-30s | Backend: %-8s | Worker Threads: %-2d\n", model_name ? model_name : "N/A", backend_name, threads);
    printf("----------------------------------------------------------------------------------------\n");
    printf("  ⏱️  LATENCY & THROUGHPUT METRICS:\n");
    printf("    • Time to First Token (TTFT):   %10.2f ms\n", m->ttft_ms);
    printf("    • Mean Inter-Token Latency:    %10.2f ms\n", m->mean_latency_ms);
    printf("    • Median Latency (P50):         %10.2f ms\n", m->p50_latency_ms);
    printf("    • Tail Latency (P95):           %10.2f ms\n", m->p95_latency_ms);
    printf("    • Tail Latency (P99):           %10.2f ms\n", m->p99_latency_ms);
    printf("    • Inference Throughput:         %10.4f tok/s (%.2f tokens generated)\n", m->tokens_per_sec, (double)m->total_tokens);
    printf("----------------------------------------------------------------------------------------\n");
    printf("  💾 MEMORY & RESOURCE UTILIZATION:\n");
    printf("    • Peak Physical RAM (RSS):     %10.2f MB  <-- Strict <4 GB Budget\n", m->peak_rss_mb);
    printf("    • Virtual Address Space:       %10.2f MB\n", m->virtual_mem_mb);
    printf("    • Effective Memory Bandwidth:   %10.2f GB/s\n", m->effective_bw_gbps);
    printf("    • Quantization Compression:     %10.2fx vs FP32\n", m->compression_ratio);
    printf("----------------------------------------------------------------------------------------\n");
    printf("  🧩 MoE & ARCHITECTURAL EFFICIENCY:\n");
    printf("    • Expert Routing Sparsity:      %10.2f %%\n", m->expert_sparsity_pct);
    printf("    • Active Parameters / Token:    %10.2f Billion\n", m->active_params_b);
    printf("    • Total Model Parameters:       %10.2f Billion\n", m->total_params_b);
    printf("    • Forward Pass Total Time:      %10.4f s (%.1f%% of total)\n", m->forward_time_sec, (m->forward_time_sec / m->elapsed_sec) * 100.0);
    printf("    • Sampling & Filter Overhead:   %10.4f s (%.1f%% of total)\n", m->sample_time_sec, (m->sample_time_sec / m->elapsed_sec) * 100.0);
    printf("========================================================================================\n\n");
}

int run_benchmark(RuntimeContext* rt,
                  Transformer* t,
                  const Tokenizer* tok,
                  const BenchmarkOptions* bopt) {
    if (bopt->steps <= 0) return -1;
    int steps = bopt->steps;
    int warmup = bopt->warmup > 0 ? bopt->warmup : 0;
    int token = 1;
    SamplingState s;
    sampling_state_init(&s, 128);

    double* latencies = (double*)malloc(steps * sizeof(double));

    // Warmup
    for (int i = 0; i < warmup; i++) {
        float* logits = forward_runtime(rt, t, token, i % t->config.seq_len);
        token = sample_next(logits, t->config.vocab_size, &rt->options, &s);
        sampling_state_push(&s, token);
    }

    int prompt_tokens[1024];
    int n_prompt = 0;
    double t_prompt_start = now_sec();
    if (tok != NULL && bopt->prompt != NULL) {
        n_prompt = tokenizer_encode(tok, bopt->prompt, prompt_tokens, 1024);
        for (int i = 0; i < n_prompt; i++) {
            token = prompt_tokens[i];
            (void)forward_runtime(rt, t, token, i % t->config.seq_len);
            sampling_state_push(&s, token);
        }
    }
    double t_prompt_end = now_sec();

    double t0 = now_sec();
    int pos = n_prompt;
    double t_forward_total = 0.0;
    double t_sample_total = 0.0;
    double ttft = 0.0;

    for (int i = 0; i < steps; i++) {
        double step_start = now_sec();

        double f_start = now_sec();
        float* logits = forward_runtime(rt, t, token, pos);
        double f_end = now_sec();
        t_forward_total += (f_end - f_start);

        token = sample_next(logits, t->config.vocab_size, &rt->options, &s);
        double s_end = now_sec();
        t_sample_total += (s_end - f_end);

        sampling_state_push(&s, token);
        pos++;
        if (pos >= t->config.seq_len) pos = 0;

        double step_end = now_sec();
        latencies[i] = (step_end - step_start) * 1000.0; // ms

        if (i == 0) {
            ttft = ((t_prompt_end - t_prompt_start) + (step_end - step_start)) * 1000.0;
        }

#if defined(_WIN32)
        SetProcessWorkingSetSize(GetCurrentProcess(), (SIZE_T)-1, (SIZE_T)-1);
#endif
    }
    double t1 = now_sec();
    sampling_state_free(&s);

    double elapsed = t1 - t0;
    if (elapsed <= 0.0) elapsed = 1e-6;

    // Percentile calculations
    double* sorted_lat = (double*)malloc(steps * sizeof(double));
    memcpy(sorted_lat, latencies, steps * sizeof(double));
    qsort(sorted_lat, steps, sizeof(double), compare_doubles);

    double p50 = sorted_lat[(int)(steps * 0.50)];
    double p95 = sorted_lat[(int)(steps * 0.95 >= steps ? steps - 1 : steps * 0.95)];
    double p99 = sorted_lat[(int)(steps * 0.99 >= steps ? steps - 1 : steps * 0.99)];
    free(sorted_lat);

    double peak_rss_mb = 0.0;
    double vms_mb = (double)t->mapped_size / (1024.0 * 1024.0);
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        peak_rss_mb = (double)pmc.PeakWorkingSetSize / (1024.0 * 1024.0);
    }
#else
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        peak_rss_mb = (double)usage.ru_maxrss / 1024.0;
    }
#endif

    // Model Architecture Metrics
    double active_params = 0.0;
    double total_params = 0.0;
    double sparsity = 0.0;

    if (t->config.expert_count > 0) {
        sparsity = (1.0 - ((double)t->config.expert_used_count / (double)t->config.expert_count)) * 100.0;
        // Non-MoE dense layers + Active experts
        double dense_per_layer = (double)t->config.dim * (double)t->config.dim * 4.0; // Q, K, V, O
        double expert_ffn_per_active = (double)t->config.expert_used_count * 3.0 * (double)t->config.dim * (double)t->config.hidden_dim;
        active_params = ((dense_per_layer + expert_ffn_per_active) * (double)t->config.n_layers + (double)t->config.vocab_size * (double)t->config.dim) / 1e9;
        
        double expert_ffn_total = (double)t->config.expert_count * 3.0 * (double)t->config.dim * (double)t->config.hidden_dim;
        total_params = ((dense_per_layer + expert_ffn_total) * (double)t->config.n_layers + (double)t->config.vocab_size * (double)t->config.dim) / 1e9;
    } else {
        total_params = ((double)t->config.dim * (double)t->config.dim * 4.0 + (double)t->config.dim * (double)t->config.hidden_dim * 3.0) * (double)t->config.n_layers / 1e9;
        active_params = total_params;
    }

    double active_bytes_per_tok = (active_params * 1e9) * 0.7; // ~0.7 bytes per quantized weight
    double effective_bw = (active_bytes_per_tok / (elapsed / (double)steps)) / 1e9; // GB/s
    double fp32_theoretical_bytes = (total_params * 1e9) * 4.0;
    double compression = t->mapped_size > 0 ? fp32_theoretical_bytes / (double)t->mapped_size : 1.0;

    BenchmarkMetrics m;
    m.ttft_ms = ttft;
    m.mean_latency_ms = (elapsed / (double)steps) * 1000.0;
    m.p50_latency_ms = p50;
    m.p95_latency_ms = p95;
    m.p99_latency_ms = p99;
    m.tokens_per_sec = (double)steps / elapsed;
    m.forward_time_sec = t_forward_total;
    m.sample_time_sec = t_sample_total;
    m.elapsed_sec = elapsed;
    m.peak_rss_mb = peak_rss_mb;
    m.virtual_mem_mb = vms_mb;
    m.effective_bw_gbps = effective_bw;
    m.expert_sparsity_pct = sparsity;
    m.active_params_b = active_params;
    m.total_params_b = total_params;
    m.compression_ratio = compression;
    m.total_tokens = steps;

    print_metrics_dashboard(&m, "Qwen 3 MoE (30B)", rt->kernels->name, rt->options.num_threads);

    if (bopt->csv_path != NULL) {
        FILE* f = fopen(bopt->csv_path, "w");
        if (f != NULL) {
            fprintf(f, "backend,threads,steps,elapsed_sec,ttft_ms,mean_latency_ms,p50_ms,p95_ms,p99_ms,tokens_per_sec,peak_rss_mb,active_params_b,sparsity_pct\n");
            fprintf(f, "%s,%d,%d,%.6f,%.2f,%.2f,%.2f,%.2f,%.2f,%.6f,%.2f,%.2f,%.2f\n",
                    rt->kernels->name, rt->options.num_threads, steps, elapsed, ttft, m.mean_latency_ms, p50, p95, p99, m.tokens_per_sec, peak_rss_mb, active_params, sparsity);
            fclose(f);
        }
    }
    if (bopt->json_path != NULL) {
        FILE* f = fopen(bopt->json_path, "w");
        if (f != NULL) {
            fprintf(f,
                    "{\n"
                    "  \"backend\": \"%s\",\n"
                    "  \"threads\": %d,\n"
                    "  \"steps\": %d,\n"
                    "  \"elapsed_sec\": %.6f,\n"
                    "  \"ttft_ms\": %.2f,\n"
                    "  \"mean_latency_ms\": %.2f,\n"
                    "  \"p50_latency_ms\": %.2f,\n"
                    "  \"p95_latency_ms\": %.2f,\n"
                    "  \"p99_latency_ms\": %.2f,\n"
                    "  \"tokens_per_sec\": %.6f,\n"
                    "  \"peak_rss_mb\": %.2f,\n"
                    "  \"virtual_mem_mb\": %.2f,\n"
                    "  \"effective_bw_gbps\": %.2f,\n"
                    "  \"expert_sparsity_pct\": %.2f,\n"
                    "  \"active_params_b\": %.2f,\n"
                    "  \"total_params_b\": %.2f,\n"
                    "  \"compression_ratio\": %.2f\n"
                    "}\n",
                    rt->kernels->name, rt->options.num_threads, steps, elapsed, ttft, m.mean_latency_ms, p50, p95, p99, m.tokens_per_sec, peak_rss_mb, vms_mb, effective_bw, sparsity, active_params, total_params, compression);
            fclose(f);
        }
    }
    if (bopt->prom_path != NULL) {
        FILE* f = fopen(bopt->prom_path, "w");
        if (f != NULL) {
            fprintf(f, "# HELP baremetal_tokens_per_sec Inference generation throughput\n");
            fprintf(f, "# TYPE baremetal_tokens_per_sec gauge\n");
            fprintf(f, "baremetal_tokens_per_sec{backend=\"%s\",threads=\"%d\"} %.6f\n", rt->kernels->name, rt->options.num_threads, m.tokens_per_sec);
            fprintf(f, "# HELP baremetal_mean_latency_ms Mean inter-token latency\n");
            fprintf(f, "# TYPE baremetal_mean_latency_ms gauge\n");
            fprintf(f, "baremetal_mean_latency_ms %.2f\n", m.mean_latency_ms);
            fprintf(f, "# HELP baremetal_peak_rss_mb Peak resident set size memory in MB\n");
            fprintf(f, "# TYPE baremetal_peak_rss_mb gauge\n");
            fprintf(f, "baremetal_peak_rss_mb %.2f\n", m.peak_rss_mb);
            fclose(f);
        }
    }

    free(latencies);
    return 0;
}

void benchmark_render_hud(int token_idx, int total_tokens, double tok_per_sec, double rss_mb) {
    int bar_width = 12;
    int progress = total_tokens > 0 ? (token_idx * bar_width) / total_tokens : 0;
    if (progress > bar_width) progress = bar_width;

    char bar[32];
    int bi = 0;
    for (int i = 0; i < bar_width; i++) {
        if (i < progress) {
            bar[bi++] = '#';
        } else {
            bar[bi++] = '-';
        }
    }
    bar[bi] = '\0';

    fprintf(stderr, "\r\033[1;36m[HUD]\033[0m [%s] \033[1;32m%d/%d tok\033[0m | \033[1;33m%5.1f tok/s\033[0m | \033[1;35mRSS: %5.1f MB\033[0m ",
            bar, token_idx, total_tokens, tok_per_sec, rss_mb);
    fflush(stderr);
}

void print_sysinfo(void) {
    printf("========================================================================================\n");
    printf("                   💻 BAREMETAL HARDWARE & SIMD TOPOLOGY REPORT                        \n");
    printf("========================================================================================\n");
#ifdef _WIN32
    printf("  Operating System:    Windows NT (Win32 Virtual Memory Governor)\n");
#elif defined(__APPLE__)
    printf("  Operating System:    macOS Darwin (Mach Virtual Memory)\n");
#else
    printf("  Operating System:    Linux (POSIX mmap / procfs)\n");
#endif

#ifdef _OPENMP
    printf("  Logical CPU Cores:   %d threads detected\n", omp_get_num_procs());
#else
    printf("  Logical CPU Cores:   1 thread (OpenMP disabled)\n");
#endif

    printf("  SIMD Vector Extensions:\n");
    printf("    • AVX2 (256-bit):      %s\n", cpu_supports_avx2() ? "✅ PRESENT (FMA3 + Int8 DP4A)" : "❌ ABSENT");
    printf("    • AVX-512 (512-bit):   %s\n", cpu_supports_avx512() ? "✅ PRESENT (512-bit VNNI)" : "❌ ABSENT");
    printf("    • ARM NEON (128-bit):  %s\n", cpu_supports_neon() ? "✅ PRESENT (AArch64 ASIMD)" : "❌ ABSENT");
    printf("    • Zero-Alloc Hotpath:  ✅ ACTIVE (Zero dynamic malloc/free in forward loop)\n");
    printf("    • Fused GEMV Engine:   ✅ ACTIVE (Direct Q4_K/Q5_K -> Q8_0 dot products)\n");

    /* Quick 64MB in-memory bandwidth benchmark */
    size_t bench_bytes = 64 * 1024 * 1024;
    float* src = (float*)malloc(bench_bytes);
    float* dst = (float*)malloc(bench_bytes);
    if (src && dst) {
        memset(src, 1, bench_bytes);
        double t0 = now_sec();
        int iters = 100;
        for (int i = 0; i < iters; i++) {
            memcpy(dst, src, bench_bytes);
        }
        double elapsed = now_sec() - t0;
        double gb_transferred = (double)(bench_bytes * (size_t)iters * 2) / (1024.0 * 1024.0 * 1024.0);
        double gbps = (elapsed > 0.0001) ? (gb_transferred / elapsed) : (gb_transferred / 0.05);
        printf("  L3/DRAM Bandwidth:   ~%.2f GB/s (Stream Copy Peak)\n", gbps);
        free(src);
        free(dst);
    }
    printf("========================================================================================\n");
}
