#!/usr/bin/env python3
"""
Baremetal AI Inference Engine - Automated MLSys Research Benchmark Suite
Executes multi-threaded scaling, memory-governor tracking, and generates publication tables.
"""

import os
import sys
import subprocess
import time
import json
import csv

def run_cmd(cmd):
    print(f"[RUN] {cmd}")
    t0 = time.perf_counter()
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    t1 = time.perf_counter()
    return res.returncode, res.stdout, res.stderr, (t1 - t0)

def main():
    model_path = r"D:\qwen3-30b-a3b-Q5_K_M.gguf"
    if not os.path.exists(model_path):
        for alt in [r"D:\qwen3-30b-a3b-Q6_K.gguf"]:
            if os.path.exists(alt):
                model_path = alt
                break

    print("=" * 75)
    print(f"Baremetal Engine MLSys Benchmark Suite: {model_path}")
    print("=" * 75)

    # 1. Build release binary with -lpsapi
    build_cmd = "gcc -O3 -Wall -Wextra -fopenmp -std=c99 main.c model.c kernels.c runtime.c sampling.c quant.c tokenizer.c benchmark.c threadpool.c gguf.c -o inference.exe -lm -lpsapi"
    rc, out, err, dt = run_cmd(build_cmd)
    if rc != 0:
        print(f"Build failed:\n{err}")
        return 1
    print(f"Build completed successfully in {dt:.2f}s\n")

    # 2. Thread scaling experiment
    thread_counts = [2, 4, 8]
    results = []

    os.makedirs("results", exist_ok=True)

    for th in thread_counts:
        csv_out = f"results/bench_th{th}.csv"
        json_out = f"results/bench_th{th}.json"
        cmd = f".\\inference.exe --model \"{model_path}\" --steps 2 --threads {th} --backend avx2 --benchmark --csv {csv_out} --json {json_out}"
        rc, out, err, duration = run_cmd(cmd)
        print(out.strip())
        if err and len(err.strip()) > 0:
            print(f"[STDERR] {err.strip()}")
        
        # Read JSON metrics
        if os.path.exists(json_out):
            try:
                with open(json_out, "r") as f:
                    data = json.load(f)
                    results.append(data)
            except Exception as e:
                print(f"Error reading {json_out}: {e}")

    # 3. Summary Table
    print("\n" + "=" * 75)
    print("EXPERIMENT SUMMARY: Multi-Threaded Scaling & Memory Footprint")
    print("=" * 75)
    print(f"{'Threads':<10} {'Latency (ms/tok)':<20} {'Tokens/Sec':<15} {'Peak RSS (MB)':<15} {'Speedup':<10}")
    print("-" * 75)
    base_tps = results[0].get("tokens_per_sec", 1.0) if results else 1.0
    for r in results:
        th = r.get("threads", 1)
        lat = r.get("mean_latency_ms", 0.0)
        tps = r.get("tokens_per_sec", 0.0)
        rss = r.get("peak_rss_mb", 0.0)
        speedup = tps / base_tps if base_tps > 0 else 1.0
        print(f"{th:<10} {lat:<20.2f} {tps:<15.2f} {rss:<15.2f} {speedup:<10.2f}x")
    print("=" * 75)

    # 4. Save consolidated markdown summary
    summary_md = os.path.join("results", "benchmark_summary.md")
    with open(summary_md, "w") as f:
        f.write("# MLSys Benchmark Results: Qwen 3 30B MoE (A3B)\n\n")
        f.write(f"- **Model**: `{model_path}` (21.72 GB, 48 Layers, 128 Experts)\n")
        f.write(f"- **Quantization**: Q5_K_M Mixed Block Precision\n")
        f.write(f"- **Memory Management**: Zero-Allocation Win32 Virtual Mapping + Dynamic Working Set Governor\n\n")
        f.write("| Threads | Latency (ms/token) | Throughput (tok/sec) | Peak RSS Memory (MB) | Multi-Thread Speedup |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: |\n")
        for r in results:
            th = r.get("threads", 1)
            lat = r.get("mean_latency_ms", 0.0)
            tps = r.get("tokens_per_sec", 0.0)
            rss = r.get("peak_rss_mb", 0.0)
            speedup = tps / base_tps if base_tps > 0 else 1.0
            f.write(f"| **{th}** | {lat:.2f} ms | {tps:.2f} tok/s | **{rss:.2f} MB** | **{speedup:.2f}x** |\n")
    
    print(f"\nBenchmark suite complete. Summary report saved to {summary_md}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
