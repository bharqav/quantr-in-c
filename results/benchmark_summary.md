# MLSys Benchmark Results: Qwen 3 30B MoE (A3B)

- **Model**: `D:\qwen3-30b-a3b-Q5_K_M.gguf` (21.72 GB, 48 Layers, 128 Experts)
- **Quantization**: Q5_K_M Mixed Block Precision
- **Memory Management**: Zero-Allocation Win32 Virtual Mapping + Dynamic Working Set Governor

| Threads | Latency (ms/token) | Throughput (tok/sec) | Peak RSS Memory (MB) | Multi-Thread Speedup |
| :---: | :---: | :---: | :---: | :---: |
| **2** | 29015.76 ms | 0.03 tok/s | **1845.96 MB** | **1.00x** |
| **4** | 16733.17 ms | 0.06 tok/s | **1846.08 MB** | **1.73x** |
| **8** | 12021.55 ms | 0.08 tok/s | **1846.14 MB** | **2.41x** |
