# Architecture Notes

## Inference Pipeline

1. Embed token id into `x`.
2. For each layer:
   - RMSNorm -> `xb`
   - QKV projections
   - RoPE on `q/k`
   - append `k/v` to KV cache
   - attention score, softmax, weighted value sum
   - output projection + residual
   - FFN pre-norm
   - SwiGLU (`w1`, `w3`, SiLU gate)
   - `w2` projection + residual
3. Final RMSNorm and classifier projection to logits.

## KV Cache Visualization

For each layer `l`:

- `key_cache[l][time][dim]`
- `value_cache[l][time][dim]`

At time `t`, each head attends over `[0..t]`.

## RoPE Summary

RoPE rotates each pair `(x[2i], x[2i+1])` by angle `theta = pos * base^{-2i/head_dim}`.
This encodes relative position while preserving dot-product structure.

## Performance Table Template

| backend | threads | tokens | elapsed(s) | tok/s | mem(MB) |
|---|---:|---:|---:|---:|---:|
| ref | 1 | 256 | - | - | - |
| avx2 | 8 | 256 | - | - | - |
| avx512 | 8 | 256 | - | - | - |
| int8 | 8 | 256 | - | - | - |
| q4k | 8 | 256 | - | - | - |
