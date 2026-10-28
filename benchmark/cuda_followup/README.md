# CUDA 6.5 follow-up

Reproduces the remaining performance checks from
[the CUDA 6.5 plan](https://github.com/FluxML/NNlib.jl/issues/796#issuecomment-6437990327)
after [#875](https://github.com/FluxML/NNlib.jl/pull/875).
The cuDNN compatibility floor and phase 3 cleanup are unchanged.

Run from this directory:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. bench_followup.jl
julia --project=. bench_followup.jl reductions
julia --project=. bench_followup.jl batched_mul
julia --project=. bench_followup.jl attention
```

`BENCH_SECONDS` sets the sampling time per implementation (default 0.2 seconds).
Each row checks agreement before timing and reports the minimum synchronized GPU
time, including output allocation where the operation allocates its result.
Run with no other GPU jobs active.

- `reductions` compares materialized and lazy `logsumexp`, including the vocabulary
  logits and trailing-dimension cases of #827. It also compares the squared and
  centered-squared broadcasts that `_broadcast_sum` reduced in #830. Fused normalization
  kernels have since replaced that helper, so those rows measure the original
  expressions rather than the current normalization dispatch.
- `batched_mul` compares the former per-slice `mul!` loop with a single batched cuBLAS
  call for Float16 and BFloat16, with the same preallocated output.
- `attention` compares NNlib's fused kernel with the new cuDNN dispatch for decoding,
  chunked prefill, sliding windows and packed sequences. The mask is removed for a
  single bottom-right-aligned query, which can attend to every valid key.

The environment pins AcceleratedKernels 0.5.10 to reproduce the upstream reduction
fixes. NNlib comes from this checkout; every other dependency is a registered release.

The `logsumexp` optimization uses `sum(_exp, broadcasted(-, x, max_); dims)` for
GPU arrays. Current CUDA, AMDGPU and Metal share AcceleratedKernels' reduction
implementation. These timings come from CUDA; the CPU method still uses Base.

## Reference run

2026-10-28, RTX 5090, Julia 1.13.2, CUDA.jl/cuDNN.jl 6.5.0, CUDA runtime 13.4,
cuDNN runtime 9.27, GPUArrays 12.1.1, KernelAbstractions 0.10.3 and
AcceleratedKernels 0.5.10. GPU test processes were paused during timing.

| Reduction | Materialized (µs) | Lazy (µs) | Speedup |
|---|---:|---:|---:|
| `logsumexp`, Float32, 50304 × 8192, `dims=1` | 4135.6 | 2053.5 | 2.01× |
| `logsumexp`, BFloat16, 50304 × 8192, `dims=1` | 3383.9 | 1416.8 | 2.39× |
| `logsumexp`, Float32, 8192 × 768, `dims=2` | 34.6 | 22.3 | 1.55× |
| `logsumexp`, BFloat16, 32000 × 1024, `dims=1` | 272.7 | 67.1 | 4.06× |

The sampled `logsumexp` cases improve by 1.49–4.06×. The squared and centered-squared
broadcast reductions improve by 1.39–2.43×, including trailing dimensions. The regressions
that previously motivated materializing them are absent on this stack. The current
normalization kernels already bypass those broadcasts, so they need no change.

| Batched multiplication | Per-slice (µs) | Batched cuBLAS (µs) | Speedup |
|---|---:|---:|---:|
| Float16, 32 × 32 × 32, batch 128 | 995.4 | 11.4 | 87.11× |
| BFloat16, 32 × 32 × 32, batch 128 | 1014.6 | 11.4 | 88.71× |
| Float16, 128 × 128 × 128, batch 32 | 245.0 | 12.7 | 19.35× |
| BFloat16, 64 × 512 × 64, batch 64 | 513.3 | 15.3 | 33.65× |

Attention uses Float16, head dimension 128, 32 query heads and 8 key/value heads.

| Attention | NNlib kernel (µs) | cuDNN dispatch (µs) | Speedup |
|---|---:|---:|---:|
| One query, 16384 cached keys, batch 1 | 3382.7 | 28.1 | 120.46× |
| One query, 16384 cached keys, batch 8 | 5584.5 | 335.7 | 16.64× |
| 512 queries, 4096 keys, bottom-right causal | 1640.7 | 218.3 | 7.52× |
| 8192 tokens, causal window 1024 | 5450.2 | 690.9 | 7.89× |
| 8 packed causal documents, 1024 tokens each | 3351.4 | 479.6 | 6.99× |

[Raw output](results_rtx5090_cuda65_ak0510.txt) includes every measured configuration.

## CPU comparison

`julia --project=. bench_cpu_logsumexp.jl` compares the current CPU method with
`sum(_exp, instantiate(broadcasted(-, x, max_)); dims, init=...)`. Base needs the
explicit axes and initial value; the GPU reduction does not.

Julia 1.13.2, one thread, Threadripper PRO 9955WX:

| Input and reduction | Current (µs) | Lazy (µs) |
|---|---:|---:|
| Float32, 1000 × 4096, `dims=1` | 1396.1 | 674.0 |
| Float32, 8192 × 768, `dims=2` | 2743.6 | 1947.4 |
| Float32, 64 × 64 × 32 × 16, `dims=(1,2)` | 532.5 | 389.2 |
| Float32, 1000 × 4096, `dims=:` | 1318.5 | 9484.9 |
| Float16, 1000 × 4096, `dims=1` | 1162.7 | 10032.1 |

The dimension-wise Float32 reductions improve, but the scalar and Float16 reductions
regress substantially. On Julia 1.10.12 the lazy Base reduction also fails with
`MethodError: LinearIndices(::Broadcasted)`. The CPU implementation stays unchanged.
[Raw CPU output](results_tr9955wx_cpu.txt).
