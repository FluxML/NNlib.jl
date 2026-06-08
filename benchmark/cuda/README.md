# CUDA activation broadcast benchmarks

Benchmarks accompanying
[PR #686](https://github.com/FluxML/NNlib.jl/pull/686), which removes the custom
cuDNN-routed broadcast overloads for the activations `relu`, `σ`, `elu` and
`tanh` on `CuArray`s.

Before the PR, `relu.(x::CuArray)` (and `materialize!`/in-place forms) were
pirated onto cuDNN's `cudnnActivationForward!`. The PR drops those overloads and
relies on CUDA.jl's native broadcast, because:

- the type piracy of `Base.materialize`/`materialize!` caused method
  invalidations and load-time latency (#504), and
- cuDNN does not propagate `NaN`s by default, so `relu.(cu([NaN]))` returned `0`
  instead of `NaN` (#509).

The throughput rationale was that these elementwise ops are memory-bandwidth
bound, so the native broadcast should be "just as fast". These benchmarks check
that claim.

## Running

```
julia --project=benchmark/cuda -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=benchmark/cuda benchmark/cuda/activations.jl
```

The script measures both paths in the same process, so the comparison is
apples-to-apples on the same machine/driver:

| column   | path                                   | corresponds to |
| -------- | -------------------------------------- | -------------- |
| `native` | `f.(x)` / `broadcast!(f, dst, x)`      | **post-PR**    |
| `cudnn`  | `cudnnActivationForward!(dst, x; mode)`| **pre-PR**     |

`ratio = native / cudnn`; a ratio `> 1` means the post-PR native path is slower.
Times are the minimum of 1000 GPU-synced samples, in microseconds.

## Results

See [`results.txt`](results.txt) for a full run. Machine used:
RTX 5090, CUDA 13.2, cuDNN 9.2, Julia 1.12.

Summary of what the numbers show:

- **Float32 / Float64, large (memory-bound) tensors** — native broadcast is
  competitive. For `elu`/`relu` in Float64 the native path is on par or *faster*
  than cuDNN (cuDNN's ELU/RELU kernels do extra work); for `tanh`/`σ` it is
  within a few percent. This matches the PR's "just as fast" claim.
- **Small tensors** — cuDNN has lower CPU-side launch overhead, so the native
  path is a few µs slower in absolute terms (launch-overhead dominated regime).
- **Float16** — native broadcast is markedly slower (up to ~5–6×). CUDA.jl's
  native broadcast does not vectorize Float16 (`half2`), so the Float16 native
  time is essentially identical to the Float32 native time (no bandwidth
  benefit), whereas cuDNN's Float16 kernel is faster than its Float32 one as
  expected for a bandwidth-bound op.
- **fast variants** (`tanh_fast`, `sigmoid_fast`; relu/elu have none) — these
  were never cuDNN-routed, so the PR doesn't change them. On GPU they are no
  faster than the plain native versions for Float16/Float32 (GPUs have
  hardware-fast transcendentals, so the approximations save nothing), so they do
  *not* close the Float16 gap. The exception is Float64 `tanh_fast`, ~10–14%
  faster than native `tanh` — enough to beat cuDNN.

**Takeaway:** the correctness (NaN propagation) and latency (invalidation)
arguments for the PR stand on their own. On pure throughput, native broadcast
is competitive for Float32/Float64 but leaves Float16 performance on the table.
If Float16 elementwise throughput becomes important, the fix belongs in CUDA.jl's
broadcast (Float16 vectorization), not in re-introducing the cuDNN piracy.
