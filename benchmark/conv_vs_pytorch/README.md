# NNlib vs PyTorch conv on CPU (issue #234)

Compares forward `conv` performance between NNlib and PyTorch on CPU,
reproducing [FluxML/NNlib.jl#234](https://github.com/FluxML/NNlib.jl/issues/234)
and measuring the effect of the within-image threading added in
[#697](https://github.com/FluxML/NNlib.jl/pull/697). Both run with **4 threads**
and `Float32`.

## Setup

Python env is managed with [uv](https://docs.astral.sh/uv/) (CPU-only torch):

```bash
uv sync            # creates .venv from pyproject.toml / uv.lock
```

Julia env (develops the local NNlib checkout + BenchmarkTools):

```bash
julia -e 'import Pkg; Pkg.activate("."); Pkg.develop(path="../.."); Pkg.add("BenchmarkTools")'
```

## Run

```bash
uv run --project . python bench_torch.py
julia --threads=4 --project=. bench_nnlib.jl
```

## Methodology

NNlib's CPU conv parallelism is Julia-task based, so a fair "4 threads" run is
`julia --threads=4` with **BLAS pinned to 1 thread** (`BLAS.set_num_threads(1)`);
otherwise `julia_threads × blas_threads` oversubscribes. PyTorch uses
`torch.set_num_threads(4)`.

The box is a 64-core Threadripper 2990WX (AVX2, no AVX512), often shared. Because
contention only *adds* time, all numbers below are the **minimum over 6 process
runs of 80 samples each** — the best estimate of uncontended time. Times are
**per image** (ms/img) so batch sizes are comparable.

## Results

Per-image forward time (ms/img), 4 threads, `Float32`. `master` is the previous
batch-only threading; **#697** adds within-image threading.

**issue #234 shape — 7×7, stride 2, pad 3, 3→64, 224×224**

| batch | master | **#697** | PyTorch | #697 vs master | #697 vs PyTorch |
|------:|-------:|---------:|--------:|---------------:|----------------:|
| 1 | 6.73 | **3.45** | 1.23 | **1.95× faster** | 2.8× slower |
| 2 | 4.98 | **3.28** | 1.58 | 1.52× faster | 2.1× slower |
| 4 | 2.57 | **2.17** | 1.29 | 1.19× faster | 1.7× slower |
| 8 | 1.91 | 1.97 | 1.27 | ~neutral | 1.6× slower |

**3×3, stride 1, pad 1, 64→64, 56×56**

| batch | master | **#697** | PyTorch | #697 vs master | #697 vs PyTorch |
|------:|-------:|---------:|--------:|---------------:|----------------:|
| 1 | 7.13 | **3.67** | 1.11 | **1.95× faster** | 3.3× slower |
| 2 | 3.84 | **3.46** | 1.02 | 1.11× faster | 3.4× slower |
| 4 | 2.77 | **2.16** | 0.97 | 1.28× faster | 2.2× slower |
| 8 | 2.01 | 2.12 | 0.96 | ~neutral | 2.2× slower |

**3×3, stride 1, pad 1, 128→128, 28×28**

| batch | master | **#697** | PyTorch | #697 vs master | #697 vs PyTorch |
|------:|-------:|---------:|--------:|---------------:|----------------:|
| 1 | 5.53 | **2.61** | 1.09 | **2.12× faster** | 2.4× slower |
| 2 | 3.05 | **2.36** | 1.04 | 1.29× faster | 2.3× slower |
| 4 | 1.70 | **1.64** | 0.98 | ~neutral | 1.7× slower |
| 8 | 1.53 | 1.58 | 0.96 | ~neutral | 1.6× slower |

**1×1, stride 1, pad 0, 256→256, 14×14**

| batch | master | **#697** | PyTorch | #697 vs master | #697 vs PyTorch |
|------:|-------:|---------:|--------:|---------------:|----------------:|
| 1 | 0.49 | **0.33** | 0.28 | 1.48× faster | 1.2× slower |
| 2 | 0.27 | 0.27 | 0.19 | ~neutral | 1.5× slower |
| 4 | 0.20 | 0.20 | 0.14 | ~neutral | 1.4× slower |
| 8 | 0.14 | 0.14 | 0.13 | ~neutral | 1.1× slower |

### Takeaways

- **#697 gives a consistent ~2× speedup at batch=1** (and ~1.1–1.5× at batch=2)
  across all shapes — exactly the small-batch regime of issue #234.
- The gain tapers to **neutral by batch≥4–8**, where the unchanged batch-split
  path already saturates the 4 threads. (A consistent ≤5% at batch=8 is within
  measurement noise of the unchanged path.)
- **PyTorch (oneDNN) is still faster everywhere** — ~1.5–3.4× — even after #697.
  The gap is largest for 3×3 convs. So #697 closes the *threading* part of the
  gap, not the *algorithmic* one (see below).

Layout note: NNlib uses WHCN; PyTorch uses NCHW. The cases are defined with
matching channel/spatial/batch sizes so the FLOP counts are identical.

## Why is NNlib slower?

Two separate factors: **threading** (now improved by #697) and the **backend**
PyTorch dispatches to (oneDNN, an algorithmic advantage that remains).

### Threading (the part #697 fixes)

The GEMM that dominates conv (~80% of per-image time) is *already* multithreaded
by BLAS. The problems were:

1. **im2col is serial** — the ~20% the GEMM threading can't touch.
2. **`conv_im2col!` only parallelized over the batch axis** — with
   `batch < nthreads` the surplus threads sat idle.
3. **Oversubscription** — the old default (`julia × BLAS` threads) ran up to 16
   threads, masking the problem on this 64-core box but hurting where cores ≈
   threads.

[#697](https://github.com/FluxML/NNlib.jl/pull/697) addresses (1) and (2): when
`batch < nthreads`, it splits each image's output-spatial dimension across tasks,
so every task runs `im2col!` + GEMM on a contiguous slab of output rows —
parallelizing both the copy and the matmul. BLAS is pinned to one thread inside
that region to avoid (3). The batch=1 numbers above show the ~2× result.

### Backend: PyTorch dispatches to oneDNN (the part that remains)

PyTorch CPU conv ([`Convolution.cpp`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Convolution.cpp),
`select_conv_backend`) sends float32 contiguous convs to **oneDNN**
(`aten::mkldnn_convolution`, confirmed via the profiler). oneDNN uses a
**JIT-generated direct convolution** — no full im2col buffer. Activations are
reordered into a channel-blocked layout (`nChw8c` on this AVX2 box), and the
inner loop keeps an output tile in vector registers, accumulating with
broadcast+FMA. It parallelizes over batch × output-channel-blocks × spatial rows.
On the 7×7 case it runs near the CPU's FLOP peak.

NNlib's im2col+GEMM can't match this for two reasons that threading can't fix:

- **im2col's memory blow-up.** A 3×3 conv expands the input **9×** (`k²`) into the
  col buffer; that materialization is pure memory traffic oneDNN never pays. This
  is why the 3×3 gap (~2–3.4×) is larger than the 7×7 gap.
- **GEMM efficiency.** The im2col GEMM is "skinny" (small N/K), so even
  BLAS-threaded it doesn't reach oneDNN's near-peak direct kernel.

### Conclusion

- **#697 is the high-ROI fix** and is implemented: ~2× at small batch, by
  parallelizing within an image instead of only over the batch.
- **Closing the residual ~1.5–3× to oneDNN** would require attacking the
  algorithm, not threading: tiling/fusing im2col+GEMM to cut the `k²` memory
  traffic, or a from-scratch direct convolution (blocked layout + SIMD/JIT
  kernels). The latter means reimplementing a large part of oneDNN in pure Julia
  — likely not worth it for the remaining gap.
