# NNlib vs PyTorch conv on CPU (issue #234)

Compares forward `conv` performance between NNlib and PyTorch on CPU,
reproducing [FluxML/NNlib.jl#234](https://github.com/FluxML/NNlib.jl/issues/234).
Both run with **4 threads** and `Float32`.

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

## Results

Median forward time, 4 threads, `Float32`. Measured on a 64-core box; because
the machine was shared, the **minimum** across repeated runs is reported as the
best estimate of uncontended time (contention only adds time). Numbers below
were stable across 3 consecutive runs.

| case               | shape (in→out, k, s, p, HxW, batch) | PyTorch (ms) | NNlib (ms) | NNlib / PyTorch |
|--------------------|-------------------------------------|-------------:|-----------:|----------------:|
| issue234 7x7 s2 p3 | 3→64, 7, s2, p3, 224x224, b2        | ~5.0         | ~10.5      | **2.1x**        |
| 3x3 s1 p1 c64      | 64→64, 3, s1, p1, 56x56, b2         | ~2.5         | ~8.4       | **3.4x**        |
| 3x3 s1 p1 c128     | 128→128, 3, s1, p1, 28x28, b2       | ~2.5         | ~6.9       | **2.8x**        |
| 1x1 s1 p0 c256     | 256→256, 1, s1, p0, 14x14, b2       | ~0.48        | ~0.61      | **1.3x**        |

### Takeaways

- The original issue's 7x7/stride-2 case reproduces: NNlib is **~2x slower**.
- The gap is **larger (~3x) for 3x3 convs**, the most common conv shape.
- The gap shrinks for 1x1 convs, which reduce to a single GEMM (BLAS-bound) —
  consistent with the bottleneck being NNlib's im2col + scattered work around
  the matmul rather than the matmul itself.
- NNlib also allocates a lot (e.g. ~34 MiB for the 7x7 case) due to im2col
  buffers, whereas PyTorch's CPU conv keeps overhead low.

Layout note: NNlib uses WHCN; PyTorch uses NCHW. The cases are defined with
matching channel/spatial/batch sizes so the FLOP counts are identical.

## Root-cause analysis

The gap is **not** in the im2col + BLAS algorithm itself. It comes from
**where the two libraries place their threading**:

- **NNlib parallelizes only over the batch dimension.** In
  [`src/impl/conv_im2col.jl`](../../src/impl/conv_im2col.jl) the work is split by
  partitioning the batch axis (dim 5) and spawning one task per batch chunk
  (`Iterators.partition(axes(x, 5), ...)` → `Threads.@spawn conv_part(...)`).
  Each task runs a single-threaded `im2col!` followed by a `gemm!`. When
  `batch < nthreads`, the surplus threads sit idle.
- **PyTorch (oneDNN) parallelizes inside a single image**, across output
  spatial tiles / channels, so it saturates all cores at any batch size.

### Evidence: per-image time (ms/img), 7x7 s2 p3 case

|                       | batch=1 | batch=2 | batch=4 | batch=8 |
|-----------------------|--------:|--------:|--------:|--------:|
| NNlib, 1 thread       | 7.23    | 7.43    | 7.38    | 7.50    |
| NNlib, 2 threads      | 7.36    | 6.65    | 5.59    | 3.66    |
| NNlib, 4 threads      | 6.46    | 4.76    | 2.76    | **2.39**|
| PyTorch, 4 threads    | 2.58    | 2.46    | 2.51    | **2.41**|

Two things stand out:

1. **NNlib's per-image time only improves as the batch grows; PyTorch's is
   flat** (~2.5 ms/img regardless of batch). That is the signature of
   batch-only parallelism vs. sub-image parallelism.
2. **Once `batch ≥ nthreads`, NNlib matches PyTorch exactly** (2.39 vs
   2.41 ms/img at batch 8). NNlib's compute throughput is identical to
   PyTorch when the cores are saturated.

So the ~2x slowdown reported in the issue exists because the benchmark uses
**batch=2 with 4 threads**: NNlib keeps 2 cores busy, PyTorch keeps all 4.

### Secondary factor (batch=1)

At batch=1 NNlib does not spawn at all — it runs one `im2col!` + one BLAS
`sgemm`. With 4 threads it's only marginally faster than single-threaded
(6.46 vs 7.23 ms) because the `im2col!` patch copy is serial and memory-bound,
and the resulting GEMM is *skinny* (M=12544, K=147, N=64), so BLAS
multithreading buys little. PyTorch's blocked/direct conv tiles the output
across all cores and never materializes the full im2col buffer.

### Fix direction

Parallelize **within** a single image — partition the `im2col!` + `gemm!` over
output spatial blocks (or output channels) across threads, instead of (or in
addition to) the batch axis. That would close the small-batch gap.
