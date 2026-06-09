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

## Why is NNlib slower?

The gap is **not** in the im2col+GEMM math — it's about **threading granularity**
and the backend PyTorch dispatches to.

### Root cause: NNlib threads only over the batch dimension

In [`src/impl/conv_im2col.jl`](../../src/impl/conv_im2col.jl) the work is split by
partitioning the batch axis and spawning one task per batch chunk; each task runs a
single-threaded `im2col!` + `gemm!`. When `batch < nthreads`, the extra threads sit
idle. PyTorch parallelizes *inside* a single image, so it saturates all cores at any
batch size. Per-image time (ms/img) for the 7x7 case makes this clear:

|                    | batch=1 | batch=2 | batch=4 | batch=8 |
|--------------------|--------:|--------:|--------:|--------:|
| NNlib, 4 threads   | 6.46    | 4.76    | 2.76    | **2.39**|
| PyTorch, 4 threads | 2.58    | 2.46    | 2.51    | **2.41**|

NNlib only improves as the batch grows; PyTorch is flat. **Once `batch ≥ nthreads`
the two match exactly** (2.39 vs 2.41) — NNlib's throughput is fine when cores are
saturated. The issue's 2x gap is because it uses batch=2 with ≥4 threads.

### What PyTorch dispatches to: oneDNN

PyTorch CPU conv ([`aten/.../Convolution.cpp`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Convolution.cpp),
`select_conv_backend`) sends float32 contiguous convs (batch>1 or threads>1) to
**oneDNN** (`aten::mkldnn_convolution`, confirmed via the profiler). The reference
`Slow2d`/`thnn` kernel (im2col+GEMM, closest to NNlib) is only reached if oneDNN is
disabled or the dtype/shape disqualifies it (e.g. f64, or bf16 without AVX512-BF16).

### How oneDNN's direct conv works

oneDNN uses a **JIT-generated direct convolution** (no full im2col buffer): activations
are reordered into a channel-blocked layout (`nChw16c` on AVX512, **`nChw8c` on this
AVX2 box**), weights into `OIhw8i8o`, and the inner loop keeps an output tile in vector
registers, accumulating with broadcast+FMA. It parallelizes over
batch × output-channel-blocks × spatial rows — hence full-core utilization on a single
image. (Winograd is x86-unsupported in current oneDNN; an implicit-GEMM path is the
fallback.)

### Fix direction for NNlib

Parallelize **within** an image — partition `im2col!` + `gemm!` over output spatial
blocks or output channels across threads, instead of only over the batch axis.
