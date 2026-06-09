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
