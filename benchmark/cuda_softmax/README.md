# GPU softmax/logsoftmax: cuDNN vs custom kernels (issue #513)

Benchmarks `softmax`/`logsoftmax` on the GPU, comparing the specialized **cuDNN**
routines against NNlib's **custom** generic kernels (the ones used on the CPU):

- **forward** — cuDNN `cudnnSoftmaxForward!` vs the `exp.(x .- max)/sum` kernel
- **backward** — cuDNN `cudnnSoftmaxBackward` vs the broadcast ∇ rule
  (`_∇softmax!`/`_∇logsoftmax!`)

for both `softmax` and `logsoftmax`, and for `dims=1` and `dims=2`. Reproduces
and extends [FluxML/NNlib.jl#513](https://github.com/FluxML/NNlib.jl/issues/513).

A third contender, [**LogExpFunctions**](https://github.com/JuliaStats/LogExpFunctions.jl)
`softmax` (forward `softmax!` + its ChainRules gradient), is included for
`softmax` only — it has no `logsoftmax`. Its math is identical to NNlib's custom
kernels, so it serves as a cross-check / second data point on the same approach.

## Setup

```bash
julia -e 'import Pkg; Pkg.activate("."); Pkg.develop(path="../.."); Pkg.add(["CUDA","cuDNN","LogExpFunctions","BenchmarkTools"])'
```

## Run

```bash
julia --project=. bench_softmax.jl
```

Pin a GPU with `CUDA_VISIBLE_DEVICES` (UUID form is robust when other cards on the
box are unavailable):

```bash
CUDA_VISIBLE_DEVICES=GPU-<uuid> julia --project=. bench_softmax.jl
```

## Methodology

- `Float32`. `softmax` acts along axis `dims`; for `dims=1` the softmax-vector
  length is `size[1]`, for `dims=2` it is `size[2]`, and the remaining axes are
  batch.
- The **cuDNN routines are called directly** (`cudnn_softmax!`/`cudnn_∇softmax!`
  in the script), not via NNlib's `∇softmax!`. NNlib's `softmaxdims` heuristic
  ([`ext/NNlibCUDACUDNNExt/softmax.jl`](../../ext/NNlibCUDACUDNNExt/softmax.jl))
  diverts some shapes to the custom kernel, which would otherwise hide the
  comparison — here every shape really goes through cuDNN.
- Each call is timed with `@belapsed CUDA.@sync ...` (minimum, GPU synchronized).
  Both paths are verified numerically equal before timing (max |Δ| ≲ 1e-6).
- cuDNN softmax always uses the *accurate* algorithm (per the
  [#506 fix](https://github.com/FluxML/NNlib.jl/issues/506)).

Numbers below: **TITAN RTX**, CUDA driver 12.5, cuDNN 9.2, NNlib `master`.
Times in **µs** (lower is better). `cuDNN/NNlib > 1` means the custom kernel is
faster. `LEF` = LogExpFunctions; it tracks `NNlib` closely because the math is the
same (softmax only).

## Results — `dims=1`

**FORWARD softmax**

| size | cuDNN | NNlib | LEF | cuDNN/NNlib |
|---|---:|---:|---:|---:|
| (256, 10, 32) | 9.5 | 40.4 | 40.7 | 0.23× |
| (256, 1000) | 10.3 | 41.8 | 41.7 | 0.25× |
| (1000, 1000) | 26.2 | 125.1 | 125.0 | 0.21× |
| (100, 10000) | 19.1 | 130.1 | 130.4 | 0.15× |
| (10000, 100) | 36.6 | 85.5 | 85.3 | 0.43× |
| (32000, 64) | 100.8 | 159.7 | 160.5 | 0.63× |
| (1000, 128) | 10.4 | 36.4 | 36.3 | 0.29× |

**BACKWARD softmax**

| size | cuDNN | NNlib | LEF | cuDNN/NNlib |
|---|---:|---:|---:|---:|
| (256, 10, 32) | 9.9 | 37.5 | 37.6 | 0.26× |
| (256, 1000) | 10.8 | 38.3 | 38.3 | 0.28× |
| (1000, 1000) | 1699.1 | 100.7 | 100.2 | **16.87×** |
| (100, 10000) | 28.2 | 101.6 | 100.6 | 0.28× |
| (10000, 100) | 1000.1 | 80.8 | 79.6 | **12.39×** |
| (32000, 64) | 1652.7 | 137.5 | 137.0 | **12.02×** |
| (1000, 128) | 174.8 | 39.5 | 39.3 | 4.42× |

**logsoftmax** (cuDNN vs NNlib custom; LEF has none)

| size | fwd cuDNN | fwd NNlib | fwd ratio | bwd cuDNN | bwd NNlib | bwd ratio |
|---|---:|---:|---:|---:|---:|---:|
| (256, 10, 32) | 9.5 | 51.3 | 0.19× | 10.5 | 25.4 | 0.41× |
| (256, 1000) | 10.3 | 53.2 | 0.19× | 11.3 | 26.6 | 0.42× |
| (1000, 1000) | 28.2 | 147.1 | 0.19× | 3109.4 | 71.8 | **43.32×** |
| (100, 10000) | 19.0 | 152.5 | 0.12× | 28.4 | 72.9 | 0.39× |
| (10000, 100) | 30.7 | 107.5 | 0.29× | 1548.6 | 51.9 | **29.82×** |
| (32000, 64) | 85.7 | 195.9 | 0.44× | 2684.5 | 90.7 | **29.58×** |
| (1000, 128) | 9.8 | 47.9 | 0.21× | 273.1 | 23.2 | 11.79× |

## Results — `dims=2`

**FORWARD softmax**

| size | cuDNN | NNlib | LEF | cuDNN/NNlib |
|---|---:|---:|---:|---:|
| (256, 10, 32) | 9.9 | 51.2 | 50.8 | 0.19× |
| (256, 1000) | 38.1 | 50.7 | 50.6 | 0.75× |
| (1000, 1000) | 496.8 | 139.4 | 139.5 | **3.56×** |
| (100, 10000) | 121.3 | 97.7 | 97.3 | 1.24× |
| (10000, 100) | 73.5 | 130.6 | 130.5 | 0.56× |
| (32000, 64) | 85.1 | 239.4 | 239.9 | 0.36× |
| (1000, 128) | 48.7 | 40.9 | 39.2 | 1.19× |

**BACKWARD softmax**

| size | cuDNN | NNlib | LEF | cuDNN/NNlib |
|---|---:|---:|---:|---:|
| (256, 10, 32) | 10.6 | 41.7 | 41.8 | 0.25× |
| (256, 1000) | 185.3 | 41.9 | 41.7 | 4.42× |
| (1000, 1000) | 351.6 | 106.6 | 105.7 | 3.30× |
| (100, 10000) | 3102.5 | 86.9 | 86.1 | **35.69×** |
| (10000, 100) | 60.8 | 102.3 | 101.1 | 0.59× |
| (32000, 64) | 76.4 | 176.6 | 176.2 | 0.43× |
| (1000, 128) | 32.4 | 35.2 | 34.9 | 0.92× |

**logsoftmax** (cuDNN vs NNlib custom; LEF has none)

| size | fwd cuDNN | fwd NNlib | fwd ratio | bwd cuDNN | bwd NNlib | bwd ratio |
|---|---:|---:|---:|---:|---:|---:|
| (256, 10, 32) | 9.8 | 57.5 | 0.17× | 10.9 | 30.5 | 0.36× |
| (256, 1000) | 30.8 | 57.3 | 0.54× | 185.5 | 31.4 | 5.90× |
| (1000, 1000) | 478.5 | 162.6 | **2.94×** | 343.5 | 78.7 | 4.37× |
| (100, 10000) | 84.9 | 122.2 | 0.69× | 3121.5 | 58.8 | **53.12×** |
| (10000, 100) | 72.3 | 152.0 | 0.48× | 53.5 | 73.2 | 0.73× |
| (32000, 64) | 83.9 | 272.7 | 0.31× | 63.5 | 126.1 | 0.50× |
| (1000, 128) | 45.7 | 50.4 | 0.91× | 31.9 | 25.0 | 1.28× |

## Takeaways

- **Forward: cuDNN is the right default.** For `dims=1` it wins everywhere
  (2–7×). For `dims=2` it still wins when the softmax axis is short, and only
  loses (up to ~3.5×) when the softmax axis is *both long and strided*
  (e.g. `(1000,1000)`, `(100,10000)` softmax).

- **Backward: cuDNN collapses when the softmax axis is long.** The backward pass
  is the real problem from #513. cuDNN's CHANNEL-mode backward scales badly with
  the length of the softmax axis, while the custom broadcast rule stays roughly
  flat:
  - `dims=1`, long first axis: **12–17× slower (softmax)**, **30–43× slower
    (logsoftmax)** — e.g. `(1000,1000)`, `(10000,100)`, `(32000,64)`.
  - `dims=2`, long second axis: same pathology, peaking at **36× (softmax)** and
    **53× (logsoftmax)** for `(100,10000)`.
  - `logsoftmax` backward is consistently the worst case.

- **Rule of thumb:** the cuDNN-vs-custom crossover is governed by the **length of
  the softmax axis** (`dimsize`), independent of `dims` or batch size. Short axis
  → cuDNN; long axis → custom, and the gap on the backward pass is large.

- **LogExpFunctions ≈ NNlib custom.** LEF's `softmax` forward and gradient track
  NNlib's custom kernels to within measurement noise (same `exp/sum` forward and
  the same `y .* (dy .- sum(dy.*y))` ∇ rule), so it has the same large advantage
  over cuDNN on the long-axis backward pass. It offers no `logsoftmax` and no
  separate GPU path — it is the *same approach*, not a faster one.

## Implication for the dispatch heuristic

`softmaxdims` currently falls back to the custom kernel only when

```julia
batchsize == 1 && 64 <= stride <= 4096 && 64 <= dimsize <= 4096
```

None of the slow backward cases above qualify — they have `batchsize > 1` (e.g.
`(1000,1000)`/`dims=1` → `dimsize=1000, batchsize=1000`), so NNlib sends them to
the slow cuDNN backward. The fix is to route the **backward** pass to the custom
broadcast rule whenever the softmax axis (`dimsize`) is long, regardless of
`batchsize`, while keeping cuDNN for the **forward** pass.
