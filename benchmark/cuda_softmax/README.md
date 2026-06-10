# GPU softmax/logsoftmax: cuDNN vs custom kernels (issue #513)

Benchmarks `softmax`/`logsoftmax` on the GPU, comparing the specialized **cuDNN**
routines against NNlib's **custom** generic kernels (the ones used on the CPU):

- **forward** — cuDNN `cudnnSoftmaxForward!` vs the `exp.(x .- max)/sum` kernel
- **backward** — cuDNN `cudnnSoftmaxBackward` vs the broadcast ∇ rule
  (`_∇softmax!`/`_∇logsoftmax!`)

for both `softmax` and `logsoftmax`, and for `dims=1` and `dims=2`. Reproduces
and extends [FluxML/NNlib.jl#513](https://github.com/FluxML/NNlib.jl/issues/513).

cuDNN is measured in **both of its softmax modes**:

- **CHANNEL** — reduces over the C axis. What NNlib used unconditionally before
  this PR. Pathologically slow on the **backward** pass when the softmax axis is
  long (the #513 bug).
- **INSTANCE** — reduces over C·H·W per sample. Equivalent to CHANNEL *only* when
  the softmax axis is the leading contiguous dimension (`stride == 1`, i.e.
  `dims=1` or a `Colon`); otherwise marked "—". Much faster than CHANNEL on the
  backward pass — and the basis for the fix.

A third contender, [**LogExpFunctions**](https://github.com/JuliaStats/LogExpFunctions.jl)
`softmax` (forward `softmax!` + its ChainRules gradient), is included for
`softmax` only — it has no `logsoftmax`. Its math is identical to NNlib's custom
kernels, so it serves as a cross-check / second data point on the same approach.

## The fix

Based on these results, the cuDNN extension now:

- uses cuDNN **INSTANCE** mode (not CHANNEL) when the softmax dims are a leading
  contiguous block (`dims=1`, leading ranges, or `Colon`) — fast forward *and*
  backward;
- uses the **custom broadcast kernels** for every other `dims` (a non-leading
  axis, e.g. `dims≥2`, or non-contiguous dims), where cuDNN's only option is the
  slow CHANNEL mode. (Permuting the array to make the axis leading and reusing
  INSTANCE was tried and is slower than the broadcast — see [below](#why-not-permute-for-dims2).)

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
  in the script, with an explicit `mode`), not via NNlib's `∇softmax!`, so each
  cuDNN mode is measured on every shape regardless of what the
  ([`softmaxdims`](../../ext/NNlibCUDACUDNNExt/softmax.jl)) dispatch would pick.
- Each call is timed with `@belapsed CUDA.@sync ...` (minimum, GPU synchronized).
  All paths are verified numerically equal before timing (max |Δ| ≲ 1e-6).
- cuDNN softmax always uses the *accurate* algorithm (per the
  [#506 fix](https://github.com/FluxML/NNlib.jl/issues/506)).

Numbers below: **TITAN RTX**, CUDA driver 12.5, cuDNN 9.2. Times in **µs** (lower
is better). Columns: **cuDNN-CH** (CHANNEL), **cuDNN-IN** (INSTANCE; "—" where
`stride>1`, not correctness-preserving), **NNlib** (custom), **LEF**
(LogExpFunctions, softmax only). `CH/IN` is the CHANNEL/INSTANCE ratio (>1 ⇒
INSTANCE faster). LEF tracks NNlib closely (same math).

## Results — `dims=1` (softmax axis = `size[1]`, contiguous → INSTANCE applies)

**FORWARD softmax**

| size | cuDNN-CH | cuDNN-IN | NNlib | LEF | CH/IN |
|---|---:|---:|---:|---:|---:|
| (256, 10, 32) | 9.6 | 9.6 | 43.0 | 42.7 | 1.00× |
| (256, 1000) | 10.5 | 10.5 | 44.5 | 44.3 | 1.00× |
| (1000, 1000) | 26.2 | 21.0 | 126.7 | 126.8 | 1.25× |
| (100, 10000) | 19.0 | 18.9 | 131.7 | 132.1 | 1.01× |
| (10000, 100) | 36.3 | 19.7 | 86.5 | 86.2 | 1.84× |
| (32000, 64) | 100.9 | 39.1 | 161.0 | 160.8 | 2.58× |
| (1000, 128) | 10.1 | 9.5 | 36.6 | 38.6 | 1.06× |

**BACKWARD softmax** — the #513 pathology, and the headline INSTANCE win:

| size | cuDNN-CH | cuDNN-IN | NNlib | LEF | CH/IN |
|---|---:|---:|---:|---:|---:|
| (256, 10, 32) | 10.1 | 10.2 | 39.3 | 40.0 | 1.00× |
| (256, 1000) | 11.0 | 11.1 | 39.9 | 39.7 | 0.99× |
| (1000, 1000) | 1706.3 | 31.8 | 101.3 | 100.2 | **53.6×** |
| (100, 10000) | 28.4 | 28.2 | 101.6 | 100.6 | 1.01× |
| (10000, 100) | 999.4 | 46.9 | 81.7 | 79.7 | **21.3×** |
| (32000, 64) | 1650.0 | 89.3 | 136.4 | 136.3 | **18.5×** |
| (1000, 128) | 174.5 | 10.3 | 33.4 | 31.9 | **16.9×** |

**BACKWARD logsoftmax** (CH/IN, worst case): `(1000,1000)` 3108→34 µs (**92×**),
`(10000,100)` 1549→40 (**39×**), `(32000,64)` 2676→78 (**34×**). Forward
logsoftmax mirrors softmax (INSTANCE up to 2.2× faster than CHANNEL).

→ INSTANCE matches CHANNEL where CHANNEL was already fine and is **17–92× faster**
where it was pathological. It also beats the custom (NNlib/LEF) kernel on the
backward pass, so for `dims=1` cuDNN-INSTANCE is the best of all three.

## Results — `dims=2` (softmax axis = `size[2]`, strided → INSTANCE not applicable)

INSTANCE is not correctness-preserving here (`stride>1`), so the choice is
cuDNN-CHANNEL vs the custom kernel.

**BACKWARD softmax**

| size | cuDNN-CH | NNlib | LEF | CH/NNlib |
|---|---:|---:|---:|---:|
| (256, 10, 32) | 10.3 | 41.3 | 42.8 | 0.25× |
| (256, 1000) | 185.1 | 41.5 | 41.3 | 4.46× |
| (1000, 1000) | 351.6 | 107.3 | 106.1 | 3.28× |
| (100, 10000) | 3088.7 | 87.8 | 87.7 | **35.2×** |
| (10000, 100) | 61.0 | 103.5 | 101.9 | 0.59× |
| (32000, 64) | 77.5 | 177.4 | 179.8 | 0.44× |
| (1000, 128) | 32.4 | 33.4 | 32.9 | 0.97× |

**BACKWARD logsoftmax** peaks at `(100,10000)` 3377→59 µs (**58×**) for custom;
**FORWARD** is mixed — CHANNEL wins for a short softmax axis (`(32000,64)`,
`(10000,100)`), the custom kernel wins for a long one (`(1000,1000)`,
`(100,10000)`).

## Takeaways

- **The crossover is the length of the softmax axis** (`dimsize`), independent of
  `dims` or batch size: short axis → cuDNN, long axis → custom, and the gap on the
  backward pass is large (up to ~50× for softmax, ~90× for logsoftmax).

- **For `dims=1` / `Colon` (contiguous, `stride==1`): cuDNN INSTANCE mode wins
  outright.** It is identical to CHANNEL numerically, never slower, and **17–92×
  faster** on the previously-pathological backward shapes — beating even the
  custom kernel. This is the main fix.

- **For `dims≥2` (strided, `stride>1`): the custom kernel is the right backward
  choice.** It is 3–58× faster than CHANNEL when the softmax axis is long. The
  tradeoff is a *short* non-leading axis (`(10000,100)`, `(32000,64)`), where
  CHANNEL was already efficient and the custom kernel is ~1.5–2× slower — an
  acceptable price to remove the catastrophic cases.

- **LogExpFunctions ≈ NNlib custom** (same `exp/sum` forward and
  `y .* (dy .- sum(dy.*y))` ∇ rule), to within measurement noise — the *same
  approach*, not a faster one, and softmax only (no `logsoftmax`).

## Why not permute for `dims≥2`?

A tempting alternative for `dims≥2`: `permutedims` to bring the softmax axis to the
front (making it contiguous), reuse the fast INSTANCE path, then permute back.
Measured (permute cost included), it is **correct but slower than the custom
kernel** — the backward needs 3 input transposes + 1 output transpose, whose
memory traffic outweighs the gain:

| `dims=2` backward softmax, µs | CHANNEL | PERMUTE+INSTANCE | custom |
|---|---:|---:|---:|
| (1000, 1000) | 348 | 195 | **108** |
| (100, 10000) | 3091 | 209 | **90** |
| (10000, 100) | **61** | 193 | 102 |

PERMUTE beats the slow CHANNEL but never beats the custom kernel, so the extension
uses the custom kernel for `dims≥2`.

## The dispatch (after this PR)

[`softmaxdims`](../../ext/NNlibCUDACUDNNExt/softmax.jl) now returns a cuDNN reshape
**only when the softmax dims are a leading contiguous block** (`dims=1`, a leading
range, or `Colon`), and that path runs in INSTANCE mode. Every other `dims`
returns `nothing` and uses the custom kernels.
