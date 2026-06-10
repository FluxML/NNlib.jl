# GPU softmax/logsoftmax gradient benchmark (Zygote AD path)

Times the **end-to-end Zygote autodiff path** for `softmax`/`logsoftmax` on the
GPU — the forward and the reverse pass that Flux actually runs — rather than the
`∇softmax`/`∇logsoftmax` kernels called directly. The distinction is the whole
point: the backward pass is only as fast as whatever the `rrule` routes to.

For each shape and `dims` the script reports (times in µs, `Float32`):

- `fwd-sm` / `fwd-ls` — forward `softmax` / `logsoftmax`.
- `bwd-sm` / `bwd-ls` — the **Zygote pullback only** (`Zygote.pullback` then
  `back(dȳ)`), isolating the reverse pass.
- `grad-sm` / `grad-ls` — a full `Zygote.gradient(z -> sum(abs2, f(z; dims)), x)`
  (forward + reverse; dominated by the generic AD glue around a single op).

## Why the gradient path matters here

Before this PR the softmax `rrule` called the generic broadcast
`∇softmax_data(dy, y)`, so the cuDNN backward (and the
[#513](https://github.com/FluxML/NNlib.jl/issues/513) INSTANCE-mode fix) was
**never reached from AD** — every Zygote gradient used the broadcast kernel. This
PR routes the `rrule` through `∇softmax!`, which the cuDNN extension overloads, so
for a **leading contiguous** softmax axis (`dims=1` / `Colon`) the Zygote backward
now runs cuDNN **INSTANCE** mode. (For a non-leading axis, e.g. `dims≥2`, the
backward stays on the broadcast kernel — cuDNN's only option there is the slow
CHANNEL mode; see #513.)

## Run

```bash
julia --project=. bench_softmax.jl
# pin a GPU (UUID form is robust when other cards on the box are unavailable):
CUDA_VISIBLE_DEVICES=GPU-<uuid> julia --project=. bench_softmax.jl
```

Each call is timed with `@belapsed CUDA.@sync(...)` (minimum, GPU-synchronized).

## Results — TITAN RTX, cuDNN 9.2, Float32 (µs, lower is better)

**Before** = a few commits back, pre-#717 (CHANNEL forward; broadcast AD
backward). **After** = this PR (INSTANCE forward for `dims=1`; AD backward routed
to cuDNN). The comparison therefore captures the combined effect of
[#717](https://github.com/FluxML/NNlib.jl/pull/717) and the gradient-API refactor.

### `dims=1` (leading contiguous axis → INSTANCE applies)

The headline: the Zygote **backward** now hits cuDNN INSTANCE instead of the
broadcast, **1.5–3.4× faster** (forward also improves, CHANNEL→INSTANCE).

| size | bwd-sm before | bwd-sm after | speedup | bwd-ls before | bwd-ls after | speedup |
|---|---:|---:|---:|---:|---:|---:|
| (256, 10, 32) | 35.6 | 12.5 | 2.8× | 27.2 | 11.4 | 2.4× |
| (256, 1000)   | 40.4 | 12.0 | 3.4× | 27.1 | 12.4 | 2.2× |
| (1000, 1000)  | 101.5 | 33.3 | 3.0× | 71.4 | 34.8 | 2.1× |
| (100, 10000)  | 103.0 | 30.2 | 3.4× | 71.5 | 30.2 | 2.4× |
| (10000, 100)  | 81.9 | 49.1 | 1.7× | 52.2 | 40.2 | 1.3× |
| (32000, 64)   | 141.6 | 91.9 | 1.5× | 92.5 | 80.2 | 1.2× |
| (1000, 128)   | 33.5 | 11.5 | 2.9× | 26.6 | 11.9 | 2.2× |

Forward `dims=1` (cuDNN, CHANNEL→INSTANCE): largest gains on a short softmax axis,
e.g. `(32000, 64)` 104.0 → 40.1 µs, `(10000, 100)` 39.1 → 22.0 µs; others within
noise.

### `dims=2` (strided axis → INSTANCE not applicable)

The Zygote **backward is unchanged** — it used the broadcast kernel before
(`∇softmax_data`) and after (cuDNN can't help a strided axis), so the numbers
match within noise:

| size | bwd-sm before | bwd-sm after | bwd-ls before | bwd-ls after |
|---|---:|---:|---:|---:|
| (256, 10, 32) | 42.9 | 40.7 | 30.5 | 31.1 |
| (256, 1000)   | 43.5 | 41.1 | 30.8 | 33.3 |
| (1000, 1000)  | 110.5 | 105.7 | 79.9 | 79.9 |
| (100, 10000)  | 90.7 | 86.1 | 59.1 | 59.7 |
| (10000, 100)  | 103.4 | 100.6 | 74.2 | 74.2 |
| (32000, 64)   | 179.5 | 173.9 | 125.6 | 125.4 |
| (1000, 128)   | 35.9 | 32.6 | 26.8 | 24.0 |

**Forward `dims=2` regresses** for a short softmax axis: before it went to cuDNN
CHANNEL, now it uses the broadcast, e.g. `(256, 10, 32)` 13.0 → 92.8 µs,
`(32000, 64)` 88.2 → 266.8 µs. This is the deliberate #513 tradeoff — CHANNEL is
catastrophic for a *long* `dims=2` axis (hundreds of µs to ms), so the extension
takes the broadcast for all `dims≥2`. It only shows up on the forward because the
`dims=2` backward already used the broadcast.

## Takeaways

- **`dims=1` / `Colon` gradients are the win**: routing the `rrule` through cuDNN
  INSTANCE makes the Zygote backward 1.5–3.4× faster, with no extra `x` captured
  by the pullback.
- **`dims≥2` gradients are unchanged** (broadcast before and after); only the
  `dims=2` *forward* trades cuDNN CHANNEL for the broadcast, the documented #513
  tradeoff.
- `grad-*` (full `Zygote.gradient`) is dominated by generic AD overhead around a
  single op (~120 µs floor here), so `bwd-*` is the cleaner signal for the kernel
  change.
