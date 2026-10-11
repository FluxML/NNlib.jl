# GPU multi-head attention benchmark — NNlib (cuDNN fused / materialized) vs LuxLib

Compares implementations of the same scaled dot-product (multi-head) attention on the
GPU, across input sizes from tiny up to a single Llama3-8B attention layer, timing both
the forward pass and the full **Zygote** gradient (forward + backward).

| Name | Function | Approach | Memory |
|---|---|---|---|
| `nnlib` | NNlib `scaled_dot_product_attention` | for `Float16`/`BFloat16`, cuDNN's fused kernel: the scores are never materialized. Other eltypes take the materialized path below | O(L) |
| `nnlib-mat` | the same function, kept on its generic path | materialized: builds the full score tensor, `softmax`, second `batched_mul` | O(L²) |
| `lux` | [LuxLib](https://github.com/LuxDL/Lux.jl/tree/main/lib/LuxLib) `scaled_dot_product_attention` | materialized softmax path | O(L²) |

All of them compute the identical math (scale `1/√E`) on the same
`(head_dim, nheads, seq, batch) = (E, H, L, B)` arrays.

`nnlib-mat` passes a no-op `fdrop` that is not `identity`: NNlib only takes the fused
kernel without dropout, bias and mask, so this keeps it on the generic path at no extra
cost.

## Run

Whole suite (all dtypes × both causal modes × all sizes × all impls):

```bash
julia --project=. bench_attention.jl
```

The environment uses the NNlib of this checkout (`[sources]` in `Project.toml`). The
script takes CLI flags so a single combination can be measured without rerunning the
whole suite:

```bash
# fused vs materialized, Float16, non-causal:
julia --project=. bench_attention.jl --dtypes f16 --impls nnlib,nnlib-mat --causal false
# the materialized paths in bf16+f16 on the small configs:
julia --project=. bench_attention.jl --dtypes f16,bf16 --impls nnlib-mat,lux --sizes tiny,small
# quick pass (shorter timing budget):
julia --project=. bench_attention.jl --seconds 0.3
julia --project=. bench_attention.jl --help     # all options
julia --project=. bench_attention.jl --list     # available configs / dtypes / impls
# pin a GPU:
CUDA_VISIBLE_DEVICES=GPU-<uuid> julia --project=. bench_attention.jl ...
```

| flag | values | default |
|---|---|---|
| `--dtypes` | `f16`,`bf16`,`f32` (aliases: `float16`/`fp16`/`half`, `bfloat16`, `float32`/`fp32`) | all |
| `--impls` | `nnlib`,`nnlib-mat`,`lux` | all |
| `--causal` | `false`,`true`, or `both` | both |
| `--sizes` | config names (see `--list`) | all |
| `--seconds` | per-measurement time budget | `1.0` |

### Timing

Each cell runs `CUDA.@sync(op)` under `BenchmarkTools.run(...; seconds, evals=1)`
and reports `minimum(trial)` in **ms**. `evals=1` is *per sample*, not total:
within the `seconds` budget BenchmarkTools collects as many samples as fit
(thousands for a sub-ms kernel) and we take the fastest — the standard robust
estimator for GPU kernels.

`n/s` = that path can't run the config (reason echoed once to stderr);
`OOM` = ran out of GPU memory; `<impl>/nnlib` = time relative to `nnlib`
(>1 → `nnlib` faster).

Configs (head_dim `E`, seq `L`, heads `H`, batch `B`); the last three are a
single Llama3-8B attention layer (`E=128, H=32`) at growing context:

```
tiny         E=64  L=128  H=4  B=8
small        E=64  L=512  H=8  B=8
gpt2-ish     E=64  L=1024 H=12 B=4
llama3 L=2k  E=128 L=2048 H=32 B=1
llama3 L=4k  E=128 L=4096 H=32 B=1
llama3 L=8k  E=128 L=8192 H=32 B=1
```

## Results — RTX 5090 (Blackwell, sm_120), CUDA.jl 6.4.2 / cuDNN.jl 6.4.2 (cuDNN 9.26), LuxLib 1.15.9

Raw output: [`results_rtx5090_cuda64.txt`](results_rtx5090_cuda64.txt).
Times in ms, lower is better. "fused" is `nnlib`, "materialized" is `nnlib-mat`, and
"speedup" is materialized ÷ fused.

### Float16

Non-causal:

| config | L | fwd: fused | materialized | LuxLib | speedup | grad: fused | materialized | LuxLib | speedup |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| tiny | 128 | 0.013 | 0.750 | 0.741 | 57.7× | 0.133 | 2.576 | 3.321 | 19.4× |
| small | 512 | 0.037 | 1.459 | 1.467 | 39.4× | 0.196 | 4.855 | 5.507 | 24.8× |
| gpt2-ish | 1024 | 0.118 | 1.214 | 1.116 | 10.3× | 0.455 | 3.835 | 4.581 | 8.4× |
| llama3 L=2k | 2048 | 0.410 | 1.532 | 1.537 | 3.7× | 1.565 | 4.516 | 4.629 | 2.9× |
| llama3 L=4k | 4096 | 1.413 | 4.172 | 4.192 | 3.0× | 5.367 | 11.681 | 11.602 | 2.2× |
| llama3 L=8k | 8192 | 5.188 | n/s | n/s | — | 19.485 | n/s | n/s | — |

Causal:

| config | L | fwd: fused | materialized | LuxLib | speedup | grad: fused | materialized | LuxLib | speedup |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| tiny | 128 | 0.014 | 0.804 | 0.766 | 57.4× | 0.147 | 2.772 | 3.741 | 18.9× |
| small | 512 | 0.038 | 1.631 | 1.630 | 42.9× | 0.180 | 6.047 | 6.225 | 33.6× |
| gpt2-ish | 1024 | 0.081 | 1.580 | 1.579 | 19.5× | 0.328 | 7.451 | 7.996 | 22.7× |
| llama3 L=2k | 2048 | 0.244 | 2.454 | 2.448 | 10.1× | 0.983 | 11.123 | 11.402 | 11.3× |
| llama3 L=4k | 4096 | 0.800 | 7.858 | 7.845 | 9.8× | 3.157 | 38.008 | 38.341 | 12.0× |
| llama3 L=8k | 8192 | 2.803 | n/s | n/s | — | 10.751 | n/s | n/s | — |

### BFloat16

Non-causal:

| config | L | fwd: fused | materialized | LuxLib | speedup | grad: fused | materialized | LuxLib | speedup |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| tiny | 128 | 0.013 | 0.764 | 0.782 | 58.8× | 0.143 | 2.639 | 3.275 | 18.5× |
| small | 512 | 0.036 | 1.632 | 1.508 | 45.3× | 0.204 | 4.822 | 6.115 | 23.6× |
| gpt2-ish | 1024 | 0.118 | 1.334 | 1.243 | 11.3× | 0.456 | 4.488 | 4.931 | 9.8× |
| llama3 L=2k | 2048 | 0.408 | 1.707 | 1.706 | 4.2× | 1.568 | 4.892 | 4.854 | 3.1× |
| llama3 L=4k | 4096 | 1.390 | 4.575 | 4.585 | 3.3× | 5.362 | 12.448 | 12.437 | 2.3× |
| llama3 L=8k | 8192 | 5.202 | n/s | n/s | — | 19.452 | n/s | n/s | — |

Causal:

| config | L | fwd: fused | materialized | LuxLib | speedup | grad: fused | materialized | LuxLib | speedup |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| tiny | 128 | 0.014 | 0.802 | 0.778 | 57.3× | 0.150 | 2.592 | 3.700 | 17.3× |
| small | 512 | 0.037 | 1.517 | 1.502 | 41.0× | 0.182 | 5.631 | 6.625 | 30.9× |
| gpt2-ish | 1024 | 0.080 | 1.656 | 1.619 | 20.7× | 0.331 | 7.580 | 8.085 | 22.9× |
| llama3 L=2k | 2048 | 0.243 | 2.607 | 2.605 | 10.7× | 0.985 | 11.469 | 11.758 | 11.6× |
| llama3 L=4k | 4096 | 0.795 | 8.255 | 8.217 | 10.4× | 3.156 | 38.808 | 38.718 | 12.3× |
| llama3 L=8k | 8192 | 2.773 | n/s | n/s | — | 10.804 | n/s | n/s | — |

### Float32

cuDNN's fused kernel only takes half precision, so `nnlib` and `nnlib-mat` are the same
materialized path here (their timings agree within noise).

Non-causal:

| config | L | fwd: NNlib | LuxLib | grad: NNlib | LuxLib |
|---|--:|--:|--:|--:|--:|
| tiny | 128 | 0.042 | 0.043 | 0.270 | 0.907 |
| small | 512 | 0.242 | 0.243 | 0.874 | 1.119 |
| gpt2-ish | 1024 | 0.751 | 0.754 | 2.153 | 2.141 |
| llama3 L=2k | 2048 | 2.350 | 2.364 | 6.725 | 6.674 |
| llama3 L=4k | 4096 | 8.762 | 8.806 | 24.874 | 24.826 |
| llama3 L=8k | 8192 | n/s | n/s | n/s | n/s |

Causal:

| config | L | fwd: NNlib | LuxLib | grad: NNlib | LuxLib |
|---|--:|--:|--:|--:|--:|
| tiny | 128 | 0.057 | 0.051 | 0.392 | 1.156 |
| small | 512 | 0.379 | 0.373 | 2.272 | 2.601 |
| gpt2-ish | 1024 | 1.228 | 1.227 | 6.180 | 6.233 |
| llama3 L=2k | 2048 | 3.327 | 3.335 | 17.269 | 17.247 |
| llama3 L=4k | 4096 | 12.658 | 12.586 | 72.847 | 67.975 |
| llama3 L=8k | 8192 | n/s | n/s | OOM | OOM |

## Takeaways

- **The fused kernel is faster at every size**, in both half-precision types: 3–58×
  on the forward pass and 2–34× on the full gradient. The gain is largest at small
  sizes, where the materialized path is dominated by its fixed overhead (several
  `permutedims`, two `batched_mul` and a half-precision `softmax`), and for causal
  attention, where the materialized path additionally builds and applies a mask while
  the fused kernel does about half the work.

- **It is the only path that completes `L=8192`.** The materialized score tensor
  `(8192, 8192, 32)` has `2³¹` elements, which overflows cuDNN's `int32` softmax
  descriptor (`CUDNN_STATUS_NOT_SUPPORTED` → `n/s`), and in `Float32` it exhausts the
  32 GB card on the gradient (`OOM`).

- **`Float16` and `BFloat16` cost the same**, on both paths.

- **`Float32` has no fused path**, and is slower than fused half precision from
  `gpt2-ish` up (6× on the `llama3 L=4k` forward).

- **NNlib ≈ LuxLib on the materialized path** — the same algorithm, within noise.
  LuxLib carries more autodiff overhead at small sizes (Float32 gradient `tiny`:
  0.91 vs 0.27 ms).
