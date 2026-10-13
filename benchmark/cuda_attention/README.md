# GPU multi-head attention benchmark — NNlib (fused / materialized), LuxLib, PyTorch

Compares implementations of the same scaled dot-product (multi-head) attention on the
GPU, across input sizes from tiny up to a single Llama3-8B attention layer: the time of
the forward pass and of the full **Zygote** gradient (forward + backward), and the GPU
memory each of the two allocates.

| Name | Function | Approach | Memory |
|---|---|---|---|
| `nnlib` | NNlib `scaled_dot_product_attention`, as it dispatches | cuDNN's fused kernel for `Float16`/`BFloat16`, NNlib's own fused kernel for the other eltypes: the weights are never materialized, in the forward pass or in the gradient | O(L) |
| `nnlib-ka` | NNlib's own fused kernel, for every eltype | a KernelAbstractions kernel, which also takes the calls cuDNN has no engine for | O(L) |
| `nnlib-mat` | the same function, kept on the materialized implementation | the full weights, from two `batched_mul` and a `softmax` | O(L²) |
| `lux` | [LuxLib](https://github.com/LuxDL/Lux.jl/tree/main/lib/LuxLib) `scaled_dot_product_attention` | materialized softmax path | O(L²) |

`bench_torch.py` runs PyTorch's `scaled_dot_product_attention` with each of its kernels:

| Name | Kernel | Memory |
|---|---|---|
| `math` | materialized: matmul, softmax, matmul | O(L²) |
| `efficient` | the memory-efficient kernel (CUTLASS), whose structure NNlib's own kernel follows | O(L) |
| `flash` | FlashAttention, half precision only | O(L) |
| `cudnn` | cuDNN's fused kernel, the one NNlib hands half precision to | O(L) |

All of them compute the identical math (scale `1/√E`). NNlib and LuxLib take
`(head_dim, nheads, seq, batch) = (E, H, L, B)` arrays, PyTorch `(B, H, L, E)`.

`nnlib-mat` passes a no-op `fdrop` that is not `identity`: NNlib only takes a fused
kernel without dropout, so this keeps it on the materialized implementation at no extra
cost.

## Run

Whole suite (all dtypes × both causal modes × all sizes × all impls):

```bash
julia --project=. bench_attention.jl

uv sync                                   # PyTorch, from pyproject.toml
uv run --project . python bench_torch.py
```

The Julia environment uses the NNlib of this checkout (`[sources]` in `Project.toml`).
The scripts take CLI flags so a single combination can be measured without rerunning the
whole suite:

```bash
# NNlib's kernel against the materialized implementation, Float32, non-causal:
julia --project=. bench_attention.jl --dtypes f32 --impls nnlib,nnlib-mat --causal false
# cuDNN's kernel against NNlib's, in half precision, on the small configs:
julia --project=. bench_attention.jl --dtypes f16,bf16 --impls nnlib,nnlib-ka --sizes tiny,small
# quick pass (shorter timing budget):
julia --project=. bench_attention.jl --seconds 0.3
julia --project=. bench_attention.jl --help     # all options
julia --project=. bench_attention.jl --list     # available configs / dtypes / impls
# PyTorch's memory-efficient kernel only, Float32:
uv run --project . python bench_torch.py --dtypes f32 --backends efficient
# pin a GPU:
CUDA_VISIBLE_DEVICES=GPU-<uuid> julia --project=. bench_attention.jl ...
```

| flag | values | default |
|---|---|---|
| `--dtypes` | `f16`,`bf16`,`f32` (aliases: `float16`/`fp16`/`half`, `bfloat16`, `float32`/`fp32`) | all |
| `--impls` | `nnlib`,`nnlib-ka`,`nnlib-mat`,`lux` | all |
| `--causal` | `false`,`true`, or `both` | both |
| `--sizes` | config names (see `--list`) | all |
| `--seconds` | per-measurement time budget | `1.0` |

### Timing

Each cell runs `CUDA.@sync(op)` under `BenchmarkTools.run(...; seconds, evals=1)`
and reports `minimum(trial)` in **ms**. `evals=1` is *per sample*, not total:
within the `seconds` budget BenchmarkTools collects as many samples as fit
(thousands for a sub-ms kernel) and we take the fastest — the standard robust
estimator for GPU kernels. The memory is `CUDA.@allocated` of one call. The
PyTorch script takes the minimum the same way, synchronizing after each call.

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

## Results — RTX 5090 (Blackwell, sm_120), CUDA.jl 6.4.2 / cuDNN.jl 6.4.2 (cuDNN 9.26), LuxLib 1.15.9, PyTorch 2.14.1 (CUDA 13.0, cuDNN 9.24)

Raw output: [`results_rtx5090_cuda64.txt`](results_rtx5090_cuda64.txt) and
[`results_rtx5090_torch214.txt`](results_rtx5090_torch214.txt).
Times in ms, lower is better.

### Float32

"fused" is `nnlib`, which here is NNlib's own kernel, in the forward pass and in the
gradient, and "materialized" is `nnlib-mat`.

Non-causal, forward (ms) and the memory a call allocates (MiB):

| config | L | fused | materialized | LuxLib | PyTorch efficient | PyTorch math | materialized ÷ fused | MiB fused | MiB materialized |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| tiny | 128 | 0.026 | 0.042 | 0.043 | 0.030 | 0.050 | 1.6× | 1.0 | 10 |
| small | 512 | 0.205 | 0.246 | 0.244 | 0.182 | 0.319 | 1.2× | 8.0 | 176 |
| gpt2-ish | 1024 | 0.511 | 0.747 | 0.751 | 0.440 | 1.090 | 1.5× | 12 | 456 |
| llama3 L=2k | 2048 | 2.291 | 2.342 | 2.355 | 2.332 | 4.000 | 1.0× | 32 | 1216 |
| llama3 L=4k | 4096 | 8.216 | 8.753 | 8.750 | 8.593 | 15.824 | 1.1× | 64 | 4480 |
| llama3 L=8k | 8192 | 31.184 | n/s | n/s | 32.957 | 63.174 | — | 128 | n/s |

Non-causal, full gradient (ms) and the memory it allocates (MiB):

| config | L | fused | materialized | LuxLib | PyTorch efficient | PyTorch math | materialized ÷ fused | MiB fused | MiB materialized |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| tiny | 128 | 0.184 | 0.397 | 0.949 | 0.099 | 0.188 | 2.2× | 5.0 | 23 |
| small | 512 | 1.054 | 0.963 | 1.108 | 0.771 | 0.785 | 0.9× | 40 | 376 |
| gpt2-ish | 1024 | 2.587 | 2.228 | 2.122 | 1.920 | 2.590 | 0.9× | 60 | 948 |
| llama3 L=2k | 2048 | 11.742 | 6.842 | 6.733 | 9.762 | 9.900 | 0.6× | 161 | 2528 |
| llama3 L=4k | 4096 | 42.295 | 24.839 | 24.740 | 36.216 | 38.744 | 0.6× | 321 | 9152 |
| llama3 L=8k | 8192 | 159.561 | n/s | n/s | 139.340 | n/s | — | 642 | n/s |

Causal, forward (ms) and the memory a call allocates (MiB):

| config | L | fused | materialized | LuxLib | PyTorch efficient | PyTorch math | materialized ÷ fused | MiB fused | MiB materialized |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| tiny | 128 | 0.027 | 0.054 | 0.052 | 0.030 | 0.065 | 2.0× | 1.0 | 12 |
| small | 512 | 0.143 | 0.376 | 0.372 | 0.121 | 0.372 | 2.6× | 8.0 | 240 |
| gpt2-ish | 1024 | 0.339 | 1.225 | 1.222 | 0.285 | 1.362 | 3.6× | 12 | 650 |
| llama3 L=2k | 2048 | 1.390 | 3.308 | 3.299 | 1.277 | 4.708 | 2.4× | 32 | 1736 |
| llama3 L=4k | 4096 | 4.597 | 12.557 | 12.499 | 4.503 | 19.871 | 2.7× | 64 | 6560 |
| llama3 L=8k | 8192 | 16.217 | n/s | n/s | 16.868 | 79.201 | — | 128 | n/s |

Causal, full gradient (ms) and the memory it allocates (MiB):

| config | L | fused | materialized | LuxLib | PyTorch efficient | PyTorch math | materialized ÷ fused | MiB fused | MiB materialized |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| tiny | 128 | 0.179 | 0.451 | 1.067 | 0.099 | 0.197 | 2.5× | 5.0 | 39 |
| small | 512 | 0.781 | 2.371 | 2.621 | 0.481 | 0.833 | 3.0× | 40 | 890 |
| gpt2-ish | 1024 | 1.822 | 6.273 | 6.336 | 1.193 | 2.843 | 3.4× | 60 | 2490 |
| llama3 L=2k | 2048 | 6.998 | 17.413 | 17.234 | 5.254 | 10.402 | 2.5× | 161 | 6648 |
| llama3 L=4k | 4096 | 23.171 | 67.739 | 67.836 | 18.829 | 42.105 | 2.9× | 321 | 25632 |
| llama3 L=8k | 8192 | 82.346 | OOM | OOM | 71.065 | n/s | — | 642 | OOM |

### Float16

`nnlib` is cuDNN's kernel here, in the forward pass and in the gradient. "NNlib's kernel"
is `nnlib-ka`, which NNlib takes for the calls cuDNN has no engine for, and "MiB fused"
is what either of them allocates in the forward pass.

Non-causal, forward (ms) and the memory a call allocates (MiB):

| config | L | cuDNN (`nnlib`) | NNlib's kernel | materialized | PyTorch flash | PyTorch cudnn | PyTorch efficient | PyTorch math | MiB fused | MiB materialized |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| tiny | 128 | 0.013 | 0.025 | 0.753 | 0.016 | 0.016 | 0.022 | 0.060 | 0.5 | 5.0 |
| small | 512 | 0.036 | 0.204 | 1.451 | 0.040 | 0.039 | 0.072 | 0.361 | 4.0 | 88 |
| gpt2-ish | 1024 | 0.118 | 0.506 | 1.103 | 0.089 | 0.118 | 0.157 | 1.305 | 6.0 | 228 |
| llama3 L=2k | 2048 | 0.407 | 2.288 | 1.533 | 0.444 | 0.408 | 0.880 | 4.600 | 16 | 608 |
| llama3 L=4k | 4096 | 1.388 | 8.410 | 4.161 | 1.508 | 1.401 | 3.253 | 17.980 | 32 | 2240 |
| llama3 L=8k | 8192 | 5.126 | 31.814 | n/s | 5.496 | 5.182 | 12.404 | 71.177 | 64 | n/s |

Non-causal, full gradient (ms) and the memory it allocates (MiB):

| config | L | cuDNN (`nnlib`) | NNlib's kernel | materialized | PyTorch flash | PyTorch cudnn | PyTorch efficient | PyTorch math | MiB cuDNN | MiB NNlib's kernel | MiB materialized |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| tiny | 128 | 0.168 | 0.145 | 2.683 | 0.068 | 0.080 | 0.069 | 0.201 | 4.5 | 2.5 | 12 |
| small | 512 | 0.198 | 1.045 | 5.172 | 0.131 | 0.146 | 0.263 | 0.820 | 36 | 20 | 188 |
| gpt2-ish | 1024 | 0.454 | 2.569 | 3.992 | 0.315 | 0.372 | 0.626 | 2.805 | 54 | 30 | 474 |
| llama3 L=2k | 2048 | 1.561 | 11.557 | 4.627 | 1.416 | 1.490 | 4.856 | 10.427 | 144 | 80 | 1264 |
| llama3 L=4k | 4096 | 5.352 | 41.514 | 11.799 | 4.988 | 5.292 | 18.610 | 40.749 | 289 | 161 | 4576 |
| llama3 L=8k | 8192 | 19.270 | 156.423 | n/s | 18.945 | 19.569 | 72.780 | n/s | 578 | 322 | n/s |

Causal, forward (ms) and the memory a call allocates (MiB):

| config | L | cuDNN (`nnlib`) | NNlib's kernel | materialized | PyTorch flash | PyTorch cudnn | PyTorch efficient | PyTorch math | MiB fused | MiB materialized |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| tiny | 128 | 0.014 | 0.025 | 0.764 | 0.016 | 0.017 | 0.022 | 0.088 | 0.5 | 6.0 |
| small | 512 | 0.037 | 0.144 | 1.486 | 0.039 | 0.040 | 0.054 | 0.405 | 4.0 | 120 |
| gpt2-ish | 1024 | 0.080 | 0.344 | 1.532 | 0.060 | 0.082 | 0.110 | 1.577 | 6.0 | 326 |
| llama3 L=2k | 2048 | 0.245 | 1.336 | 2.453 | 0.255 | 0.245 | 0.492 | 5.297 | 16 | 872 |
| llama3 L=4k | 4096 | 0.798 | 4.683 | 7.834 | 0.818 | 0.796 | 1.716 | 21.985 | 32 | 3296 |
| llama3 L=8k | 8192 | 2.793 | 16.929 | n/s | 2.853 | 2.786 | 6.372 | 87.205 | 64 | n/s |

Causal, full gradient (ms) and the memory it allocates (MiB):

| config | L | cuDNN (`nnlib`) | NNlib's kernel | materialized | PyTorch flash | PyTorch cudnn | PyTorch efficient | PyTorch math | MiB cuDNN | MiB NNlib's kernel | MiB materialized |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| tiny | 128 | 0.176 | 0.140 | 2.820 | 0.068 | 0.084 | 0.069 | 0.259 | 4.5 | 2.5 | 20 |
| small | 512 | 0.219 | 0.784 | 5.617 | 0.115 | 0.139 | 0.187 | 0.876 | 36 | 20 | 445 |
| gpt2-ish | 1024 | 0.329 | 1.823 | 7.405 | 0.213 | 0.264 | 0.418 | 3.046 | 54 | 30 | 1246 |
| llama3 L=2k | 2048 | 0.989 | 6.962 | 11.337 | 0.807 | 0.847 | 2.660 | 11.000 | 144 | 80 | 3328 |
| llama3 L=4k | 4096 | 3.155 | 23.281 | 38.110 | 2.670 | 2.847 | 9.611 | 44.181 | 289 | 161 | 12832 |
| llama3 L=8k | 8192 | 10.706 | 83.450 | n/s | 9.616 | 10.069 | 36.736 | n/s | 578 | 322 | n/s |

### BFloat16

Within noise of `Float16` in every column (0.118 against 0.118 ms for cuDNN at
`gpt2-ish`, 0.509 against 0.506 ms for NNlib's kernel): see the raw output.

## Takeaways

- **`Float32` has a fused path, and it is the faster one in the forward pass.** NNlib's
  kernel runs it 1.0–1.6× faster than the materialized implementation without a causal
  mask, the two being level at `llama3 L=2k`, and 2.0–3.6× faster with one, where it
  does about half the work while the materialized path builds and applies a mask.

- **Its memory is the output, and the three gradients when it is differentiated.** At
  `llama3 L=4k` a forward call allocates 64 MiB against 4480 MiB (6560 MiB with a causal
  mask), and a full gradient 321 MiB against 9152 MiB (25632 MiB). It is the only NNlib
  path that completes `L=8192` in `Float32`, forward or gradient.

- **The forward pass is level with PyTorch's memory-efficient kernel**, the one it is
  modelled on: 13% faster at `tiny`, 13–16% slower at `small` and `gpt2-ish` (0.511
  against 0.440 ms), and 2–5% faster from `llama3 L=2k` up. With a causal mask it is
  9–19% slower from `small` to `L=2k` and within 4% beyond.

- **The gradient costs four forward passes.** It recomputes the weights in three passes
  of the forward kernel, one for each of the gradients of the query, the key and the
  value, each accumulating in the registers of the blocks that own its positions: no
  update is atomic, and the result is deterministic. With a causal mask it is 2.5–3.4×
  faster than the materialized gradient (23.2 against 67.7 ms at `L=4k`). Without one it
  is faster at `tiny`, 9–16% slower at `small` and `gpt2-ish`, and 1.7× slower from
  `L=2k` up for as long as the materialized implementation fits (42.3 against 24.8 ms at
  `L=4k`).

- **PyTorch's memory-efficient gradient is ahead**: 1.15–1.4× faster from `small` up
  without a causal mask (36.2 ms at `L=4k`), 1.2–1.6× with one, and 1.8–1.9× at `tiny`.

- **In half precision cuDNN's kernel stays the one to take**: 1.8–6.2× faster than
  NNlib's kernel in the forward pass and up to 8× in the gradient, since it runs on
  tensor cores and NNlib's does not. PyTorch's memory-efficient kernel uses them as
  well, and is 1.1–3.2× faster than NNlib's there. NNlib's cuDNN path and PyTorch's
  `cudnn` kernel are the same code (0.118 against 0.118 ms at `gpt2-ish`, 5.13 against
  5.18 ms at `L=8k`).

- **Where cuDNN has no engine** (a mask, a head dimension that is not a multiple of 8),
  NNlib's kernel is what half precision takes, in the forward pass and in the gradient.
  Against the materialized implementation both are faster up to `gpt2-ish` and with a
  causal mask at every size (23.3 against 38.1 ms for the gradient at `L=4k`), slower
  from `L=2k` up without one (8.41 against 4.16 ms forward and 41.5 against 11.8 ms for
  the gradient at `L=4k`), and allocate far less (161 against 4576 MiB for that
  gradient).

- **NNlib ≈ LuxLib on the materialized path** — the same algorithm, within noise — and
  both are faster than PyTorch's `math` kernel in `Float32` (8.75 against 15.8 ms at
  `L=4k`).
