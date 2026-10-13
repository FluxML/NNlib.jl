# GPU multi-head attention benchmark — NNlib (fused / materialized), LuxLib, PyTorch

Compares implementations of the same scaled dot-product (multi-head) attention on the
GPU, across input sizes from tiny up to a single Llama3-8B attention layer: the time of
the forward pass and of the full **Zygote** gradient (forward + backward), and the GPU
memory a forward call allocates.

| Name | Function | Approach | Memory |
|---|---|---|---|
| `nnlib` | NNlib `scaled_dot_product_attention`, as it dispatches | cuDNN's fused kernel for `Float16`/`BFloat16`, NNlib's own fused kernel for the other eltypes: the weights are never materialized | O(L) |
| `nnlib-ka` | NNlib's own fused kernel, for every eltype | a KernelAbstractions kernel, which also takes the calls cuDNN has no engine for. No gradient yet | O(L) |
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
estimator for GPU kernels. The memory is `CUDA.@allocated` of one forward call. The
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

"fused" is `nnlib`, which here is NNlib's own kernel, and "materialized" is `nnlib-mat`.
That kernel has no gradient yet, so the gradient NNlib computes is the one of the
materialized implementation.

Non-causal, forward (ms) and the memory a call allocates (MiB):

| config | L | fused | materialized | LuxLib | PyTorch efficient | PyTorch math | materialized ÷ fused | MiB fused | MiB materialized |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| tiny | 128 | 0.032 | 0.041 | 0.042 | 0.030 | 0.050 | 1.3× | 1.0 | 10 |
| small | 512 | 0.220 | 0.245 | 0.246 | 0.182 | 0.319 | 1.1× | 8.0 | 176 |
| gpt2-ish | 1024 | 0.540 | 0.748 | 0.753 | 0.440 | 1.090 | 1.4× | 12 | 456 |
| llama3 L=2k | 2048 | 2.352 | 2.353 | 2.359 | 2.332 | 4.000 | 1.0× | 32 | 1216 |
| llama3 L=4k | 4096 | 8.288 | 8.770 | 8.773 | 8.593 | 15.824 | 1.1× | 64 | 4480 |
| llama3 L=8k | 8192 | 31.168 | n/s | n/s | 32.957 | 63.174 | — | 128 | n/s |

Non-causal, full gradient (ms):

| config | L | NNlib (materialized) | LuxLib | PyTorch efficient | PyTorch math |
|---|--:|--:|--:|--:|--:|
| tiny | 128 | 0.376 | 0.932 | 0.099 | 0.188 |
| small | 512 | 0.951 | 1.085 | 0.771 | 0.785 |
| gpt2-ish | 1024 | 2.224 | 2.118 | 1.920 | 2.590 |
| llama3 L=2k | 2048 | 6.789 | 6.705 | 9.762 | 9.900 |
| llama3 L=4k | 4096 | 24.791 | 24.597 | 36.216 | 38.744 |
| llama3 L=8k | 8192 | n/s | n/s | 139.340 | n/s |

Causal, forward (ms) and the memory a call allocates (MiB):

| config | L | fused | materialized | LuxLib | PyTorch efficient | PyTorch math | materialized ÷ fused | MiB fused | MiB materialized |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| tiny | 128 | 0.032 | 0.057 | 0.051 | 0.030 | 0.065 | 1.8× | 1.0 | 12 |
| small | 512 | 0.159 | 0.374 | 0.373 | 0.121 | 0.372 | 2.4× | 8.0 | 240 |
| gpt2-ish | 1024 | 0.366 | 1.222 | 1.222 | 0.285 | 1.362 | 3.3× | 12 | 650 |
| llama3 L=2k | 2048 | 1.435 | 3.302 | 3.296 | 1.277 | 4.708 | 2.3× | 32 | 1736 |
| llama3 L=4k | 4096 | 4.703 | 12.555 | 12.538 | 4.503 | 19.871 | 2.7× | 64 | 6560 |
| llama3 L=8k | 8192 | 16.666 | n/s | n/s | 16.868 | 79.201 | — | 128 | n/s |

Causal, full gradient (ms):

| config | L | NNlib (materialized) | LuxLib | PyTorch efficient | PyTorch math |
|---|--:|--:|--:|--:|--:|
| tiny | 128 | 0.465 | 1.025 | 0.099 | 0.197 |
| small | 512 | 2.342 | 2.695 | 0.481 | 0.833 |
| gpt2-ish | 1024 | 6.253 | 6.332 | 1.193 | 2.843 |
| llama3 L=2k | 2048 | 17.259 | 17.278 | 5.254 | 10.402 |
| llama3 L=4k | 4096 | 69.160 | 67.563 | 18.829 | 42.105 |
| llama3 L=8k | 8192 | OOM | OOM | 71.065 | n/s |

### Float16

`nnlib` is cuDNN's kernel here. "NNlib's kernel" is `nnlib-ka`, which NNlib takes for
the calls cuDNN has no engine for, and "MiB fused" is what either of them allocates.

Non-causal, forward (ms) and the memory a call allocates (MiB):

| config | L | cuDNN (`nnlib`) | NNlib's kernel | materialized | PyTorch flash | PyTorch cudnn | PyTorch efficient | PyTorch math | MiB fused | MiB materialized |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| tiny | 128 | 0.012 | 0.031 | 0.747 | 0.016 | 0.016 | 0.022 | 0.060 | 0.5 | 5.0 |
| small | 512 | 0.036 | 0.217 | 1.451 | 0.040 | 0.039 | 0.072 | 0.361 | 4.0 | 88 |
| gpt2-ish | 1024 | 0.117 | 0.533 | 1.210 | 0.089 | 0.118 | 0.157 | 1.305 | 6.0 | 228 |
| llama3 L=2k | 2048 | 0.408 | 2.333 | 1.527 | 0.444 | 0.408 | 0.880 | 4.600 | 16 | 608 |
| llama3 L=4k | 4096 | 1.376 | 8.432 | 4.162 | 1.508 | 1.401 | 3.253 | 17.980 | 32 | 2240 |
| llama3 L=8k | 8192 | 5.086 | 31.549 | n/s | 5.496 | 5.182 | 12.404 | 71.177 | 64 | n/s |

Non-causal, full gradient (ms):

| config | L | cuDNN (`nnlib`) | materialized | LuxLib | PyTorch flash | PyTorch cudnn | PyTorch efficient | PyTorch math |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| tiny | 128 | 0.168 | 2.693 | 3.242 | 0.068 | 0.080 | 0.069 | 0.201 |
| small | 512 | 0.198 | 4.907 | 5.433 | 0.131 | 0.146 | 0.263 | 0.820 |
| gpt2-ish | 1024 | 0.455 | 3.958 | 4.612 | 0.315 | 0.372 | 0.626 | 2.805 |
| llama3 L=2k | 2048 | 1.558 | 4.660 | 4.667 | 1.416 | 1.490 | 4.856 | 10.427 |
| llama3 L=4k | 4096 | 5.306 | 11.802 | 11.588 | 4.988 | 5.292 | 18.610 | 40.749 |
| llama3 L=8k | 8192 | 19.210 | n/s | n/s | 18.945 | 19.569 | 72.780 | n/s |

Causal, forward (ms) and the memory a call allocates (MiB):

| config | L | cuDNN (`nnlib`) | NNlib's kernel | materialized | PyTorch flash | PyTorch cudnn | PyTorch efficient | PyTorch math | MiB fused | MiB materialized |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| tiny | 128 | 0.014 | 0.031 | 0.760 | 0.016 | 0.017 | 0.022 | 0.088 | 0.5 | 6.0 |
| small | 512 | 0.038 | 0.159 | 1.474 | 0.039 | 0.040 | 0.054 | 0.405 | 4.0 | 120 |
| gpt2-ish | 1024 | 0.080 | 0.367 | 1.512 | 0.060 | 0.082 | 0.110 | 1.577 | 6.0 | 326 |
| llama3 L=2k | 2048 | 0.244 | 1.367 | 2.445 | 0.255 | 0.245 | 0.492 | 5.297 | 16 | 872 |
| llama3 L=4k | 4096 | 0.791 | 4.664 | 7.822 | 0.818 | 0.796 | 1.716 | 21.985 | 32 | 3296 |
| llama3 L=8k | 8192 | 2.769 | 16.750 | n/s | 2.853 | 2.786 | 6.372 | 87.205 | 64 | n/s |

Causal, full gradient (ms):

| config | L | cuDNN (`nnlib`) | materialized | LuxLib | PyTorch flash | PyTorch cudnn | PyTorch efficient | PyTorch math |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| tiny | 128 | 0.177 | 2.780 | 3.363 | 0.068 | 0.084 | 0.069 | 0.259 |
| small | 512 | 0.198 | 5.585 | 6.149 | 0.115 | 0.139 | 0.187 | 0.876 |
| gpt2-ish | 1024 | 0.328 | 7.380 | 8.002 | 0.213 | 0.264 | 0.418 | 3.046 |
| llama3 L=2k | 2048 | 0.987 | 11.403 | 11.431 | 0.807 | 0.847 | 2.660 | 11.000 |
| llama3 L=4k | 4096 | 3.137 | 38.206 | 37.860 | 2.670 | 2.847 | 9.611 | 44.181 |
| llama3 L=8k | 8192 | 10.713 | n/s | n/s | 9.616 | 10.069 | 36.736 | n/s |

### BFloat16

Within noise of `Float16` in every column (0.118 against 0.117 ms for cuDNN at
`gpt2-ish`, 0.536 against 0.533 ms for NNlib's kernel): see the raw output.

## Takeaways

- **`Float32` has a fused path now, and it is the faster one.** NNlib's kernel runs the
  forward pass 1.0–1.4× faster than the materialized implementation without a causal
  mask, the two being level at `llama3 L=2k`, and 1.8–3.3× faster with one, where it
  does about half the work while the materialized path builds and applies a mask.

- **Its memory is the output.** A forward call allocates 64 MiB at `llama3 L=4k`
  against 4480 MiB (6560 MiB with a causal mask), and it is the only NNlib path that
  completes `L=8192` in `Float32`.

- **It is level with PyTorch's memory-efficient kernel**, the one it is modelled on:
  7–23% slower up to `gpt2-ish` (0.540 against 0.440 ms), equal at `llama3 L=2k`, and
  4–5% faster at `L=4k` and `L=8k`. With a causal mask it is 7–31% slower up to
  `L=2k` and within 5% beyond.

- **In half precision cuDNN's kernel stays the one to take**: 2.2–6.2× faster than
  NNlib's kernel, since it runs on tensor cores and NNlib's does not. PyTorch's
  memory-efficient kernel uses them as well, and is 1.4–3.4× faster than NNlib's there.
  NNlib's cuDNN path and PyTorch's `cudnn` kernel are the same code (0.117 against
  0.118 ms at `gpt2-ish`, 5.09 against 5.18 ms at `L=8k`).

- **Where cuDNN has no engine** (a mask, a bias, a head dimension that is not a
  multiple of 8), NNlib's kernel is what half precision now takes. Against the
  materialized implementation it is faster up to `gpt2-ish` and with a causal mask at
  every size, slower from `L=2k` up without one (8.43 against 4.16 ms at `L=4k`), and
  allocates 10–100× less.

- **The gradient is the next step.** NNlib differentiates the materialized
  implementation in `Float32`: with a causal mask that is 3.3–5.2× slower than PyTorch's
  memory-efficient kernel from `gpt2-ish` up (69 against 19 ms at `L=4k`) and runs out of
  memory at `L=8k`. Without a mask NNlib is the faster one from `L=2k` up (24.8 against
  36.2 ms at `L=4k`), but cannot run `L=8k`.

- **NNlib ≈ LuxLib on the materialized path** — the same algorithm, within noise — and
  both are faster than PyTorch's `math` kernel in `Float32` (8.77 against 15.8 ms at
  `L=4k`).
