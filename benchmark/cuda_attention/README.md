# GPU multi-head attention benchmark — NNlib vs LuxLib vs NNkernels

Compares three implementations of the same scaled dot-product (multi-head)
attention on the GPU, across input sizes from tiny up to a single Llama3-8B
attention layer, timing both the forward pass and the full **Zygote** gradient
(forward + backward).

| Library | Function | Approach | Memory |
|---|---|---|---|
| [NNlib](https://github.com/FluxML/NNlib.jl) | `dot_product_attention` | materialized: builds the full score tensor, `softmax`, second `batched_mul` | O(L²) |
| [LuxLib](https://github.com/LuxDL/Lux.jl/tree/main/lib/LuxLib) | `scaled_dot_product_attention` | materialized softmax path (own layout, KV-grouping) | O(L²) |
| [NNkernels](https://github.com/FluxML/NNkernels.jl) | `flash_attention` | fused [FlashAttention](https://arxiv.org/abs/2205.14135); scores never materialized | O(L) |

All three compute the identical math (scale `1/√E`). They only differ in the
memory layout they want for `q,k,v`, so the script builds all three from one
source tensor:

```
NNkernels: (head_dim, seq, nheads, batch)        = (E, L, H, B)
NNlib:     (head_dim*nheads, seq, batch), nheads=H = (E*H, L, B)
LuxLib:    (head_dim, nheads, seq, batch)         = (E, H, L, B)   (head_dim=1, token_dim=3)
```

## Run

Whole suite (all dtypes × both causal modes × all sizes × all impls):

```bash
julia --project=. bench_attention.jl
```

The script takes CLI flags so a **single dtype/impl combination can be measured
without rerunning the whole suite**:

```bash
# just NNkernels flash, Float32, non-causal:
julia --project=. bench_attention.jl --dtypes f32 --impls flash --causal false
# the two materialized paths in bf16+f16 on the small configs:
julia --project=. bench_attention.jl --dtypes f16,bf16 --impls nnlib,lux --sizes tiny,small
# quick pass (shorter timing budget):
julia --project=. bench_attention.jl --seconds 0.3
julia --project=. bench_attention.jl --help     # all options
julia --project=. bench_attention.jl --list      # available configs / dtypes / impls
# pin a GPU:
CUDA_VISIBLE_DEVICES=GPU-<uuid> julia --project=. bench_attention.jl ...
```

| flag | values | default |
|---|---|---|
| `--dtypes` | `f16`,`bf16`,`f32` (aliases: `float16`/`fp16`/`half`, `bfloat16`, `float32`/`fp32`) | all |
| `--impls` | `nnlib`,`lux`,`flash` | all |
| `--causal` | `false`,`true`, or `both` | both |
| `--sizes` | config names (see `--list`) | all |
| `--seconds` | per-measurement time budget | `1.0` |

### Timing

Each cell runs `CUDA.@sync(op)` under `BenchmarkTools.run(...; seconds, evals=1)`
and reports `minimum(trial)` in **ms**. `evals=1` is *per sample*, not total:
within the `seconds` budget BenchmarkTools collects as many samples as fit
(thousands for a sub-ms kernel) and we take the fastest — the standard robust
estimator for GPU kernels. `evals=1` (rather than auto-tuned `evals`) keeps one
clean synchronized GPU execution per sample, matching
[cuda_softmax/bench_softmax.jl](../cuda_softmax/bench_softmax.jl). The only
caveat is the multi-second `llama3 L=8k` flash cells, where a single sample
already exceeds the budget → 1 sample (directional, not tight).

`n/s` = that path can't run the config (reason echoed once to stderr);
`OOM` = ran out of GPU memory; `<impl>/fl` = time relative to flash
(>1 → flash faster).

Configs (head_dim `E`, seq `L`, heads `H`, batch `B`); the last three are a
single Llama3-8B attention layer (`E=128, H=32`) at growing context:

```
tiny       E=64  L=128  H=4  B=8
small      E=64  L=512  H=8  B=8
gpt2-ish   E=64  L=1024 H=12 B=4
llama3 L=2k  E=128 L=2048 H=32 B=1
llama3 L=4k  E=128 L=4096 H=32 B=1
llama3 L=8k  E=128 L=8192 H=32 B=1
```

## Environment / setup notes

The benchmark dev-depends on the local NNlib (`../..`). For the **CUDA.jl 6.2**
run, NNkernels 0.2.0 needed two local patches (it is `dev`-ed from
`~/.julia/dev/NNkernels`), because the registered package predates the CUDA 6 API:

1. **compat** — `CUDA = "5"` → `CUDA = "5, 6.1"` (the upstream
   `dependabot/julia/CUDA-5-and-6.1` branch; admits 6.2).
2. **CUDA 6.0 API rename** — its `NNkernelsCUDAExt` used
   `CUDA.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK`, which CUDA.jl 6.0
   dropped the `CU_` prefix from. Without this fix `flash_attention` throws
   `UndefVarError` at runtime for **every** dtype (the compat bump alone is not
   enough). Patched to pick the available name at load time.

Neither patch changes kernel behavior; flash numbers match the CUDA 5.11 run.

## Results — RTX 5090 (Blackwell, sm_120), **CUDA.jl 6.2 / cuDNN 6.2**, NNlib 0.9.36, LuxLib 1.15.9, NNkernels 0.2.0

Raw output: [`results_rtx5090_cuda62.txt`](results_rtx5090_cuda62.txt) (CUDA 6.2),
[`results_rtx5090_cuda511.txt`](results_rtx5090_cuda511.txt) (earlier CUDA 5.11).
Times in ms, lower is better.

### Float16

**Forward** — flash wins small, materialized wins from `gpt2-ish` up.

| config | L | nnlib | lux | flash | nnlib/fl | lux/fl |
|---|--:|--:|--:|--:|--:|--:|
| tiny | 128 | 0.791 | 0.856 | **0.053** | 15.0× | 16.2× |
| small | 512 | 1.526 | 1.531 | **0.402** | 3.8× | 3.8× |
| gpt2-ish | 1024 | 1.157 | 1.155 | **0.922** | 1.25× | 1.25× |
| llama3 L=2k | 2048 | **1.511** | 1.518 | 8.110 | 0.19× | 0.19× |
| llama3 L=4k | 4096 | **4.138** | 4.160 | 29.82 | 0.14× | 0.14× |
| llama3 L=8k | 8192 | n/s | n/s | **114.0** | — | — |

**Full gradient (fwd+bwd)**

| config | L | nnlib | lux | flash | nnlib/fl | lux/fl |
|---|--:|--:|--:|--:|--:|--:|
| tiny | 128 | 2.490 | 3.155 | **0.290** | 8.6× | 10.9× |
| small | 512 | 4.726 | 5.457 | **3.227** | 1.5× | 1.7× |
| gpt2-ish | 1024 | **3.765** | 4.880 | 11.67 | 0.32× | 0.42× |
| llama3 L=2k | 2048 | **4.282** | 4.460 | 167.4 | 0.03× | 0.03× |
| llama3 L=4k | 4096 | **11.27** | 11.42 | 674.7 | 0.02× | 0.02× |
| llama3 L=8k | 8192 | n/s | n/s | **2723** | — | — |

### BFloat16 — **now runs on the materialized paths under CUDA 6.2**

Under CUDA 5.11 every bf16 cell was `n/s` (cuDNN had no bf16 softmax). CUDA 6.2's
improved BFloat16 support fixes the NNlib/LuxLib path. **Flash is still `n/s`**:
NNkernels' `_flash_attention` dispatches only on `T<:Union{Float16,Float32}` at
the *source* level — independent of the CUDA version. (bf16 numbers track
fp16 within noise.)

**Forward**

| config | L | nnlib | lux | flash |
|---|--:|--:|--:|--:|
| tiny | 128 | 0.786 | 0.791 | n/s |
| small | 512 | 1.717 | 1.561 | n/s |
| gpt2-ish | 1024 | 1.264 | 1.270 | n/s |
| llama3 L=2k | 2048 | 1.686 | 1.688 | n/s |
| llama3 L=4k | 4096 | 4.494 | 4.510 | n/s |
| llama3 L=8k | 8192 | n/s | n/s | n/s |

**Full gradient (fwd+bwd)**

| config | L | nnlib | lux | flash |
|---|--:|--:|--:|--:|
| tiny | 128 | 2.796 | 3.172 | n/s |
| small | 512 | 5.335 | 5.508 | n/s |
| gpt2-ish | 1024 | 4.068 | 4.722 | n/s |
| llama3 L=2k | 2048 | 4.688 | 4.845 | n/s |
| llama3 L=4k | 4096 | 12.13 | 12.23 | n/s |
| llama3 L=8k | 8192 | n/s | n/s | n/s |

Both within noise of the fp16 numbers — bf16 buys the same speed here, just less
precision (8 mantissa bits, so the `lux`-vs-`nnlib` sanity diff is ~1e-2, a
precision/accumulation artifact rather than a bug).

### Float32

**Forward**

| config | L | nnlib | lux | flash | nnlib/fl | lux/fl |
|---|--:|--:|--:|--:|--:|--:|
| tiny | 128 | **0.050** | 0.051 | 0.068 | 0.73× | 0.75× |
| small | 512 | **0.243** | 0.242 | 0.806 | 0.30× | 0.30× |
| gpt2-ish | 1024 | **0.743** | 0.746 | 2.238 | 0.33× | 0.33× |
| llama3 L=2k | 2048 | **2.338** | 2.353 | 18.33 | 0.13× | 0.13× |
| llama3 L=4k | 4096 | **8.759** | 8.787 | 69.50 | 0.13× | 0.13× |
| llama3 L=8k | 8192 | n/s | n/s | **270.5** | — | — |

**Full gradient (fwd+bwd)**

| config | L | nnlib | lux | flash | nnlib/fl | lux/fl |
|---|--:|--:|--:|--:|--:|--:|
| tiny | 128 | **0.281** | 0.895 | 0.621 | 0.45× | 1.44× |
| small | 512 | **0.766** | 1.027 | 8.702 | 0.09× | 0.12× |
| gpt2-ish | 1024 | **2.014** | 2.098 | 35.54 | 0.06× | 0.06× |
| llama3 L=2k | 2048 | **6.522** | 6.670 | 467.6 | 0.01× | 0.01× |
| llama3 L=4k | 4096 | **24.61** | 24.52 | 1891 | 0.01× | 0.01× |
| llama3 L=8k | 8192 | OOM | OOM | **7552** | — | — |

## What changed with CUDA.jl 6.2 (vs 5.11)

- **BFloat16 became usable** on the NNlib/LuxLib materialized paths (was `n/s`
  everywhere). This is the headline of the update.
- **FP32 materialized path sped up ~20–25%** at large sizes from the
  cuBLAS/cuDNN bump — e.g. `llama3 L=4k` FP32 gradient **31.9 → 24.6 ms**,
  forward **10.5 → 8.8 ms**; `L=2k` gradient **8.4 → 6.5 ms**. Float16 was
  already fast and is unchanged.
- **Flash is unchanged** (same kernel) once patched for the CUDA 6.0 API rename —
  but note it would have been **completely broken** (`UndefVarError` for all
  dtypes) on a stock NNkernels 0.2.0 under CUDA 6.2.

## Takeaways

- **NNlib ≈ LuxLib** everywhere — same materialized algorithm, within noise.
  LuxLib carries more autodiff overhead at tiny sizes (Float32 gradient `tiny`:
  0.90 vs 0.28 ms).

- **Flash wins only at small `Float16` sizes, then loses badly at scale on this
  GPU.** For `tiny`/`small` `Float16` it is 3–17× faster, but from `gpt2-ish`
  upward the materialized path overtakes it, and at Llama sizes flash is **5–6×
  slower on the forward and ~50–100× slower on the full gradient** (its backward
  kernel is the bottleneck). For `Float32` flash is slower at *every* size.

  Two distinct causes, one confirmed in the NNkernels source:
  - **`Float32` is slow by construction.** The kernel only takes the `WMMA`
    tensor-core path when `sizeof(T) == 2` (`src/attention.jl:162`); `Float32`
    always falls back to the scalar `mma!`, while cuDNN/CUBLAS use FP32 SIMT/TF32
    paths the materialized route rides on.
  - **`Float16` *does* use WMMA** (`WMMA.is_available(::CUDABackend)` is hardcoded
    `true`), yet still trails at Llama sizes — its sm_70/80-era `wmma` intrinsics
    don't map to Blackwell's newest tensor cores, and the tiling/occupancy and
    the backward kernel aren't tuned for **sm_120**. On the Ampere/Ada cards
    NNkernels' CI targets these numbers would likely look different — **re-run
    before drawing hardware-agnostic conclusions.**

- **Flash's real, layout-independent advantage is memory.** It is the only path
  that completes `L=8192`: the materialized score tensor `(8192, 8192, 32)` has
  `2³¹` elements, which overflows cuDNN's `int32` softmax descriptor
  (`CUDNN_STATUS_NOT_SUPPORTED` → `n/s`), and in `Float32`-causal it exhausts the
  32 GB card (`OOM`). FlashAttention's O(L) memory is what lets it scale to long
  context even when it is slower per step here.

- **Causal masking** roughly halves flash's work (it skips masked tiles) while
  making the materialized path slightly *slower* (it builds and applies an
  explicit mask), narrowing — but not closing — the gap at large sizes.
