# CPU multi-head attention benchmark — NNlib (fused / materialized) and PyTorch

Compares NNlib's two implementations of scaled dot-product (multi-head) attention on the
CPU, and PyTorch's, across input sizes from tiny up to a single Llama3-8B attention
layer: the time of the forward pass and the memory it allocates.

| Name | Function | Approach | Memory |
|---|---|---|---|
| `nnlib` | NNlib `scaled_dot_product_attention` | fused: Julia tasks walk the keys in tiles, with the softmax taken online, and never form the attention weights | O(L) |
| `nnlib-mat` | the same function, kept on the materialized implementation | the full weights, from two `batched_mul` (BLAS) and a `softmax` | O(L²) |
| PyTorch | `torch.nn.functional.scaled_dot_product_attention`, in `bench_torch.py` | materialized: matmul, softmax, matmul | O(L²) |

All of them compute the identical math (scale `1/√E`). NNlib takes
`(head_dim, nheads, seq, batch) = (E, H, L, B)` arrays, PyTorch `(B, H, L, E)`.

`nnlib-mat` passes a no-op `fdrop` that is not `identity`: NNlib only takes the fused
implementation without dropout, so this keeps it on the materialized one at no extra cost.

## Run

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --threads=16 --project=. bench_attention.jl

uv sync                                             # PyTorch, from pyproject.toml
uv run --project . python bench_torch.py --threads 16
```

The Julia environment uses the NNlib of this checkout (`[sources]` in `Project.toml`).
Both scripts take flags, so that a single combination can be measured:

```bash
julia --threads=16 --project=. bench_attention.jl --dtypes f32 --causal true --sizes gpt2-ish
julia --project=. bench_attention.jl --help     # all options
julia --project=. bench_attention.jl --list     # available configs / dtypes / impls
```

| flag | values | default |
|---|---|---|
| `--dtypes` | `f16`,`f32`,`f64` | `f32` |
| `--impls` | `nnlib`,`nnlib-mat` | all |
| `--causal` | `false`,`true` | both |
| `--sizes` | config names (see `--list`) | all |
| `--seconds` | per-measurement time budget | `2.0` |

### Timing and threads

Each cell is the minimum over the samples that fit in the time budget, one evaluation
per sample, and the memory is what one call allocates.

The fused implementation runs on Julia threads. The materialized one runs its two
products on BLAS threads and its `softmax` on a single one. PyTorch uses its own
intra-op pool, set with `--threads`.

Configs (head_dim `E`, seq `L`, heads `H`, batch `B`), those of the
[GPU benchmark](../cuda_attention); the last three are a single Llama3-8B attention
layer (`E=128, H=32`) at growing context:

```
tiny         E=64  L=128  H=4  B=8
small        E=64  L=512  H=8  B=8
gpt2-ish     E=64  L=1024 H=12 B=4
llama3 L=2k  E=128 L=2048 H=32 B=1
llama3 L=4k  E=128 L=4096 H=32 B=1
llama3 L=8k  E=128 L=8192 H=32 B=1
```

## Results — Threadripper PRO 9955WX (16 cores / 32 threads, AVX-512), Julia 1.13.1, OpenBLAS, PyTorch 2.14.1

Raw output: [`results_tr9955wx_16threads.txt`](results_tr9955wx_16threads.txt),
[`results_tr9955wx_32threads.txt`](results_tr9955wx_32threads.txt) and
[`results_tr9955wx_torch214.txt`](results_tr9955wx_torch214.txt).

`Float32`, forward pass, times in ms, lower is better. "fused" is `nnlib` and
"materialized" is `nnlib-mat`, with 16 Julia threads and 16 BLAS threads unless stated;
PyTorch has 16 threads. The memory columns are what one call allocates.

Non-causal:

| config | L | fused | fused, 32 threads | materialized | PyTorch | materialized ÷ fused | PyTorch ÷ fused | MiB fused | MiB materialized |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| tiny | 128 | 0.16 | 0.15 | 3.4 | 0.18 | 21.6× | 1.1× | 2.2 | 10 |
| small | 512 | 2.92 | 2.87 | 103.3 | 3.17 | 35.4× | 1.1× | 10 | 176 |
| gpt2-ish | 1024 | 8.07 | 7.53 | 305.3 | 7.35 | 37.8× | 0.9× | 15 | 456 |
| llama3 L=2k | 2048 | 42.78 | 42.57 | 848.3 | 38.55 | 19.8× | 0.9× | 38 | 1216 |
| llama3 L=4k | 4096 | 159.98 | 150.86 | 2812.3 | 152.32 | 17.6× | 1.0× | 70 | 4481 |
| llama3 L=8k | 8192 | 637.93 | 593.44 | 10798.5 | 718.66 | 16.9× | 1.1× | 134 | 17153 |

Causal:

| config | L | fused | fused, 32 threads | materialized | PyTorch | materialized ÷ fused | PyTorch ÷ fused | MiB fused | MiB materialized |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| tiny | 128 | 0.13 | 0.12 | 3.2 | 0.18 | 24.5× | 1.4× | 2.2 | 12 |
| small | 512 | 2.02 | 1.80 | 121.2 | 3.16 | 59.9× | 1.6× | 10 | 241 |
| gpt2-ish | 1024 | 4.78 | 4.34 | 305.0 | 5.69 | 63.8× | 1.2× | 15 | 650 |
| llama3 L=2k | 2048 | 26.66 | 23.27 | 933.2 | 27.32 | 35.0× | 1.0× | 38 | 1736 |
| llama3 L=4k | 4096 | 90.94 | 79.63 | 3213.6 | 110.14 | 35.3× | 1.2× | 70 | 6561 |
| llama3 L=8k | 8192 | 334.16 | 307.12 | 11592.1 | 341.79 | 34.7× | 1.0× | 134 | 25473 |

## Takeaways

- **The fused implementation is 17–38× faster than the materialized one**, and 25–64×
  with a causal mask, where it skips the keys a query does not attend to. The
  materialized implementation is held back by its `softmax`, which runs on one thread:
  at `gpt2-ish` that is about 85% of its time, against 3% for the first of the two BLAS
  products.

- **Its memory is the output**, plus a small workspace for each task: 134 MiB at
  `llama3 L=8k` against 17 GiB, or 25 GiB with a causal mask.

- **It is level with PyTorch**, which materializes the weights too: within 14% either
  way without a causal mask (PyTorch ahead from `gpt2-ish` to `llama3 L=4k`, NNlib at
  `tiny`, `small` and `L=8k`), and 1.0–1.6× faster with one.

- **The second hardware thread of each core adds little**: 32 threads are 0–7% faster
  than 16 without a causal mask and 8–13% with one, from `small` up.
