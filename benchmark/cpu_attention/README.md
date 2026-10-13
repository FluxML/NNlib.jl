# CPU multi-head attention benchmark — NNlib (fused / materialized) and PyTorch

Compares NNlib's two implementations of scaled dot-product (multi-head) attention on the
CPU, and PyTorch's, across input sizes from tiny up to a single Llama3-8B attention
layer: the time of the forward pass and of the full **Zygote** gradient (forward +
backward), and the memory each of the two allocates.

| Name | Function | Approach | Memory |
|---|---|---|---|
| `nnlib` | NNlib `scaled_dot_product_attention` | fused: Julia tasks walk the keys in tiles, with the softmax taken online, and never form the attention weights, in the forward pass or in the gradient | O(L) |
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
| `--passes` | `forward`,`gradient` | both |
| `--seconds` | per-measurement time budget | `2.0` |

### Timing and threads

Each cell is the minimum over the samples that fit in the time budget, one evaluation
per sample, and the memory is what one call allocates. The gradient is that of the sum
of the output with respect to the query, the key and the value.

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

`Float32`, times in ms, lower is better. "fused" is `nnlib` and "materialized" is
`nnlib-mat`, with 16 Julia threads and 16 BLAS threads unless stated; PyTorch has 16
threads. The memory columns are what one call allocates.

### Forward

Non-causal:

| config | L | fused | fused, 32 threads | materialized | PyTorch | materialized ÷ fused | PyTorch ÷ fused | MiB fused | MiB materialized |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| tiny | 128 | 0.13 | 0.14 | 3.3 | 0.16 | 24.6× | 1.2× | 2.7 | 10 |
| small | 512 | 2.62 | 2.53 | 114.9 | 3.11 | 43.8× | 1.2× | 11 | 176 |
| gpt2-ish | 1024 | 7.72 | 7.21 | 307.0 | 7.45 | 39.8× | 1.0× | 15 | 456 |
| llama3 L=2k | 2048 | 38.43 | 37.16 | 880.3 | 38.32 | 22.9× | 1.0× | 39 | 1216 |
| llama3 L=4k | 4096 | 145.55 | 140.05 | 3083.4 | 152.56 | 21.2× | 1.0× | 71 | 4481 |
| llama3 L=8k | 8192 | 567.40 | 547.54 | 11022.4 | 607.71 | 19.4× | 1.1× | 135 | 17153 |

Causal:

| config | L | fused | fused, 32 threads | materialized | PyTorch | materialized ÷ fused | PyTorch ÷ fused | MiB fused | MiB materialized |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| tiny | 128 | 0.12 | 0.12 | 3.2 | 0.17 | 26.1× | 1.4× | 2.7 | 12 |
| small | 512 | 1.79 | 1.58 | 106.8 | 3.08 | 59.6× | 1.7× | 11 | 241 |
| gpt2-ish | 1024 | 4.49 | 4.20 | 314.1 | 5.66 | 70.0× | 1.3× | 15 | 650 |
| llama3 L=2k | 2048 | 23.39 | 20.20 | 949.7 | 25.03 | 40.6× | 1.1× | 39 | 1736 |
| llama3 L=4k | 4096 | 79.44 | 74.51 | 3201.8 | 85.06 | 40.3× | 1.1× | 71 | 6561 |
| llama3 L=8k | 8192 | 300.71 | 286.64 | 11979.8 | 330.87 | 39.8× | 1.1× | 135 | 25473 |

### Full gradient

Non-causal:

| config | L | fused | fused, 32 threads | materialized | PyTorch | materialized ÷ fused | PyTorch ÷ fused | MiB fused | MiB materialized |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| tiny | 128 | 0.84 | 0.99 | 5.7 | 0.80 | 6.8× | 1.0× | 9.9 | 23 |
| small | 512 | 10.81 | 10.35 | 153.9 | 15.32 | 14.2× | 1.4× | 47 | 376 |
| gpt2-ish | 1024 | 30.50 | 27.31 | 372.9 | 32.03 | 12.2× | 1.1× | 68 | 949 |
| llama3 L=2k | 2048 | 154.01 | 133.84 | 1268.2 | 150.52 | 8.2× | 1.0× | 177 | 2529 |
| llama3 L=4k | 4096 | 561.89 | 502.15 | 4132.7 | 572.77 | 7.4× | 1.0× | 337 | 9153 |
| llama3 L=8k | 8192 | 2154.59 | 1885.12 | 15176.5 | 2195.30 | 7.0× | 1.0× | 657 | 34690 |

Causal:

| config | L | fused | fused, 32 threads | materialized | PyTorch | materialized ÷ fused | PyTorch ÷ fused | MiB fused | MiB materialized |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| tiny | 128 | 0.87 | 0.90 | 7.5 | 0.80 | 8.6× | 0.9× | 9.9 | 39 |
| small | 512 | 7.71 | 7.13 | 349.2 | 15.20 | 45.3× | 2.0× | 47 | 890 |
| gpt2-ish | 1024 | 18.00 | 17.11 | 970.9 | 25.64 | 53.9× | 1.4× | 68 | 2491 |
| llama3 L=2k | 2048 | 88.80 | 78.29 | 2610.6 | 95.10 | 29.4× | 1.1× | 177 | 6649 |
| llama3 L=4k | 4096 | 337.66 | 276.72 | 9289.7 | 341.69 | 27.5× | 1.0× | 337 | 25633 |
| llama3 L=8k | 8192 | 1177.16 | 1003.89 | 35195.4 | 1215.88 | 29.9× | 1.0× | 657 | 100610 |

## Takeaways

- **The fused implementation is 19–44× faster than the materialized one** in the forward
  pass, and 26–70× with a causal mask, where it skips the keys a query does not attend
  to. The materialized implementation is held back by its `softmax`, which runs on one
  thread: at `gpt2-ish` that is about 85% of its time, against 3% for the first of the
  two BLAS products.

- **Its memory is the output**, plus a small workspace for each task, and the three
  gradients when it is differentiated: at `llama3 L=8k` a forward call allocates
  135 MiB against 17 GiB, and a full gradient 657 MiB against 34 GiB.

- **The forward pass is level with PyTorch**, which materializes the weights too: 1.2×
  faster at `tiny` and `small`, and within 7% either way from `gpt2-ish` up (PyTorch 4%
  ahead at `gpt2-ish`, NNlib 5–7% at `L=4k` and `L=8k`). With a causal mask it is
  1.1–1.7× faster.

- **The gradient takes about four times the forward pass** from `small` up, the forward
  pass included. Each pair of a tile of queries and a tile of keys is visited once, and
  gives its part of the three gradients. It is 7–14× faster than the materialized
  gradient, and 9–54× with a causal mask.

- **The gradient is level with PyTorch's too**: within 5% either way from `gpt2-ish` up
  and 1.4× faster at `small`, and with a causal mask 1.4–2.0× faster at `small` and
  `gpt2-ish` and 1–7% faster beyond. At `tiny` PyTorch is 5–9% ahead.

- **The second hardware thread of each core adds little to the forward pass, more to the
  gradient**: from `small` up, 32 threads are 3–7% faster than 16 in the forward pass
  (5–14% with a causal mask) and 4–13% in the gradient (5–18%).
