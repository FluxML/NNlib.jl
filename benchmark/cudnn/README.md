# cuDNN operator benchmark — which implementation should NNlib dispatch to?

For each operator that NNlib accelerates on CUDA, this times every implementation it
could dispatch to, on the same inputs, so that the choice can be made, and revisited as
cuDNN and CUDA.jl evolve, on measurements.

| Column | Implementation |
|---|---|
| `nnlib` | what NNlib runs today for a `CuArray`: one of the candidates below, plus NNlib's own wrapper |
| `generic` | NNlib's device-agnostic implementation, built from broadcasts and reductions that CUDA.jl compiles to GPU kernels; no cuDNN involved |
| `fixed` | cuDNN's fixed-function API (`cudnnPoolingForward` and the like), which cuDNN.jl keeps under `legacy` |
| `graph` | cuDNN's graph (backend) API, through cuDNN.jl's operation wrappers |

Operators: batch normalization (training, inference, gradient), max and mean pooling with
their gradients, the `relu`/`tanh`/`σ`/`elu` broadcasts, `softmax` and its gradient along
the first dimension and along the channels, convolution and its two gradients. Attention
has its own benchmark in [`../cuda_attention`](../cuda_attention).

## Where NNlib stands

From the run below (RTX 5090, cuDNN 9.26), slowdowns relative to the fastest candidate:

| Operator | NNlib uses | Why |
|---|---|---|
| batchnorm: training, inference, gradient | `fixed` | `graph` is 1.6–3.0× slower in training and 1.5–2.6× in the gradient, and has no engine for inference or for a number of channels that is not a multiple of 4 (nor for `Float64`, which this run does not include). `generic` is 2–21× slower |
| `maxpool`, `meanpool` and their gradients | `fixed`; `graph` only for asymmetric padding, which `fixed` cannot do | `graph` is 1.06–1.22× slower forward and up to 2.6× in `∇maxpool` |
| `relu`, `tanh`, `σ`, `elu` broadcasts | `fixed` | `generic` is 1.2–5.7× slower. cuDNN has no engine for a graph made of one pointwise operation |
| `softmax` over leading dimensions | `fixed` (instance mode) | `generic` is 2.7–23× slower |
| `softmax` over other dimensions | `generic` | **worth revisiting**: `fixed` (channel mode) is 4–15× faster forward and 2–10× in the gradient on these inputs. See the note below |
| `conv` and its gradients | `graph` | `fixed` is up to 2.2× slower below the largest size and 4–8% slower at the largest. The exceptions are in `Float32`: at the largest size `fixed` is 7% faster for `∇conv_data` and 14% for `∇conv_filter`, and 6% for `∇conv_filter` with 30 channels. `graph` also takes asymmetric padding natively |

## Run

```bash
julia --project=. bench_cudnn.jl
```

The environment uses the NNlib of this checkout (`[sources]` in `Project.toml`). A subset
can be run on its own:

```bash
julia --project=. bench_cudnn.jl --ops batchnorm,pooling --dtypes f16
julia --project=. bench_cudnn.jl --sizes large --seconds 1 --out large.tsv
julia --project=. bench_cudnn.jl --help     # all options
julia --project=. bench_cudnn.jl --list     # operator families, sizes, dtypes
# pin a GPU:
CUDA_VISIBLE_DEVICES=GPU-<uuid> julia --project=. bench_cudnn.jl ...
```

| flag | values | default |
|---|---|---|
| `--ops` | `batchnorm`,`pooling`,`activations`,`softmax`,`conv` | all |
| `--dtypes` | `f16`,`bf16`,`f32`,`f64` | `f32,f16` |
| `--sizes` | `small`,`medium`,`large`,`odd-C` | all |
| `--seconds` | per-measurement time budget | `0.5` |
| `--out` | file to also write the timings to, as tab-separated values | none |
| `--no-check` | skip the check that the implementations agree | off |

Inputs are `(width, height, channels, batch)`:

```
small   8×8×16×4
medium  32×32×64×16
large   64×64×128×32
odd-C   32×32×30×16     a number of channels that is not a multiple of 4
```

### How it measures

- Each cell runs `CUDA.@sync(op)` under `BenchmarkTools.run(...; seconds, evals=1)` and
  reports `minimum(trial)` in **μs**.
- Every candidate allocates its outputs, as NNlib's functions do, so the candidates of
  an operator differ only in the kernel they run.
- Before timing, the outputs of the candidates are compared with NNlib's on a small
  input, and the largest deviation is printed.
- `n/s` = the implementation cannot run that configuration, e.g. cuDNN has no engine for
  it (the reason is echoed once to stderr); `OOM` = out of GPU memory; `-` = there is no
  such implementation.
- The cuDNN batchnorm gradients are given the batch statistics saved by their forward
  pass, which is not timed. `generic` recomputes them, and so does `nnlib` when called
  without a cache, as from its `rrule`.

`nnlib` can be slower than the candidate it dispatches to: that difference is NNlib's
wrapper. In this run it shows in `∇batchnorm` (cuDNN recomputing the batch statistics,
19.6 vs 11.2 μs at the medium `Float16` size) and in pooling, whose allocating functions
zero the output before cuDNN overwrites it.

### Adding an operator

Write a `<family>_ops(T, sz)` function returning its `Op`s, each a name and the thunks of
its implementations, and register it in `FAMILIES`. The candidates call cuDNN.jl through
qualified names inside functions, so that one whose API has gone shows up as `n/s` rather
than breaking the script.

## Results — RTX 5090 (Blackwell, sm_120), Julia 1.13.1, CUDA.jl 6.4.2 / cuDNN.jl 6.4.2 (cuDNN 9.26)

Raw output: [`results_rtx5090_cudnn926.txt`](results_rtx5090_cudnn926.txt), and as
tab-separated values [`results_rtx5090_cudnn926.tsv`](results_rtx5090_cudnn926.tsv).
The tables show the medium and large sizes; times in μs, lower is better, with the
fastest candidate in bold.

### Batch normalization

| operator | eltype | size | nnlib | generic | fixed | graph | fastest |
|---|---|---|--:|--:|--:|--:|---|
| `batchnorm (training)` | Float16 | 32×32×64×16 | 9.4 | 79.5 | **9.1** | 25.1 | fixed |
| `batchnorm (inference)` | Float16 | 32×32×64×16 | 7.8 | 45.0 | **7.6** | n/s | fixed |
| `∇batchnorm (training)` | Float16 | 32×32×64×16 | 19.6 | 148.1 | **11.2** | 25.2 | fixed |
| `batchnorm (training)` | Float16 | 64×64×128×32 | 57.7 | 714.9 | **57.7** | 94.5 | fixed |
| `batchnorm (inference)` | Float16 | 64×64×128×32 | 23.1 | 480.6 | **22.9** | n/s | fixed |
| `∇batchnorm (training)` | Float16 | 64×64×128×32 | 105.8 | 1462.8 | **86.3** | 155.6 | fixed |
| `batchnorm (training)` | Float32 | 32×32×64×16 | 12.3 | 60.2 | **12.2** | 26.9 | fixed |
| `batchnorm (inference)` | Float32 | 32×32×64×16 | 8.6 | 25.8 | **8.2** | n/s | fixed |
| `∇batchnorm (training)` | Float32 | 32×32×64×16 | 19.5 | 118.5 | **16.5** | 28.3 | fixed |
| `batchnorm (training)` | Float32 | 64×64×128×32 | 108.2 | 468.9 | **108.2** | 217.3 | fixed |
| `batchnorm (inference)` | Float32 | 64×64×128×32 | 93.8 | 221.2 | **92.9** | n/s | fixed |
| `∇batchnorm (training)` | Float32 | 64×64×128×32 | 269.3 | 1070.8 | **244.8** | 355.9 | fixed |

With 30 channels (`odd-C`) `graph` is `n/s` for all three, and inference is `n/s` at
every size.

### Pooling

| operator | eltype | size | nnlib | generic | fixed | graph | fastest |
|---|---|---|--:|--:|--:|--:|---|
| `maxpool` | Float16 | 32×32×64×16 | 8.6 | - | **7.2** | 8.2 | fixed |
| `∇maxpool` | Float16 | 32×32×64×16 | 10.3 | - | **9.0** | 12.9 | fixed |
| `meanpool` | Float16 | 32×32×64×16 | 8.5 | - | **7.3** | 8.3 | fixed |
| `∇meanpool` | Float16 | 32×32×64×16 | 9.8 | - | **8.5** | 10.2 | fixed |
| `maxpool` | Float16 | 64×64×128×32 | 19.3 | - | **17.7** | 19.5 | fixed |
| `∇maxpool` | Float16 | 64×64×128×32 | 52.9 | - | **34.1** | 87.2 | fixed |
| `meanpool` | Float16 | 64×64×128×32 | 20.9 | - | **18.9** | 20.2 | fixed |
| `∇meanpool` | Float16 | 64×64×128×32 | 47.5 | - | **37.0** | 39.0 | fixed |
| `maxpool` | Float32 | 32×32×64×16 | 9.1 | - | **7.3** | 8.5 | fixed |
| `∇maxpool` | Float32 | 32×32×64×16 | 10.8 | - | **9.6** | 13.3 | fixed |
| `meanpool` | Float32 | 32×32×64×16 | 8.5 | - | **7.2** | 8.5 | fixed |
| `∇meanpool` | Float32 | 32×32×64×16 | 10.1 | - | **8.4** | 10.2 | fixed |
| `maxpool` | Float32 | 64×64×128×32 | 51.4 | - | **19.0** | 20.7 | fixed |
| `∇maxpool` | Float32 | 64×64×128×32 | 144.2 | - | **118.4** | 128.6 | fixed |
| `meanpool` | Float32 | 64×64×128×32 | 51.0 | - | **18.7** | 19.9 | fixed |
| `∇meanpool` | Float32 | 64×64×128×32 | 65.7 | - | **39.0** | 43.1 | fixed |

### Activations

`tanh` and `elu` time like the two shown.

| operator | eltype | size | nnlib | generic | fixed | graph | fastest |
|---|---|---|--:|--:|--:|--:|---|
| `relu.(x)` | Float16 | 32×32×64×16 | 7.1 | 16.1 | **7.6** | n/s | fixed |
| `σ.(x)` | Float16 | 32×32×64×16 | 7.7 | 17.0 | **7.6** | n/s | fixed |
| `relu.(x)` | Float16 | 64×64×128×32 | 23.0 | 130.1 | **22.9** | n/s | fixed |
| `σ.(x)` | Float16 | 64×64×128×32 | 27.3 | 143.1 | **27.5** | n/s | fixed |
| `relu.(x)` | Float32 | 32×32×64×16 | 7.6 | 16.4 | **8.1** | n/s | fixed |
| `σ.(x)` | Float32 | 32×32×64×16 | 7.9 | 16.6 | **8.0** | n/s | fixed |
| `relu.(x)` | Float32 | 64×64×128×32 | 90.4 | 139.6 | **91.1** | n/s | fixed |
| `σ.(x)` | Float32 | 64×64×128×32 | 91.1 | 144.8 | **90.6** | n/s | fixed |

### Softmax

| operator | eltype | size | nnlib | generic | fixed | graph | fastest |
|---|---|---|--:|--:|--:|--:|---|
| `softmax (dims=1)` | Float16 | 32×32×64×16 | 8.0 | 85.6 | **7.7** | - | fixed |
| `∇softmax (dims=1)` | Float16 | 32×32×64×16 | 7.8 | 49.6 | **7.8** | - | fixed |
| `softmax (dims=3)` | Float16 | 32×32×64×16 | 83.5 | 83.6 | **10.8** | - | fixed |
| `∇softmax (dims=3)` | Float16 | 32×32×64×16 | 49.1 | 49.2 | **9.9** | - | fixed |
| `softmax (dims=1)` | Float16 | 64×64×128×32 | 25.7 | 594.1 | **25.6** | - | fixed |
| `∇softmax (dims=1)` | Float16 | 64×64×128×32 | 54.4 | 459.0 | **54.1** | - | fixed |
| `softmax (dims=3)` | Float16 | 64×64×128×32 | 807.8 | 810.0 | **53.0** | - | fixed |
| `∇softmax (dims=3)` | Float16 | 64×64×128×32 | 562.5 | 563.6 | **55.6** | - | fixed |
| `softmax (dims=1)` | Float32 | 32×32×64×16 | 7.9 | 85.1 | **8.0** | - | fixed |
| `∇softmax (dims=1)` | Float32 | 32×32×64×16 | 8.1 | 49.7 | **8.1** | - | fixed |
| `softmax (dims=3)` | Float32 | 32×32×64×16 | 82.7 | 83.2 | **19.8** | - | fixed |
| `∇softmax (dims=3)` | Float32 | 32×32×64×16 | 48.3 | 48.3 | **17.7** | - | fixed |
| `softmax (dims=1)` | Float32 | 64×64×128×32 | 89.4 | 625.9 | **75.7** | - | fixed |
| `∇softmax (dims=1)` | Float32 | 64×64×128×32 | 131.1 | 483.7 | **130.6** | - | fixed |
| `softmax (dims=3)` | Float32 | 64×64×128×32 | 826.8 | 824.3 | **171.7** | - | fixed |
| `∇softmax (dims=3)` | Float32 | 64×64×128×32 | 585.6 | 587.6 | **218.3** | - | fixed |

NNlib uses its generic kernel for a softmax that is not over leading dimensions since
[#513](https://github.com/FluxML/NNlib.jl/issues/513), where cuDNN's channel mode was
much slower on the backward pass for a long softmax axis. The channel axes here are short
(16 to 128), and there channel mode wins. [`../cuda_softmax`](../cuda_softmax) has the
long axes; both should be looked at before changing the routing.

### Convolution

3×3 kernel, padding 1, as many output as input channels.

| operator | eltype | size | nnlib | generic | fixed | graph | fastest |
|---|---|---|--:|--:|--:|--:|---|
| `conv` | Float16 | 32×32×64×16 | 25.0 | - | 42.3 | **24.3** | graph |
| `∇conv_data` | Float16 | 32×32×64×16 | 25.0 | - | 41.7 | **24.4** | graph |
| `∇conv_filter` | Float16 | 32×32×64×16 | 30.5 | - | 44.9 | **30.3** | graph |
| `conv` | Float16 | 64×64×128×32 | 293.1 | - | 317.1 | **293.9** | graph |
| `∇conv_data` | Float16 | 64×64×128×32 | 262.2 | - | 282.3 | **261.6** | graph |
| `∇conv_filter` | Float16 | 64×64×128×32 | 394.3 | - | 412.8 | **392.3** | graph |
| `conv` | Float32 | 32×32×64×16 | 32.6 | - | 41.1 | **32.4** | graph |
| `∇conv_data` | Float32 | 32×32×64×16 | 32.5 | - | 42.2 | **32.7** | graph |
| `∇conv_filter` | Float32 | 32×32×64×16 | 49.2 | - | 62.3 | **49.0** | graph |
| `conv` | Float32 | 64×64×128×32 | 491.5 | - | 511.6 | **491.2** | graph |
| `∇conv_data` | Float32 | 64×64×128×32 | 547.8 | - | **508.6** | 547.9 | fixed |
| `∇conv_filter` | Float32 | 64×64×128×32 | 675.0 | - | **584.2** | 677.8 | fixed |
