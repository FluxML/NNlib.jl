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
their gradients, the `relu`/`tanh`/`σ`/`elu` broadcasts, `softmax` and `logsoftmax` with
their gradients along the first dimension and along the channels, convolution and its two
gradients. Attention has its own benchmark in [`../cuda_attention`](../cuda_attention).

## Where NNlib stands

From the run below (RTX 5090, cuDNN 9.26), slowdowns relative to the fastest candidate:

| Operator | NNlib uses | Why |
|---|---|---|
| batchnorm: training, inference, gradient | `fixed` | `graph` is 1.6–3.0× slower in training and 1.5–2.6× in the gradient, and has no engine for inference or for a number of channels that is not a multiple of 4 (nor for `Float64`, which this run does not include). `generic` is 2–21× slower |
| `maxpool`, `meanpool` and their gradients | `fixed`; `graph` only for asymmetric padding, which `fixed` cannot do | `graph` is 1.06–1.19× slower forward and up to 2.6× in `∇maxpool` |
| `relu`, `tanh`, `σ`, `elu` broadcasts | `fixed` | `generic` is 1.2–5.6× slower. cuDNN has no engine for a graph made of one pointwise operation |
| `softmax`, `logsoftmax` and their gradients over leading dimensions | `fixed` (instance mode) | `generic` is 2.0–23× slower, long softmax axes included |
| the same over other dimensions | `generic` | **worth revisiting**: right for a long softmax axis, wrong for a short one. `fixed` (channel mode) is 2.3–16× faster forward and 1.3–9.9× in the gradient with up to 128 channels, 3.3–4.6× slower with 1000, and 38–48× slower in the gradient with 10000. See the note below |
| `conv` and its gradients | `graph` | `fixed` is up to 2.3× slower below the largest size and 4–8% slower at the largest. The exceptions are in `Float32`: at the largest size `fixed` is 8% faster for `∇conv_data` and 14% for `∇conv_filter`, and with 30 channels 8% for `∇conv_filter` and level with `graph` for `conv`. `graph` also takes asymmetric padding natively, though NNlib still pads the input for `∇conv_filter`, whose graph version returns wrong values there with some engines |

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
| `--sizes` | `small`,`medium`,`large`,`odd-C`, and for softmax alone `long-W`,`long-C`,`long-WC` | all |
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

The softmax family also runs on matrices with a long softmax axis, laid out as
`(rows, 1, columns, 1)` so that the first dimension and the channels are their two axes:

```
long-W   10000×1×100×1
long-C   100×1×10000×1
long-WC  1000×1×1000×1
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
- The `Zygote ∇softmax` and `Zygote ∇logsoftmax` rows time the pullback Zygote builds
  from NNlib's `rrule`, which is how a training step reaches the gradient. They have no
  candidates, and are to be read against the `nnlib` column of the row above them.

`nnlib` can be slower than the candidate it dispatches to: that difference is NNlib's
wrapper. In this run it shows in `∇batchnorm` (cuDNN recomputing the batch statistics,
19.0 vs 11.3 μs at the medium `Float16` size) and in pooling, whose allocating functions
zero the output before cuDNN overwrites it.

### Adding an operator

Write a `<family>_ops(T, sz)` function returning its `Op`s, each a name and the thunks of
its implementations, and register it in `FAMILIES`. The candidates call cuDNN.jl through
qualified names inside functions, so that one whose API has gone shows up as `n/s` rather
than breaking the script.

## Results — RTX 5090 (Blackwell, sm_120), Julia 1.13.1, CUDA.jl 6.4.2 / cuDNN.jl 6.4.2 (cuDNN 9.26)

Raw output: [`results_rtx5090_cudnn926.txt`](results_rtx5090_cudnn926.txt). The tables
show the medium and large sizes, and for softmax the long axes as well; times in μs, lower
is better, with the fastest candidate in bold.

### Batch normalization

| operator | eltype | size | nnlib | generic | fixed | graph | fastest |
|---|---|---|--:|--:|--:|--:|---|
| `batchnorm (training)` | Float16 | 32×32×64×16 | 9.1 | 78.7 | **8.9** | 24.0 | fixed |
| `batchnorm (inference)` | Float16 | 32×32×64×16 | 7.7 | 44.8 | **7.9** | n/s | fixed |
| `∇batchnorm (training)` | Float16 | 32×32×64×16 | 19.0 | 146.8 | **11.3** | 24.3 | fixed |
| `batchnorm (training)` | Float16 | 64×64×128×32 | 57.2 | 710.6 | **57.4** | 92.6 | fixed |
| `batchnorm (inference)` | Float16 | 64×64×128×32 | 22.7 | 474.8 | **22.7** | n/s | fixed |
| `∇batchnorm (training)` | Float16 | 64×64×128×32 | 101.4 | 1449.9 | **85.8** | 154.0 | fixed |
| `batchnorm (training)` | Float32 | 32×32×64×16 | 12.0 | 59.8 | **12.1** | 26.1 | fixed |
| `batchnorm (inference)` | Float32 | 32×32×64×16 | 8.2 | 25.4 | **8.1** | n/s | fixed |
| `∇batchnorm (training)` | Float32 | 32×32×64×16 | 19.0 | 117.4 | **16.4** | 27.4 | fixed |
| `batchnorm (training)` | Float32 | 64×64×128×32 | 114.5 | 467.2 | **116.1** | 215.8 | fixed |
| `batchnorm (inference)` | Float32 | 64×64×128×32 | 93.9 | 220.2 | **93.8** | n/s | fixed |
| `∇batchnorm (training)` | Float32 | 64×64×128×32 | 268.3 | 1066.0 | **242.8** | 354.9 | fixed |

With 30 channels (`odd-C`) `graph` is `n/s` for all three, and inference is `n/s` at
every size.

### Pooling

| operator | eltype | size | nnlib | generic | fixed | graph | fastest |
|---|---|---|--:|--:|--:|--:|---|
| `maxpool` | Float16 | 32×32×64×16 | 8.5 | - | **7.1** | 8.3 | fixed |
| `∇maxpool` | Float16 | 32×32×64×16 | 10.0 | - | **8.7** | 13.1 | fixed |
| `meanpool` | Float16 | 32×32×64×16 | 8.6 | - | **7.4** | 8.4 | fixed |
| `∇meanpool` | Float16 | 32×32×64×16 | 9.9 | - | **8.5** | 10.1 | fixed |
| `maxpool` | Float16 | 64×64×128×32 | 19.3 | - | **17.3** | 19.4 | fixed |
| `∇maxpool` | Float16 | 64×64×128×32 | 52.8 | - | **33.7** | 87.1 | fixed |
| `meanpool` | Float16 | 64×64×128×32 | 20.7 | - | **18.6** | 19.8 | fixed |
| `∇meanpool` | Float16 | 64×64×128×32 | 47.0 | - | **36.7** | 38.1 | fixed |
| `maxpool` | Float32 | 32×32×64×16 | 8.6 | - | **7.3** | 8.4 | fixed |
| `∇maxpool` | Float32 | 32×32×64×16 | 10.6 | - | **9.2** | 13.0 | fixed |
| `meanpool` | Float32 | 32×32×64×16 | 8.7 | - | **7.4** | 8.3 | fixed |
| `∇meanpool` | Float32 | 32×32×64×16 | 10.0 | - | **8.6** | 10.2 | fixed |
| `maxpool` | Float32 | 64×64×128×32 | 50.4 | - | **18.5** | 19.8 | fixed |
| `∇maxpool` | Float32 | 64×64×128×32 | 143.8 | - | **119.0** | 129.0 | fixed |
| `meanpool` | Float32 | 64×64×128×32 | 51.1 | - | **18.4** | 19.6 | fixed |
| `∇meanpool` | Float32 | 64×64×128×32 | 64.9 | - | **39.5** | 39.7 | fixed |

### Activations

`tanh` and `elu` time like the two shown.

| operator | eltype | size | nnlib | generic | fixed | graph | fastest |
|---|---|---|--:|--:|--:|--:|---|
| `relu.(x)` | Float16 | 32×32×64×16 | 7.0 | 15.9 | **7.2** | n/s | fixed |
| `σ.(x)` | Float16 | 32×32×64×16 | 7.5 | 16.8 | **7.6** | n/s | fixed |
| `relu.(x)` | Float16 | 64×64×128×32 | 22.6 | 129.1 | **23.0** | n/s | fixed |
| `σ.(x)` | Float16 | 64×64×128×32 | 26.9 | 141.6 | **27.0** | n/s | fixed |
| `relu.(x)` | Float32 | 32×32×64×16 | 7.5 | 15.9 | **7.7** | n/s | fixed |
| `σ.(x)` | Float32 | 32×32×64×16 | 7.8 | 16.3 | **8.1** | n/s | fixed |
| `relu.(x)` | Float32 | 64×64×128×32 | 90.3 | 138.8 | **91.4** | n/s | fixed |
| `σ.(x)` | Float32 | 64×64×128×32 | 90.6 | 143.7 | **90.6** | n/s | fixed |

### Softmax

The `Zygote` rows are left out: at every size they are within 5% (at most 4 μs) of the
`nnlib` gradient above them, so the pullback costs what the kernel it reaches costs.

Along the first dimension, where NNlib uses cuDNN's instance mode:

| operator | eltype | size | nnlib | generic | fixed | graph | fastest |
|---|---|---|--:|--:|--:|--:|---|
| `softmax (dims=1)` | Float16 | 32×32×64×16 | 7.9 | 85.6 | **7.8** | - | fixed |
| `∇softmax (dims=1)` | Float16 | 32×32×64×16 | 7.6 | 49.5 | **7.8** | - | fixed |
| `logsoftmax (dims=1)` | Float16 | 32×32×64×16 | 7.7 | 85.2 | **7.9** | - | fixed |
| `∇logsoftmax (dims=1)` | Float16 | 32×32×64×16 | 7.7 | 38.2 | **7.8** | - | fixed |
| `softmax (dims=1)` | Float16 | 64×64×128×32 | 25.3 | 592.6 | **25.6** | - | fixed |
| `∇softmax (dims=1)` | Float16 | 64×64×128×32 | 53.6 | 458.5 | **52.9** | - | fixed |
| `logsoftmax (dims=1)` | Float16 | 64×64×128×32 | 25.9 | 598.7 | **26.1** | - | fixed |
| `∇logsoftmax (dims=1)` | Float16 | 64×64×128×32 | 53.2 | 319.7 | **55.4** | - | fixed |
| `softmax (dims=1)` | Float32 | 32×32×64×16 | 7.7 | 84.3 | **7.9** | - | fixed |
| `∇softmax (dims=1)` | Float32 | 32×32×64×16 | 7.8 | 49.4 | **8.0** | - | fixed |
| `logsoftmax (dims=1)` | Float32 | 32×32×64×16 | 7.7 | 85.1 | **7.9** | - | fixed |
| `∇logsoftmax (dims=1)` | Float32 | 32×32×64×16 | 8.0 | 37.6 | **8.0** | - | fixed |
| `softmax (dims=1)` | Float32 | 64×64×128×32 | 89.1 | 621.7 | **75.3** | - | fixed |
| `∇softmax (dims=1)` | Float32 | 64×64×128×32 | 126.4 | 481.9 | **128.0** | - | fixed |
| `logsoftmax (dims=1)` | Float32 | 64×64×128×32 | 75.7 | 631.8 | **90.2** | - | fixed |
| `∇logsoftmax (dims=1)` | Float32 | 64×64×128×32 | 127.0 | 334.1 | **130.6** | - | fixed |

Along the channels, where NNlib uses its generic kernel:

| operator | eltype | size | nnlib | generic | fixed | graph | fastest |
|---|---|---|--:|--:|--:|--:|---|
| `softmax (dims=3)` | Float16 | 32×32×64×16 | 83.1 | 83.3 | **11.2** | - | fixed |
| `∇softmax (dims=3)` | Float16 | 32×32×64×16 | 48.8 | 48.9 | **9.8** | - | fixed |
| `logsoftmax (dims=3)` | Float16 | 32×32×64×16 | 84.3 | 84.4 | **9.9** | - | fixed |
| `∇logsoftmax (dims=3)` | Float16 | 32×32×64×16 | 38.3 | 38.3 | **18.0** | - | fixed |
| `softmax (dims=3)` | Float16 | 64×64×128×32 | 802.1 | 800.0 | **53.3** | - | fixed |
| `∇softmax (dims=3)` | Float16 | 64×64×128×32 | 559.6 | 560.0 | **56.7** | - | fixed |
| `logsoftmax (dims=3)` | Float16 | 64×64×128×32 | 806.2 | 804.8 | **51.1** | - | fixed |
| `∇logsoftmax (dims=3)` | Float16 | 64×64×128×32 | 427.6 | 427.4 | **65.8** | - | fixed |
| `softmax (dims=3)` | Float32 | 32×32×64×16 | 82.2 | 83.1 | **20.6** | - | fixed |
| `∇softmax (dims=3)` | Float32 | 32×32×64×16 | 48.5 | 48.8 | **18.0** | - | fixed |
| `logsoftmax (dims=3)` | Float32 | 32×32×64×16 | 83.1 | 83.7 | **19.7** | - | fixed |
| `∇logsoftmax (dims=3)` | Float32 | 32×32×64×16 | 37.5 | 37.0 | **18.0** | - | fixed |
| `softmax (dims=3)` | Float32 | 64×64×128×32 | 834.2 | 833.2 | **167.9** | - | fixed |
| `∇softmax (dims=3)` | Float32 | 64×64×128×32 | 588.1 | 588.5 | **216.9** | - | fixed |
| `logsoftmax (dims=3)` | Float32 | 64×64×128×32 | 834.4 | 834.1 | **168.8** | - | fixed |
| `∇logsoftmax (dims=3)` | Float32 | 64×64×128×32 | 439.1 | 440.7 | **174.5** | - | fixed |

The long softmax axes, along the channels again, by number of channels (100, 1000, 10000).
`Float16` has the same winner in every row:

| operator | eltype | size | nnlib | generic | fixed | graph | fastest |
|---|---|---|--:|--:|--:|--:|---|
| `softmax (dims=3)` | Float32 | 10000×1×100×1 | 79.8 | 82.0 | **35.9** | - | fixed |
| `∇softmax (dims=3)` | Float32 | 10000×1×100×1 | 43.9 | 43.8 | **24.4** | - | fixed |
| `logsoftmax (dims=3)` | Float32 | 10000×1×100×1 | 82.1 | 80.9 | **34.5** | - | fixed |
| `∇logsoftmax (dims=3)` | Float32 | 10000×1×100×1 | 36.3 | 36.1 | **24.6** | - | fixed |
| `softmax (dims=3)` | Float32 | 1000×1×1000×1 | 92.6 | **95.7** | 323.1 | - | generic |
| `∇softmax (dims=3)` | Float32 | 1000×1×1000×1 | 51.9 | **51.8** | 188.3 | - | generic |
| `logsoftmax (dims=3)` | Float32 | 1000×1×1000×1 | 94.8 | **92.8** | 308.9 | - | generic |
| `∇logsoftmax (dims=3)` | Float32 | 1000×1×1000×1 | 42.5 | **42.4** | 190.8 | - | generic |
| `softmax (dims=3)` | Float32 | 100×1×10000×1 | 87.6 | 83.9 | **65.8** | - | fixed |
| `∇softmax (dims=3)` | Float32 | 100×1×10000×1 | 45.9 | **45.6** | 1753.1 | - | generic |
| `logsoftmax (dims=3)` | Float32 | 100×1×10000×1 | 84.7 | 89.6 | **46.3** | - | fixed |
| `∇logsoftmax (dims=3)` | Float32 | 100×1×10000×1 | 40.6 | **39.5** | 1770.1 | - | generic |

NNlib uses its generic kernel for a softmax that is not over leading dimensions since
[#513](https://github.com/FluxML/NNlib.jl/issues/513), where cuDNN's channel mode was
much slower on the backward pass for a long softmax axis. It still is: with 10000 channels
its gradient takes 1.8 ms against 39–46 μs, though its forward pass is 1.1–1.9× faster,
and with 1000 channels it is 3.3–4.6× slower in both directions. With up to 128 channels
it wins instead, by 2.3–16× forward and 1.3–9.9× in the gradient, the 10000 positions of
`long-W` included. Routing on the length of the softmax axis would get both; these sizes
place the crossover between 128 and 1000 channels, no closer.

### Convolution

3×3 kernel, padding 1, as many output as input channels.

| operator | eltype | size | nnlib | generic | fixed | graph | fastest |
|---|---|---|--:|--:|--:|--:|---|
| `conv` | Float16 | 32×32×64×16 | 24.6 | - | 44.0 | **25.2** | graph |
| `∇conv_data` | Float16 | 32×32×64×16 | 24.5 | - | 38.8 | **24.1** | graph |
| `∇conv_filter` | Float16 | 32×32×64×16 | 30.5 | - | 44.8 | **30.3** | graph |
| `conv` | Float16 | 64×64×128×32 | 292.6 | - | 315.7 | **292.9** | graph |
| `∇conv_data` | Float16 | 64×64×128×32 | 261.2 | - | 279.7 | **260.8** | graph |
| `∇conv_filter` | Float16 | 64×64×128×32 | 393.2 | - | 412.5 | **392.8** | graph |
| `conv` | Float32 | 32×32×64×16 | 32.4 | - | 41.0 | **32.2** | graph |
| `∇conv_data` | Float32 | 32×32×64×16 | 32.3 | - | 39.9 | **32.1** | graph |
| `∇conv_filter` | Float32 | 32×32×64×16 | 49.1 | - | 61.1 | **49.0** | graph |
| `conv` | Float32 | 64×64×128×32 | 491.3 | - | 512.5 | **491.6** | graph |
| `∇conv_data` | Float32 | 64×64×128×32 | 547.2 | - | **505.5** | 548.0 | fixed |
| `∇conv_filter` | Float32 | 64×64×128×32 | 676.8 | - | **581.2** | 676.2 | fixed |
