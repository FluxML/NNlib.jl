# Benchmarks

| Path | What it measures |
|---|---|
| `benchmarks.jl`, `runbenchmarks.jl` | the PkgBenchmark suite, comparing a target revision with a baseline. CI runs it on pull requests labelled `benchmark` |
| [`cpu_conv_vs_pytorch`](cpu_conv_vs_pytorch) | `conv` on the CPU, against PyTorch |
| [`cuda_attention`](cuda_attention) | attention on the GPU: cuDNN's fused kernel, the generic implementation, LuxLib |
| [`cuda_softmax`](cuda_softmax) | `softmax`/`logsoftmax` on the GPU, through Zygote |
| [`cudnn`](cudnn) | every cuDNN-accelerated operator: what NNlib dispatches to against the generic, fixed-function and graph implementations |

## Conventions for the GPU benchmarks

Each one is a folder with a script, its own `Project.toml` and a README.

- **Environment.** `[sources]` points NNlib at the checkout, so the script measures the
  code it sits next to. Other packages come from the registry.
- **Selection.** Command-line flags pick the operators, eltypes and sizes, so that one
  combination can be measured without rerunning the suite; `--list` shows what there is.
- **Timing.** The minimum over samples of `CUDA.@sync(op)`, from
  `BenchmarkTools.run(...; seconds, evals=1)`. The implementations being compared do the
  same work, output allocation included.
- **Correctness first.** Before timing, the implementations are checked against each
  other on a small input.
- **Failures are data.** A configuration an implementation cannot run is reported as
  `n/s` (with the reason on stderr) or `OOM`, and the run continues.
- **Output.** Tables are printed, and flushed, as each block completes. Results that
  feed a decision are also written as tab-separated values.
- **Results.** The raw output of a reference run is committed as
  `results_<gpu>_<library version>.txt` (and `.tsv`), and the README summarizes it:
  hardware, versions, the tables that matter and what they imply for NNlib.
