# Benchmarks

| Path | What it measures |
|---|---|
| [`cpu_conv_vs_pytorch`](cpu_conv_vs_pytorch) | `conv` on the CPU, against PyTorch |
| [`cpu_attention`](cpu_attention) | attention on the CPU: the fused implementation, the kernel of the GPU backends run on the CPU, the materialized implementation, PyTorch |
| [`cpu_softmax`](cpu_softmax) | `softmax` on the CPU: Julia tasks, the same loops as KernelAbstractions kernels, the generic implementation, PyTorch |
| [`cuda_attention`](cuda_attention) | attention on the GPU: cuDNN's fused kernel, NNlib's own, the materialized implementation, LuxLib, PyTorch |
| [`cudnn`](cudnn) | every cuDNN-accelerated operator: what NNlib dispatches to against the generic, fixed-function and graph implementations |
| [`others`](others) | `benchmarks.jl` and `runbenchmarks.jl`: the PkgBenchmark suite, comparing a target revision with a baseline. CI runs it on pull requests labelled `benchmark` |

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
- **Output.** Tables are printed, and flushed, as each block completes.
- **Results.** The raw output of a reference run is committed as
  `results_<gpu>_<library version>.txt`, and the README summarizes it: hardware,
  versions, the tables that matter and what they imply for NNlib.
