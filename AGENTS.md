# NNlib.jl

Fundamental neural-network primitives for Julia: activations, convolution/pooling,
attention, batched matrix ops, gather/scatter, dropout, normalization, softmax, and
upsampling. Primarily consumed by Flux.jl but usable standalone. Minimum Julia: 1.10.

## Layout

- `src/` — CPU implementations. Key files: `NNlib.jl` (module + exports),
  `activations.jl`, `attention.jl`, `conv.jl`, `pooling.jl`, `softmax.jl`,
  `batched/`, `gather.jl`/`scatter.jl`, `dropout.jl`, `normalization.jl`,
  `audio/` (STFT/mel/spectrogram).
  - `src/dim_helpers/` — `ConvDims`, `DenseConvDims`, `DepthwiseConvDims`, `PoolDims`.
  - `src/impl/` — direct and im2col conv/pooling kernels.
- `ext/` — package extensions for weak deps: CUDA, cuDNN (`NNlibCUDACUDNNExt`),
  AMDGPU, Metal, EnzymeCore, Mooncake (`NNlibMooncakeCUDAExt`), FFTW, ForwardDiff,
  SpecialFunctions. CPU fallbacks live in `src/`; backend code goes in the matching `ext/`.
- `docs/` — Documenter.jl source (`docs/src/`, `docs/make.jl`).
- `Project.toml` declares a workspace: `projects = ["test", "docs"]`.

## Conventions

- Functions: `lower_snake_case` (e.g. `dot_product_attention`). Types: `PascalCase`.
  In-place variants end with `!`.
- New activation: add to `src/activations.jl`, add to the `ACTIVATIONS` tuple (auto-exported),
  define the gradient (`@scalar_rule`/`rrule`), add value tests, document with an example.
- Gradients are defined with ChainRules (`rrule`). Enzyme/ForwardDiff/Mooncake support
  lives in their extensions.
- Threading: NNlib spawns tasks on divisible workloads (conv, etc.). Suppress with
  `NNlib.@disallow_spawns`. Spawning only happens when `Threads.nthreads(:default) > 1`.

## Testing

Tests use **ParallelTestRunner.jl** (not a flat `include` list). `test/runtests.jl`
auto-discovers files and runs them across worker processes.

- `test/cpu/` — CPU-only test files, discovered by filename.
- `test/common_testsuite/` — each file defines a `<name>_testsuite(Backend)` function,
  run once per (suite, active backend) pair.
- `test/gpu/{cuda,amdgpu,metal}/` — per-backend tests; only the active backend's files run.

Run the suite:

```julia
julia --project=test -e 'using Pkg; Pkg.test()'        # or from the test env
julia --project=test test/runtests.jl                  # direct
julia --project=test test/runtests.jl --list           # list discoverable test names
julia --project=test test/runtests.jl cpu/conv         # run a subset by name
```

Backends and modes are selected by env flags (default: CPU only, single-threaded):

- `NNLIB_TEST_CPU` (default `true`), `NNLIB_TEST_CUDA`, `NNLIB_TEST_AMDGPU`,
  `NNLIB_TEST_METAL` (default `false`).
- `NNLIB_TEST_THREADED=true` runs only the thread-sensitive subset (`THREADED_TESTS`
  in `runtests.jl`) on multithreaded workers; thread count via `NNLIB_TEST_NTHREADS`.

GPU backend packages are added to the test env by the `.buildkite/pipeline.yml` echo
step, not declared in `test/Project.toml`.

## CI

GitHub Actions in `.github/workflows/`: `ci.yml` (Linux/Windows/macOS × Julia versions ×
thread counts), `docs.yml`, `Downstream.yml`, `BenchmarkTrigger.yml`, `TagBot.yml`.
GPU CI runs on Buildkite (`.buildkite/pipeline.yml`).

## Links

- Docs: https://fluxml.ai/NNlib.jl/dev/
- Issues: https://github.com/FluxML/NNlib.jl/issues
