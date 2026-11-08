# NNlib.jl

Fundamental neural-network primitives for Julia: activations, convolution/pooling,
attention, batched matrix ops, gather/scatter, dropout, normalization, softmax, and
upsampling. Primarily consumed by Flux.jl but usable standalone. Minimum Julia: 1.10.

## Layout

- `src/` — CPU implementations. Key files: `NNlib.jl` (module + exports),
  `activations.jl`, `attention.jl`, `conv.jl`, `pooling.jl`, `softmax.jl`,
  `batched/`, `gather.jl`/`scatter.jl`, `dropout.jl`, `normalization.jl`,
  `causal_conv1d.jl`/`sequence_scans.jl` (Mamba-2 and gated delta rule scans, GPU kernels
  in `sequence_scans_kernels.jl`),
  `audio/` (STFT/mel/spectrogram).
  - `src/dim_helpers/` — `ConvDims`, `DenseConvDims`, `DepthwiseConvDims`, `PoolDims`.
  - `src/impl/` — direct and im2col conv/pooling kernels.
- `ext/` — package extensions for weak deps: CUDA, cuDNN (`NNlibCUDACUDNNExt`),
  AMDGPU, Metal, EnzymeCore, Mooncake (`NNlibMooncakeExt`),
  FFTW, ForwardDiff, SpecialFunctions. CPU fallbacks live in `src/`; backend code goes
  in the matching `ext/`.
- `docs/` — Documenter.jl source (`docs/src/`, `docs/make.jl`).
- `Project.toml` declares a workspace: `projects = ["test", "docs"]`.

## Conventions

- Functions: `lower_snake_case` (e.g. `dot_product_attention`). Types: `PascalCase`.
  In-place variants end with `!`.
- New activation: add to `src/activations.jl`, add to the `ACTIVATIONS` tuple (auto-exported),
  define the gradient (`@scalar_rule`/`rrule`), add value tests, document with an example.
- Gradients are defined with ChainRules (`rrule`). Enzyme/ForwardDiff/Mooncake support
  lives in their extensions.
- Mooncake rules are in `ext/NNlibMooncakeExt/rules.jl`, a port of Mooncake's own
  `MooncakeNNlibExt`. While Mooncake still ships that extension the file is not loaded
  (`MOONCAKE_SHIPS_RULES`), and `test/test_module.jl` swaps it in so the tests cover it.
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

### Keep new tests cheap

CI time goes into compilation: each `test_gradients` call compiles FiniteDifferences,
Zygote and Enzyme for its function, and each EnzymeTestUtils/Mooncake `test_rule` case
compiles anew (seconds per case). So when adding tests:

- Test an AD rule once, at a small shape, not again for every spatial rank, element type,
  activation or keyword combination, unless the code differs between them. Loop over
  configurations in forward-pass checks, which are cheap.
- Second derivatives: ForwardDiff over Zygote; Zygote over Zygote for a representative only.
- `JULIA_TEST_VERBOSE=true julia --project=test test/runtests.jl cpu/conv` prints the time
  of each testset (Julia 1.13). A file over ~5 min locally sets the wall time of its CI job:
  trim it or split it.

## CI

GitHub Actions in `.github/workflows/`: `ci.yml` (Linux/Windows/macOS × Julia versions ×
thread counts), `docs.yml`, `Downstream.yml`, `BenchmarkTrigger.yml`, `TagBot.yml`.
Enzyme is tested on Linux only (`NNLIB_TEST_ENZYME=false` on Windows and macOS).
GPU CI runs on Buildkite (`.buildkite/pipeline.yml`). Its CUDA agents have 2 CPU cores, so
the same advice applies. CUDA runs on Julia 1 for pull requests, and on Julia 1.10 too on
master. The CUDA Enzyme job runs only `gpu/cuda/normalization_enzyme`.

## Links

- Docs: https://fluxml.ai/NNlib.jl/dev/
- Issues: https://github.com/FluxML/NNlib.jl/issues
