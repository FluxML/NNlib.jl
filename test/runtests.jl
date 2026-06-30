using NNlib
using ParallelTestRunner

# --- Env flags ---

## Uncomment below to change the default test settings
# ENV["NNLIB_TEST_CUDA"] = "true"
# ENV["NNLIB_TEST_AMDGPU"] = "true"
# ENV["NNLIB_TEST_METAL"] = "true"
# ENV["NNLIB_TEST_CPU"] = "false"
# ENV["NNLIB_TEST_THREADED"] = "true"

const NNLIB_TEST_CPU      = get(ENV, "NNLIB_TEST_CPU",      "true")  == "true"
const NNLIB_TEST_CUDA     = get(ENV, "NNLIB_TEST_CUDA",     "false") == "true"
const NNLIB_TEST_AMDGPU   = get(ENV, "NNLIB_TEST_AMDGPU",   "false") == "true"
const NNLIB_TEST_METAL    = get(ENV, "NNLIB_TEST_METAL",    "false") == "true"
const NNLIB_TEST_THREADED = get(ENV, "NNLIB_TEST_THREADED", "false") == "true"

const NNLIB_TEST_ENZYME =   get(ENV, "NNLIB_TEST_ENZYME",   "true")  == "true" ||
                            (
                                VERSION <= v"1.13-" && # fails on nightly
                                !NNLIB_TEST_AMDGPU && !NNLIB_TEST_METAL && !NNLIB_TEST_CUDA && # TODO fails on GPU backends
                                !Sys.iswindows() # TODO fails on Windows
                            )

# Tests that exercise NNlib's multithreaded code paths (`@spawn` / `@threads`).
# The dedicated `NNLIB_TEST_THREADED` job runs *only* these, on multithreaded
# workers. Add or remove thread-sensitive tests here (paths as shown by `--list`).
const THREADED_TESTS = [
    "cpu/threading",
    "cpu/conv",
    "cpu/conv_bias_act",
    "cpu/batchedmul",
    "cpu/sampling",
    "common_testsuite/fold",
]

# GPU backends (CUDA/cuDNN, AMDGPU, Metal) are added to the test project beforehand
# via the `echo >>` step in .buildkite/pipeline.yml, so they are already present in
# the resolved environment when the active flag is set.

# --- Auto-discover all .jl test files (except runtests.jl) ---
testsuite = find_tests(@__DIR__)

# Library / setup files picked up by discovery that are not tests themselves.
delete!(testsuite, "test_module")
for gpu in ("gpu/cuda", "gpu/amdgpu", "gpu/metal")
    delete!(testsuite, "$gpu/test_setup")
end

# Every file in `common_testsuite/` only *defines* a `<name>_testsuite(Backend)`
# function; they are libraries driven explicitly below (one worker per (suite,
# backend)). Discover them from the directory and remove the bare entries.
const SHARED_SUITES = sort!([String(chopprefix(k, "common_testsuite/"))
                             for k in keys(testsuite) if startswith(k, "common_testsuite/")])
for s in SHARED_SUITES
    delete!(testsuite, "common_testsuite/$s")
end

if NNLIB_TEST_THREADED
    # Run `THREADED_TESTS` on workers with `NNLIB_TEST_NTHREADS` threads: keep the
    # plain discovered test files among them, and generate CPU entries for any
    # `common_testsuite/` suites in the list.
    filter!(((k, _),) -> k in THREADED_TESTS, testsuite)
    for t in THREADED_TESTS
        startswith(t, "common_testsuite/") || continue
        s = chopprefix(t, "common_testsuite/")
        testsuite["$t (CPU)"] = quote
            include($(joinpath(@__DIR__, "common_testsuite", "$s.jl")))
            $(Symbol(s, "_testsuite"))(CPU)
        end
    end

    nthreads = something(tryparse(Int, get(ENV, "NNLIB_TEST_NTHREADS", "2")), 2)
    @info "Running the multithreaded test subset on $nthreads-threaded workers."
    # `--threads` overrides the `JULIA_NUM_THREADS=1` that ParallelTestRunner sets.
    test_worker = _ -> addworker(; exeflags = ["--threads=$nthreads"])
else
    # GPU directories: keep only the active backend's files.
    !NNLIB_TEST_CUDA   && filter!(((k, _),) -> !startswith(k, "gpu/cuda"),   testsuite)
    !NNLIB_TEST_AMDGPU && filter!(((k, _),) -> !startswith(k, "gpu/amdgpu"), testsuite)
    !NNLIB_TEST_METAL  && filter!(((k, _),) -> !startswith(k, "gpu/metal"),  testsuite)
    # When CPU is disabled, drop the pure-CPU files (the shared suites are
    # re-added below for the active GPU backend).
    !NNLIB_TEST_CPU    && filter!(((k, _),) -> startswith(k, "gpu/"), testsuite)

    # One entry per (shared suite, active backend), honoring per-backend skips in the shared suites.
    # `btype` is interpolated as a symbol and resolves in the worker because the
    # backend package is loaded by `init_code`.
    backends = []
    NNLIB_TEST_CPU    && push!(backends, (label="CPU",    btype=:CPU,         skips=Set{String}()))
    NNLIB_TEST_CUDA   && push!(backends, (label="CUDA",   btype=:CUDABackend, skips=Set(["scatter", "gather"])))
    NNLIB_TEST_AMDGPU && push!(backends, (label="AMDGPU", btype=:ROCBackend,  skips=Set{String}()))
    # Metal: `spectral`/`rotation` need NNlib source work (scalar indexing, unsupported
    # imrotate kernel), and `activations` fails only on the complex-valued broadcasts.
    NNLIB_TEST_METAL  && push!(backends, (label="Metal",  btype=:MetalBackend,
        skips=Set(["activations", "rotation", "spectral"])))
    
    # Create a new entry in `testsuite` for each (suite, backend) pair.
    for s in SHARED_SUITES, b in backends
        s in b.skips && continue
        path = joinpath(@__DIR__, "common_testsuite", "$s.jl")
        fn = Symbol(s, "_testsuite")
        testsuite["common_testsuite/$s ($(b.label))"] = quote
            include($path)
            $fn($(b.btype))
        end
    end

    test_worker = Returns(nothing)
end

# --- init_code: evaluated at the top level of each test's sandbox module ---
# Bring in the shared imports + helpers, then (for a GPU run) the active backend's
# setup: its package, extra imports, and the backend-specific `gputest`. `include`
# runs at module top level, so the `using` statements inside are valid. The shared
# `common_testsuite/` suites use `test_gradients` (from test_module.jl) instead, so
# the two never collide.
init_code = quote
    include($(joinpath(@__DIR__, "test_module.jl")))
    const NNLIB_TEST_ENZYME = $NNLIB_TEST_ENZYME
    $(NNLIB_TEST_CUDA   ? :(include($(joinpath(@__DIR__, "gpu", "cuda",   "test_setup.jl")))) : nothing)
    $(NNLIB_TEST_AMDGPU ? :(include($(joinpath(@__DIR__, "gpu", "amdgpu", "test_setup.jl")))) : nothing)
    $(NNLIB_TEST_METAL  ? :(include($(joinpath(@__DIR__, "gpu", "metal",  "test_setup.jl")))) : nothing)
end

runtests(NNlib, ARGS; testsuite, init_code, test_worker)
