using Pkg
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

# --- Optional GPU package installation (main process, before workers start) ---
NNLIB_TEST_CUDA   && Pkg.add(["CUDA", "cuDNN"])
NNLIB_TEST_AMDGPU && Pkg.add("AMDGPU")
NNLIB_TEST_METAL  && Pkg.add("Metal")

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

# Wrap each `gpu/<backend>/*` test so the worker first loads that backend's setup
# (extra imports + the backend-specific `gputest`, which overrides the adapt-based
# one from `test_module.jl`). `include` runs the setup at the worker module's top
# level, so its `using` statements are valid there.
function wrap_ext_setup!(testsuite, gpu)
    setup = joinpath(@__DIR__, gpu, "test_setup.jl")
    for k in collect(keys(testsuite))
        startswith(k, "$gpu/") || continue
        inner = testsuite[k]
        testsuite[k] = quote
            include($setup)
            $inner
        end
    end
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

    # One entry per (shared suite, active backend), honoring per-backend skips.
    # `btype` is interpolated as a symbol and resolves in the worker because the
    # backend package is loaded by `init_code`.
    backends = []
    NNLIB_TEST_CPU    && push!(backends, (label="CPU",    btype=:CPU,         skips=Set{String}()))
    NNLIB_TEST_CUDA   && push!(backends, (label="CUDA",   btype=:CUDABackend, skips=Set(["scatter", "gather"])))
    NNLIB_TEST_AMDGPU && push!(backends, (label="AMDGPU", btype=:ROCBackend,  skips=Set{String}()))
    # Metal: shared suites stay disabled (matches the previous commented-out behavior).
    for s in SHARED_SUITES, b in backends
        s in b.skips && continue
        path = joinpath(@__DIR__, "common_testsuite", "$s.jl")
        fn = Symbol(s, "_testsuite")
        testsuite["common_testsuite/$s ($(b.label))"] = quote
            include($path)
            $fn($(b.btype))
        end
    end

    wrap_ext_setup!(testsuite, "gpu/cuda")
    wrap_ext_setup!(testsuite, "gpu/amdgpu")
    wrap_ext_setup!(testsuite, "gpu/metal")

    test_worker = Returns(nothing)
end

# --- init_code: runs in every worker (at module top level) before each test ---
# Load the active backend package here (top level) so backend types resolve and
# `adapt` dispatches; then bring in the shared imports and helpers.
init_code = quote
    $(NNLIB_TEST_CUDA   ? :(using CUDA, cuDNN) : nothing)
    $(NNLIB_TEST_AMDGPU ? :(using AMDGPU)      : nothing)
    $(NNLIB_TEST_METAL  ? :(using Metal)       : nothing)
    include($(joinpath(@__DIR__, "test_module.jl")))
end

runtests(NNlib, ARGS; testsuite, init_code, test_worker)
