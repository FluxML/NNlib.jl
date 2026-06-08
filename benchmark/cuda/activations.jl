# Benchmarks for the removal of the custom cuDNN activation broadcast overloads
# (PR https://github.com/FluxML/NNlib.jl/pull/686).
#
# Before that PR, broadcasting `relu`, `σ`, `elu`, `tanh` over a `CuArray` was
# routed through cuDNN's `cudnnActivationForward!` by pirating
# `Base.materialize`/`materialize!`. The PR removes those overloads and relies on
# CUDA.jl's native broadcast instead. The claim is that for these
# memory-bandwidth-bound elementwise ops the native broadcast is just as fast
# (while also propagating NaNs correctly and avoiding method invalidations).
#
# This script measures both paths side by side in the same process so we get an
# apples-to-apples "pre vs post" comparison without checking out two commits:
#
#   * "native"  -> `f.(x)` / `f.(x)` into `dst`     (the post-PR behaviour)
#   * "cudnn"   -> `cudnnActivationForward!(dst, x)` (the pre-PR behaviour)
#
# Run with:
#   julia --project=benchmark/cuda benchmark/cuda/activations.jl

using CUDA
using cuDNN
using NNlib
using BenchmarkTools
using Printf

using cuDNN: cudnnActivationForward!,
             CUDNN_ACTIVATION_TANH, CUDNN_ACTIVATION_SIGMOID,
             CUDNN_ACTIVATION_ELU, CUDNN_ACTIVATION_RELU

CUDA.allowscalar(false)

# (name, native activation fn, fast variant or `nothing`, cuDNN mode) for the four
# activations that used to have a cuDNN-routed broadcast overload. `tanh_fast` and
# `sigmoid_fast` are NNlib's faster approximations (relu/elu have no fast variant);
# they were never routed through cuDNN, so they are an alternative native path.
const ACTS = [
    ("relu",    relu,    nothing,             CUDNN_ACTIVATION_RELU),
    ("sigmoid", NNlib.σ, NNlib.sigmoid_fast,  CUDNN_ACTIVATION_SIGMOID),
    ("elu",     elu,     nothing,             CUDNN_ACTIVATION_ELU),
    ("tanh",    tanh,    NNlib.tanh_fast,     CUDNN_ACTIVATION_TANH),
]

# cuDNN supports Float16/Float32/Float64 for activations.
const ELTYPES = (Float16, Float32, Float64)

# A few representative shapes: a square matrix (as in the CPU benchmarks) and a
# couple of conv-like 4D tensors of growing size.
const SIZES = [
    (1024, 1024),
    (224, 224, 3, 32),
    (56, 56, 64, 64),
]

# Median time in seconds for a GPU-synced benchmark of `f`.
function gpu_time(f)
    b = @benchmark CUDA.@sync($f()) samples=1000 evals=1 seconds=3
    return minimum(b).time / 1e9  # seconds, use the minimum to reduce noise
end

# helper: "-" when no fast variant exists / time is missing
fmt(t) = t === nothing ? "       -" : @sprintf("%8.2f", t * 1e6)
ratio(a, b) = (a === nothing || b === nothing) ? "      -" : @sprintf("%6.2fx", a / b)

function run_suite()
    @printf("%-9s %-8s %-16s %8s %8s %8s   %7s %7s\n",
            "act", "eltype", "size", "native", "fast", "cudnn", "nat/cu", "fst/cu")
    println("-"^80)
    results = Tuple[]
    for (name, act, fast, mode) in ACTS, et in ELTYPES, sz in SIZES
        x   = CUDA.randn(et, sz...)
        dst = similar(x)

        # warm up / compile all paths
        act.(x)
        broadcast!(act, dst, x)
        cudnnActivationForward!(dst, x; mode)
        fast === nothing || fast.(x)

        # out-of-place
        t_native_oop = gpu_time(() -> act.(x))
        t_fast_oop   = fast === nothing ? nothing : gpu_time(() -> fast.(x))
        t_cudnn_oop  = gpu_time(() -> cudnnActivationForward!(similar(x), x; mode))
        # in-place
        t_native_ip  = gpu_time(() -> broadcast!(act, dst, x))
        t_fast_ip    = fast === nothing ? nothing : gpu_time(() -> broadcast!(fast, dst, x))
        t_cudnn_ip   = gpu_time(() -> cudnnActivationForward!(dst, x; mode))

        szstr = join(sz, "x")
        @printf("%-9s %-8s %-16s %s %s %s   %s %s   (out-of-place)\n",
                name, et, szstr, fmt(t_native_oop), fmt(t_fast_oop), fmt(t_cudnn_oop),
                ratio(t_native_oop, t_cudnn_oop), ratio(t_fast_oop, t_cudnn_oop))
        @printf("%-9s %-8s %-16s %s %s %s   %s %s   (in-place)\n",
                "", "", "", fmt(t_native_ip), fmt(t_fast_ip), fmt(t_cudnn_ip),
                ratio(t_native_ip, t_cudnn_ip), ratio(t_fast_ip, t_cudnn_ip))
        push!(results, (name, et, szstr,
                        t_native_oop, t_fast_oop, t_cudnn_oop,
                        t_native_ip, t_fast_ip, t_cudnn_ip))
    end
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    @info "CUDA / cuDNN environment" CUDA.device() cuDNN.version()
    run_suite()
end
