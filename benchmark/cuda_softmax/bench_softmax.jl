# GPU softmax / logsoftmax benchmark — cuDNN vs the custom (CPU-algorithm) kernels.
#
# Compares, on the GPU, the specialized cuDNN softmax against NNlib's generic
# array kernels (the ones used on the CPU):
#   * forward  — cuDNN `cudnnSoftmaxForward!`  vs the `exp.(x .- max)/sum` kernel
#   * backward — cuDNN `cudnnSoftmaxBackward`  vs the broadcast ∇ rule
# for both `softmax` and `logsoftmax`, and for `dims=1` and `dims=2`.
# Reproduces / extends FluxML/NNlib.jl#513.
#
# Also includes LogExpFunctions.softmax (forward + its ChainRules gradient) as a
# third contender for `softmax` (it has no `logsoftmax`). Its math matches NNlib's
# custom kernels, so it is a cross-check / second data point on the same approach.
#
# The cuDNN routines are called DIRECTLY here (not through NNlib's `∇softmax!`),
# so every shape really goes through cuDNN — NNlib's `softmaxdims` heuristic
# otherwise diverts some shapes to the generic kernel, which would hide the
# comparison.
#
# Run:
#     julia --project=. bench_softmax.jl
# Pin a GPU (UUID form is robust when other cards on the box are unavailable):
#     CUDA_VISIBLE_DEVICES=GPU-<uuid> julia --project=. bench_softmax.jl

using CUDA, cuDNN, NNlib, LogExpFunctions, BenchmarkTools, Printf
using cuDNN: CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_CHANNEL,
             cudnnSoftmaxForward!, cudnnSoftmaxBackward, cudnnTensorDescriptor,
             scalingParameter, handle

# ---- cuDNN path (called directly, always) -------------------------------------
# cuDNN softmax over an integer axis `dims` reshapes the array to
# (1, stride, dimsize, batchsize) and runs CHANNEL-mode softmax over `dimsize`.
function cudnn_shape(x, dims::Int)
    stride    = prod(size(x)[1:dims-1]; init = 1)
    dimsize   = size(x, dims)
    batchsize = length(x) ÷ (stride * dimsize)
    (1, stride, dimsize, batchsize)
end

function cudnn_softmax!(out, x; dims, algo)
    s = cudnn_shape(x, dims)
    cudnnSoftmaxForward!(reshape(out, s), reshape(x, s); mode = CUDNN_SOFTMAX_MODE_CHANNEL, algo)
    out
end
function cudnn_∇softmax!(dx, dy, x, y; dims, algo)
    s = cudnn_shape(x, dims)
    xDesc = cudnnTensorDescriptor(reshape(x, s))
    R = eltype(x); alpha, beta = scalingParameter(R, 1), scalingParameter(R, 0)
    cudnnSoftmaxBackward(handle(), algo, CUDNN_SOFTMAX_MODE_CHANNEL,
        alpha, xDesc, reshape(y, s), xDesc, reshape(dy, s), beta, xDesc, reshape(dx, s))
    dx
end

# ---- custom path = NNlib's generic (CPU) kernels ------------------------------
# Forward mirrors NNlib's `softmax!`/`logsoftmax!` all-finite fast path; backward
# is the broadcast ∇ rule (`_∇softmax!`/`_∇logsoftmax!`).
fast_maximum(x; dims) = reduce(max, x; dims, init = float(eltype(x))(-Inf))
function custom_softmax!(out, x; dims)
    max_ = fast_maximum(x; dims)
    out .= exp.(x .- max_)
    out ./= sum(out; dims)
end
function custom_logsoftmax!(out, x; dims)
    max_ = fast_maximum(x; dims)
    out .= x .- max_
    out .-= log.(sum(exp.(out); dims))
end
custom_∇softmax!(dx, dy, x, y; dims)    = (dx .= y .* (dy .- sum(dy .* y; dims)))
custom_∇logsoftmax!(dx, dy, x, y; dims) = (dx .= dy .- sum(dy; dims) .* exp.(y))

# ---- LogExpFunctions path (softmax only; no logsoftmax) -----------------------
# Forward is its public `softmax!`; gradient is the math from its ChainRules
# rrule pullback (`Ω.*Ω̄ .- Ω .* sum(Ω.*Ω̄; dims)`), materializing the temporary
# `Ω.*Ω̄` exactly as the rrule does.
lef_softmax!(out, x; dims) = LogExpFunctions.softmax!(out, x; dims)
lef_∇softmax!(out, dy, y; dims) = (tmp = y .* dy; out .= tmp .- y .* sum(tmp; dims))

# Array shapes to benchmark. softmax acts along axis `dims`, so for dims=1 the
# softmax-vector length is size[1] and for dims=2 it is size[2]; the remaining
# axes are batch (e.g. (256,10,32) with dims=1 = 320 vectors of length 256).
# The length of the softmax axis is what drives the cuDNN-vs-custom crossover.
const SIZES = [(256,10,32), (256,1000), (1000,1000), (100,10000),
               (10000,100), (32000,64), (1000,128)]

const A, L = CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_LOG

function bench_all(sizes, dims)
    map(sizes) do s
        x = CUDA.randn(Float32, s...); dy = CUDA.randn(Float32, s...); o = similar(x)
        ys = custom_softmax!(similar(x), x; dims); yl = custom_logsoftmax!(similar(x), x; dims)
        (; size = s,
          fsm_c = (@belapsed CUDA.@sync cudnn_softmax!($o,$x;dims=$dims,algo=$A))*1e6,
          fsm_g = (@belapsed CUDA.@sync custom_softmax!($o,$x;dims=$dims))*1e6,
          fsm_l = (@belapsed CUDA.@sync lef_softmax!($o,$x;dims=$dims))*1e6,
          fls_c = (@belapsed CUDA.@sync cudnn_softmax!($o,$x;dims=$dims,algo=$L))*1e6,
          fls_g = (@belapsed CUDA.@sync custom_logsoftmax!($o,$x;dims=$dims))*1e6,
          bsm_c = (@belapsed CUDA.@sync cudnn_∇softmax!($o,$dy,$x,$ys;dims=$dims,algo=$A))*1e6,
          bsm_g = (@belapsed CUDA.@sync custom_∇softmax!($o,$dy,$x,$ys;dims=$dims))*1e6,
          bsm_l = (@belapsed CUDA.@sync lef_∇softmax!($o,$dy,$ys;dims=$dims))*1e6,
          bls_c = (@belapsed CUDA.@sync cudnn_∇softmax!($o,$dy,$x,$yl;dims=$dims,algo=$L))*1e6,
          bls_g = (@belapsed CUDA.@sync custom_∇logsoftmax!($o,$dy,$x,$yl;dims=$dims))*1e6)
    end
end

function check_correctness()
    for dims in (1, 2)
        x = CUDA.randn(Float32, 512, 700); dy = CUDA.randn(Float32, 512, 700)
        ys = custom_softmax!(similar(x), x; dims); yl = custom_logsoftmax!(similar(x), x; dims)
        f(a, b) = maximum(abs.(Array(a) .- Array(b)))
        e1 = f(cudnn_softmax!(similar(x), x; dims, algo=A), ys)
        e2 = f(cudnn_softmax!(similar(x), x; dims, algo=L), yl)
        e3 = f(cudnn_∇softmax!(similar(x), dy, x, ys; dims, algo=A), custom_∇softmax!(similar(x), dy, x, ys; dims))
        e4 = f(cudnn_∇softmax!(similar(x), dy, x, yl; dims, algo=L), custom_∇logsoftmax!(similar(x), dy, x, yl; dims))
        e5 = f(lef_softmax!(similar(x), x; dims), ys)
        e6 = f(lef_∇softmax!(similar(x), dy, ys; dims), custom_∇softmax!(similar(x), dy, x, ys; dims))
        @printf("correctness dims=%d  fwd sm=%.0e ls=%.0e  bwd sm=%.0e ls=%.0e  LEF fwd=%.0e grad=%.0e\n",
                dims, e1, e2, e3, e4, e5, e6)
    end
end

# 2-way table (cuDNN vs NNlib custom) — used for logsoftmax
show_table(title, res, c, g) = begin
    println("\n", title)
    @printf("%-15s %10s %10s %12s\n", "size", "cuDNN", "custom", "cuDNN/custom")
    for r in res
        @printf("%-15s %8.1f %8.1f %10.2fx\n", string(r.size),
                getfield(r, c), getfield(r, g), getfield(r, c) / getfield(r, g))
    end
end

# 3-way table (cuDNN vs NNlib custom vs LogExpFunctions) — used for softmax
show_table3(title, res, c, g, l) = begin
    println("\n", title)
    @printf("%-15s %9s %9s %9s %12s\n", "size", "cuDNN", "NNlib", "LEF", "cuDNN/NNlib")
    for r in res
        @printf("%-15s %8.1f %8.1f %8.1f %10.2fx\n", string(r.size),
                getfield(r, c), getfield(r, g), getfield(r, l), getfield(r, c) / getfield(r, g))
    end
end

function main()
    @assert CUDA.functional() "CUDA is not functional"
    println("GPU:   ", CUDA.name(CUDA.device()))
    println("cuDNN: ", cuDNN.version())
    check_correctness()
    for dims in (1, 2)
        res = bench_all(SIZES, dims)
        println("\n", "#"^60, "\n### dims=$dims   (times in µs, Float32)\n", "#"^60)
        show_table3("FORWARD  softmax     (NNlib/LEF = exp/sum kernels)",  res, :fsm_c, :fsm_g, :fsm_l)
        show_table("FORWARD  logsoftmax",                                  res, :fls_c, :fls_g)
        show_table3("BACKWARD softmax     (NNlib/LEF = broadcast ∇ rule)", res, :bsm_c, :bsm_g, :bsm_l)
        show_table("BACKWARD logsoftmax",                                  res, :bls_c, :bls_g)
    end
    println("\n(cuDNN/NNlib < 1 ⇒ cuDNN faster;  > 1 ⇒ custom faster.  LEF ≈ NNlib: same math.)")
end

main()
