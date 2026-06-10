# GPU softmax / logsoftmax benchmark — cuDNN vs the custom (CPU-algorithm) kernels.
#
# Compares, on the GPU, the specialized cuDNN softmax against NNlib's generic
# array kernels (the ones used on the CPU):
#   * forward  — cuDNN `cudnnSoftmaxForward!`  vs the `exp.(x .- max)/sum` kernel
#   * backward — cuDNN `cudnnSoftmaxBackward`  vs the broadcast ∇ rule
# for both `softmax` and `logsoftmax`, and for `dims=1` and `dims=2`.
# Reproduces / extends FluxML/NNlib.jl#513.
#
# cuDNN softmax is timed in BOTH of its modes:
#   * CHANNEL  — reduces over the C axis; what NNlib used unconditionally. Very
#                slow on the backward pass when the softmax axis is long.
#   * INSTANCE — reduces over C·H·W per sample. Equivalent to CHANNEL only when the
#                softmax axis is the leading contiguous dimension (stride == 1, i.e.
#                dims=1 / Colon); otherwise it would reduce the wrong elements, so it
#                is marked "—". Much faster than CHANNEL (and the custom kernel) on
#                the backward pass — this is the fix for #513.
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
             CUDNN_SOFTMAX_MODE_INSTANCE, cudnnSoftmaxForward!, cudnnSoftmaxBackward,
             cudnnTensorDescriptor, scalingParameter, handle

# ---- cuDNN path (called directly, with an explicit mode) ----------------------
# cuDNN softmax over an integer axis `dims` reshapes the array to
# (1, stride, dimsize, batchsize). CHANNEL reduces over dimsize (= C); INSTANCE
# reduces over C·H·W and is only correct here when stride == 1.
function cudnn_shape(x, dims::Int)
    stride    = prod(size(x)[1:dims-1]; init = 1)
    dimsize   = size(x, dims)
    batchsize = length(x) ÷ (stride * dimsize)
    (1, stride, dimsize, batchsize)
end
instance_valid(x, dims::Int) = prod(size(x)[1:dims-1]; init = 1) == 1

function cudnn_softmax!(out, x; dims, algo, mode)
    s = cudnn_shape(x, dims)
    cudnnSoftmaxForward!(reshape(out, s), reshape(x, s); mode, algo)
    out
end
function cudnn_∇softmax!(dx, dy, x, y; dims, algo, mode)
    s = cudnn_shape(x, dims)
    xDesc = cudnnTensorDescriptor(reshape(x, s))
    R = eltype(x); alpha, beta = scalingParameter(R, 1), scalingParameter(R, 0)
    cudnnSoftmaxBackward(handle(), algo, mode,
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
const CHAN, INST = CUDNN_SOFTMAX_MODE_CHANNEL, CUDNN_SOFTMAX_MODE_INSTANCE
const NA = NaN  # marks an INSTANCE entry that is not correctness-preserving (stride>1)

# time a thunk, in µs
us(f) = (@belapsed CUDA.@sync $f()) * 1e6

function bench_all(sizes, dims)
    map(sizes) do s
        x = CUDA.randn(Float32, s...); dy = CUDA.randn(Float32, s...); o = similar(x)
        ys = custom_softmax!(similar(x), x; dims); yl = custom_logsoftmax!(similar(x), x; dims)
        inst = instance_valid(x, dims)
        (; size = s,
          # forward softmax: cuDNN CHANNEL / cuDNN INSTANCE / NNlib custom / LEF
          fsm_c = us(() -> cudnn_softmax!(o, x; dims, algo=A, mode=CHAN)),
          fsm_i = inst ? us(() -> cudnn_softmax!(o, x; dims, algo=A, mode=INST)) : NA,
          fsm_g = us(() -> custom_softmax!(o, x; dims)),
          fsm_l = us(() -> lef_softmax!(o, x; dims)),
          # forward logsoftmax: cuDNN CHANNEL / cuDNN INSTANCE / NNlib custom
          fls_c = us(() -> cudnn_softmax!(o, x; dims, algo=L, mode=CHAN)),
          fls_i = inst ? us(() -> cudnn_softmax!(o, x; dims, algo=L, mode=INST)) : NA,
          fls_g = us(() -> custom_logsoftmax!(o, x; dims)),
          # backward softmax
          bsm_c = us(() -> cudnn_∇softmax!(o, dy, x, ys; dims, algo=A, mode=CHAN)),
          bsm_i = inst ? us(() -> cudnn_∇softmax!(o, dy, x, ys; dims, algo=A, mode=INST)) : NA,
          bsm_g = us(() -> custom_∇softmax!(o, dy, x, ys; dims)),
          bsm_l = us(() -> lef_∇softmax!(o, dy, ys; dims)),
          # backward logsoftmax
          bls_c = us(() -> cudnn_∇softmax!(o, dy, x, yl; dims, algo=L, mode=CHAN)),
          bls_i = inst ? us(() -> cudnn_∇softmax!(o, dy, x, yl; dims, algo=L, mode=INST)) : NA,
          bls_g = us(() -> custom_∇logsoftmax!(o, dy, x, yl; dims)))
    end
end

function check_correctness()
    for dims in (1, 2)
        x = CUDA.randn(Float32, 512, 700); dy = CUDA.randn(Float32, 512, 700)
        ys = custom_softmax!(similar(x), x; dims); yl = custom_logsoftmax!(similar(x), x; dims)
        f(a, b) = maximum(abs.(Array(a) .- Array(b)))
        # cuDNN CHANNEL vs custom
        e1 = f(cudnn_softmax!(similar(x), x; dims, algo=A, mode=CHAN), ys)
        e2 = f(cudnn_∇softmax!(similar(x), dy, x, ys; dims, algo=A, mode=CHAN), custom_∇softmax!(similar(x), dy, x, ys; dims))
        # cuDNN INSTANCE vs custom (only meaningful when stride==1; here dims=1)
        einst = if instance_valid(x, dims)
            max(f(cudnn_softmax!(similar(x), x; dims, algo=A, mode=INST), ys),
                f(cudnn_∇softmax!(similar(x), dy, x, ys; dims, algo=A, mode=INST), custom_∇softmax!(similar(x), dy, x, ys; dims)))
        else
            NA
        end
        # LEF vs custom
        e5 = f(lef_softmax!(similar(x), x; dims), ys)
        e6 = f(lef_∇softmax!(similar(x), dy, ys; dims), custom_∇softmax!(similar(x), dy, x, ys; dims))
        @printf("correctness dims=%d  CHANNEL fwd=%.0e bwd=%.0e  INSTANCE=%s  LEF fwd=%.0e grad=%.0e\n",
                dims, e1, e2, isnan(einst) ? "n/a" : @sprintf("%.0e", einst), e5, e6)
    end
end

# Flexible printer. `cols` is a list of (header, field) pairs; NaN prints as "—".
# `ratio`, if given, is (header, num_field, den_field) and prints num/den.
function show_cols(title, res, cols; ratio = nothing)
    println("\n", title)
    @printf("%-14s", "size")
    for (h, _) in cols; @printf(" %9s", h); end
    ratio === nothing || @printf(" %11s", ratio[1])
    println()
    for r in res
        @printf("%-14s", string(r.size))
        for (_, sym) in cols
            v = getfield(r, sym)
            isnan(v) ? @printf(" %9s", "—") : @printf(" %9.1f", v)
        end
        if ratio !== nothing
            a, b = getfield(r, ratio[2]), getfield(r, ratio[3])
            (isnan(a) || isnan(b)) ? @printf(" %10s", "—") : @printf(" %10.2fx", a / b)
        end
        println()
    end
end

const SM_COLS(fwd) = [("cuDNN-CH", Symbol(fwd, "_c")), ("cuDNN-IN", Symbol(fwd, "_i")),
                      ("NNlib", Symbol(fwd, "_g")), ("LEF", Symbol(fwd, "_l"))]
const LS_COLS(fwd) = [("cuDNN-CH", Symbol(fwd, "_c")), ("cuDNN-IN", Symbol(fwd, "_i")),
                      ("NNlib", Symbol(fwd, "_g"))]

function main()
    @assert CUDA.functional() "CUDA is not functional"
    println("GPU:   ", CUDA.name(CUDA.device()))
    println("cuDNN: ", cuDNN.version())
    check_correctness()
    for dims in (1, 2)
        res = bench_all(SIZES, dims)
        println("\n", "#"^64, "\n### dims=$dims   (times in µs, Float32)\n", "#"^64)
        show_cols("FORWARD  softmax",     res, SM_COLS(:fsm); ratio = ("CH/IN", :fsm_c, :fsm_i))
        show_cols("FORWARD  logsoftmax",  res, LS_COLS(:fls); ratio = ("CH/IN", :fls_c, :fls_i))
        show_cols("BACKWARD softmax",     res, SM_COLS(:bsm); ratio = ("CH/IN", :bsm_c, :bsm_i))
        show_cols("BACKWARD logsoftmax",  res, LS_COLS(:bls); ratio = ("CH/IN", :bls_c, :bls_i))
    end
    println("\ncuDNN-CH = CHANNEL mode (old), cuDNN-IN = INSTANCE mode (fix, stride==1 only).")
    println("CH/IN > 1 ⇒ INSTANCE faster than CHANNEL.  LEF ≈ NNlib (same math).")
end

main()
