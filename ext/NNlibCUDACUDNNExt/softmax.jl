using NNlib

using cuDNN: CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_INSTANCE,
             CUDNN_SOFTMAX_ACCURATE, cudnnSoftmaxForward!, cudnnSoftmaxBackward


# We only route to cuDNN when the softmax dimensions form a *leading*, contiguous
# block (they include dim 1 with no gaps), so the softmax axis is contiguous in
# memory (stride 1). The array is then reshaped to (1, 1, dimsize, batchsize) and
# softmaxed in cuDNN's INSTANCE mode (reduce over C·H·W per sample).
#
# For any other `dims` — a non-leading axis (dims≥2), or non-contiguous dims — we
# use the generic kernels below. cuDNN's only alternative there is CHANNEL mode,
# which is much slower than the generic broadcast on the backward pass when the
# softmax axis is long (see #513), and the permute-to-leading trick costs more in
# transposes than it saves. softmaxdims returns nothing in those cases.
function softmaxdims(x, dims)
    dims === Colon() && return (1, 1, length(x), 1)
    mind, maxd = minimum(dims), maximum(dims)
    # require a leading (mind == 1), contiguous block of softmax dims
    (mind == 1 && all(i in dims for i in 1:maxd)) || return nothing
    dimsize = 1
    for i in 1:maxd; dimsize *= size(x, i); end # size(x,i) is robust to maxd > ndims(x)
    batchsize = length(x) ÷ dimsize
    return (1, 1, dimsize, batchsize)
end

# Always use the accurate algorithm (subtracts the row max before exp). The fast
# algorithm skips that and overflows to NaN on masked/large inputs, and is unsafe in
# Float16. We deliberately do not honour CUDA.FAST_MATH here. See
# https://github.com/FluxML/NNlib.jl/issues/506

softmaxalgo() = CUDNN_SOFTMAX_ACCURATE

function NNlib.softmax!(y::T, x::T; dims=1) where {T<:DenseCuArray}
    s = softmaxdims(x, dims)
    s === nothing && return NNlib._softmax!(y, x; dims)
    cudnnSoftmaxForward!(reshape(y,s), reshape(x,s); mode = CUDNN_SOFTMAX_MODE_INSTANCE, algo = softmaxalgo())
    return y
end

function NNlib.∇softmax!(dx::T, dy::T, y::T; dims=1) where {R,T<:DenseCuArray{R}}
    s = softmaxdims(y, dims)
    s === nothing && return NNlib._∇softmax!(dx, dy, y; dims)
    xDesc = cudnnTensorDescriptor(reshape(y,s))
    alpha, beta = scalingParameter(R,1), scalingParameter(R,0)
    cudnnSoftmaxBackward(handle(), softmaxalgo(), CUDNN_SOFTMAX_MODE_INSTANCE,
                         alpha, xDesc, y, xDesc, dy, beta, xDesc, dx)
    return dx
end

function NNlib.logsoftmax!(y::T, x::T; dims=1) where {T<:DenseCuArray}
    s = softmaxdims(x, dims)
    s === nothing && return NNlib._logsoftmax!(y, x; dims)
    cudnnSoftmaxForward!(reshape(y,s), reshape(x,s); mode = CUDNN_SOFTMAX_MODE_INSTANCE, algo = CUDNN_SOFTMAX_LOG)
    return y
end

function NNlib.∇logsoftmax!(dx::T, dy::T, y::T; dims=1) where {R,T<:DenseCuArray{R}}
    s = softmaxdims(y, dims)
    s === nothing && return NNlib._∇logsoftmax!(dx, dy, y; dims)
    xDesc = cudnnTensorDescriptor(reshape(y,s))
    alpha, beta = scalingParameter(R,1), scalingParameter(R,0)
    cudnnSoftmaxBackward(handle(), CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_INSTANCE,
                         alpha, xDesc, y, xDesc, dy, beta, xDesc, dx)
    return dx
end
