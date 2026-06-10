import NNlib: softmax, softmax!, ∇softmax, ∇softmax!,
              logsoftmax, logsoftmax!, ∇logsoftmax, ∇logsoftmax!

using cuDNN: CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_INSTANCE,
             CUDNN_SOFTMAX_ACCURATE, cudnnSoftmaxForward!, cudnnSoftmaxBackward

# Softmax

# @denizyuret: do not do inplace operations with softmax/logsoftmax when (1) cpu version is not, (2) one can use softmax!
function softmax(x::T; dims=1) where {T<:DenseCuArray}
    softmax!(similar(x), x; dims)
end

function ∇softmax(dy::T, x::T, y::T; dims=1) where {T<:DenseCuArray}
    ∇softmax!(similar(x), dy, x, y; dims)
end

function logsoftmax(x::T; dims=1) where {T<:DenseCuArray}
    logsoftmax!(similar(x), x; dims)
end

function ∇logsoftmax(dy::T, x::T, y::T; dims=1) where {T<:DenseCuArray}
    ∇logsoftmax!(similar(x), dy, x, y; dims)
end

# @denizyuret: backup implementations for unsupported/slow size/dims combinations:
function _softmax!(y::T, x::T; dims) where {T<:DenseCuArray}
    y .= exp.(x .- maximum(x; dims))
    y ./= sum(y; dims)
end

function _∇softmax!(dx::T, dy::T, x::T, y::T; dims) where {T<:DenseCuArray}
    dx .= y .* (dy .- sum(dy .* y; dims))
end

function _logsoftmax!(y::T, x::T; dims) where {T<:DenseCuArray}
    y .= x .- maximum(x; dims)
    y .-= log.(sum(exp.(y); dims))
end

function _∇logsoftmax!(dx::T, dy::T, x::T, y::T; dims) where {T<:DenseCuArray}
    dx .= dy .- sum(dy; dims) .* exp.(y)
end

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

# Main implementations:

function softmax!(y::T, x::T = y; dims=1) where {T<:DenseCuArray}
    s = softmaxdims(x, dims)
    s === nothing && return _softmax!(y, x; dims)
    cudnnSoftmaxForward!(reshape(y,s), reshape(x,s); mode = CUDNN_SOFTMAX_MODE_INSTANCE, algo = softmaxalgo())
    return y
end

function ∇softmax!(dx::T, dy::T, x::T, y::T; dims=1) where {R,T<:DenseCuArray{R}}
    s = softmaxdims(x, dims)
    s === nothing && return _∇softmax!(dx, dy, x, y; dims)
    xDesc = cudnnTensorDescriptor(reshape(x,s))
    alpha, beta = scalingParameter(R,1), scalingParameter(R,0)
    cudnnSoftmaxBackward(handle(), softmaxalgo(), CUDNN_SOFTMAX_MODE_INSTANCE,
                         alpha, xDesc, y, xDesc, dy, beta, xDesc, dx)
    return dx
end

function logsoftmax!(y::T, x::T = y; dims=1) where {T<:DenseCuArray}
    s = softmaxdims(x, dims)
    s === nothing && return _logsoftmax!(y, x; dims)
    cudnnSoftmaxForward!(reshape(y,s), reshape(x,s); mode = CUDNN_SOFTMAX_MODE_INSTANCE, algo = CUDNN_SOFTMAX_LOG)
    return y
end

function ∇logsoftmax!(dx::T, dy::T, x::T, y::T; dims=1) where {R,T<:DenseCuArray{R}}
    s = softmaxdims(x, dims)
    s === nothing && return _∇logsoftmax!(dx, dy, x, y; dims)
    xDesc = cudnnTensorDescriptor(reshape(x,s))
    alpha, beta = scalingParameter(R,1), scalingParameter(R,0)
    cudnnSoftmaxBackward(handle(), CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_INSTANCE,
                         alpha, xDesc, y, xDesc, dy, beta, xDesc, dx)
    return dx
end
