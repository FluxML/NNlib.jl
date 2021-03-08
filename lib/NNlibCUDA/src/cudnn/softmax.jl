import NNlib: softmax, softmax!, ∇softmax, ∇softmax!,
              logsoftmax, logsoftmax!, ∇logsoftmax, ∇logsoftmax!

using CUDA.CUDNN: CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_CHANNEL, 
                CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_ACCURATE, cudnnSoftmaxForward!,
                cudnnSoftmaxBackward

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

# Trick by @norci to use cudnn for softmax dims args that are contiguous: 
# If dims=(dmin:dmax) then CUDNN_SOFTMAX_MODE_CHANNEL does the trick with reshape 
#    (1, prod(size(x)[1:dmin-1]), prod(size(x)[dmin:dmax]), :)
# softmaxdims returns nothing when the backup implementation should be used.

function softmaxdims(x, dims)
    dims === Colon() && return (1, 1, length(x), 1)
    mind,maxd = minimum(dims),maximum(dims)
    all(i in dims for i in mind:maxd) || return nothing # cannot handle if not contiguous
    stride = dimsize = 1
    for i in 1:(mind-1); stride *= size(x,i); end # Using size(x,i) assumes trailing dims = 1, robust to maxd > ndims(x)
    for i in mind:maxd; dimsize *= size(x,i); end
    batchsize = length(x)÷(stride*dimsize)
    # Here is a region where cudnn is slower, so we go with the backup:
    batchsize == 1 && 64 <= stride <= 4096 && 64 <= dimsize <= 4096 && return nothing
    return (1, stride, dimsize, batchsize)
end

# Determine softmax algo based on math_mode

softmaxalgo() = (CUDA.math_mode()===CUDA.FAST_MATH ? CUDNN_SOFTMAX_FAST : CUDNN_SOFTMAX_ACCURATE)

# Main implementations:

function softmax!(y::T, x::T = y; dims=1) where {T<:DenseCuArray}
    s = softmaxdims(x, dims)
    s === nothing && return _softmax!(y, x; dims)
    cudnnSoftmaxForward!(reshape(y,s), reshape(x,s); mode = CUDNN_SOFTMAX_MODE_CHANNEL, algo = softmaxalgo())
    return y
end

function ∇softmax!(dx::T, dy::T, x::T, y::T; dims=1) where {R,T<:DenseCuArray{R}}
    s = softmaxdims(x, dims)
    s === nothing && return _∇softmax!(dx, dy, x, y; dims)
    xDesc = cudnnTensorDescriptor(reshape(x,s))
    alpha, beta = scalingParameter(R,1), scalingParameter(R,0)
    cudnnSoftmaxBackward(handle(), softmaxalgo(), CUDNN_SOFTMAX_MODE_CHANNEL, 
                         alpha, xDesc, y, xDesc, dy, beta, xDesc, dx)
    return dx
end

function logsoftmax!(y::T, x::T = y; dims=1) where {T<:DenseCuArray}
    s = softmaxdims(x, dims)
    s === nothing && return _logsoftmax!(y, x; dims)
    cudnnSoftmaxForward!(reshape(y,s), reshape(x,s); mode = CUDNN_SOFTMAX_MODE_CHANNEL, algo = CUDNN_SOFTMAX_LOG)
    return y
end

function ∇logsoftmax!(dx::T, dy::T, x::T, y::T; dims=1) where {R,T<:DenseCuArray{R}}
    s = softmaxdims(x, dims)
    s === nothing && return _∇logsoftmax!(dx, dy, x, y; dims)
    xDesc = cudnnTensorDescriptor(reshape(x,s))
    alpha, beta = scalingParameter(R,1), scalingParameter(R,0)
    cudnnSoftmaxBackward(handle(), CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_CHANNEL, 
                         alpha, xDesc, y, xDesc, dy, beta, xDesc, dx)
    return dx
end
