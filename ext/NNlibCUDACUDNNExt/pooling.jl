using cuDNN: cudnnPoolingMode_t, CUDNN_POOLING_MAX,
             CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
             CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
             cudnnPoolingForward!, pooldims, cudnnPoolingBackward,
             cudnnTensorDescriptor, handle, scalingParameter, CUDNN_NOT_PROPAGATE_NAN

import NNlib: maxpool!, ∇maxpool!, meanpool!, ∇meanpool!
import cuDNN: cudnnPoolingDescriptor

# cuDNN's fixed-function pooling only takes symmetric padding. Asymmetric padding goes
# through its graph API, which handles it natively but is otherwise somewhat slower.
function symmetric_padding(pdims::PoolDims)
    pad = NNlib.padding(pdims)
    return all(i -> pad[2i-1] == pad[2i], 1:length(pad)÷2)
end

# The legacy API pools into an output of any size, and so supports `ceil_mode`: it skips
# the positions of a window past the padding, also in the divisor. The graph API only
# takes the output size that the padding gives.
function pool_kwargs(pdims::PoolDims)
    NNlib.ceil_mode(pdims) && throw(ArgumentError(
        "cuDNN pooling supports `ceil_mode` only with symmetric padding, got $(NNlib.padding(pdims))"))
    return (window=NNlib.kernel_size(pdims),
            padding=NNlib.padding(pdims),
            stride=NNlib.stride(pdims))
end

function cudnnPoolingDescriptor(pdims::PoolDims, x::DenseCuArray{T}, mode::cudnnPoolingMode_t) where T
    pad = NNlib.padding(pdims)
    window, padding, stride = NNlib.kernel_size(pdims), ntuple(i -> pad[2i-1], length(pad)÷2), NNlib.stride(pdims)
    nanOpt = CUDNN_NOT_PROPAGATE_NAN
    cudnnPoolingDescriptor(mode, nanOpt, Cint(ndims(x)-2), pooldims(window,size(x)), pooldims(padding,size(x)), pooldims(stride,size(x)))
end

function maxpool!(y::DenseCuArray{T}, x::DenseCuArray{T}, pdims::PoolDims) where T<:CUDNNFloat
    symmetric_padding(pdims) || return cuDNN.maxpool!(y, x; pool_kwargs(pdims)...)
    d = cudnnPoolingDescriptor(pdims, x, CUDNN_POOLING_MAX)
    cudnnPoolingForward!(y, x, d)
end

function ∇maxpool!(dx::DenseCuArray{T}, dy::DenseCuArray{T}, y::DenseCuArray{T}, x::DenseCuArray{T}, pdims::PoolDims;
                   alpha=1, beta=0, kwargs...) where T<:CUDNNFloat
    symmetric_padding(pdims) ||
        return cuDNN.∇maxpool!(dx, dy, y, x; pool_kwargs(pdims)..., alpha, beta)
    xDesc, yDesc = cudnnTensorDescriptor.((x, y))
    d = cudnnPoolingDescriptor(pdims, x, CUDNN_POOLING_MAX)
    alpha, beta = scalingParameter(T,alpha), scalingParameter(T,beta)
    cudnnPoolingBackward(handle(), d, alpha, yDesc, y, yDesc, dy, xDesc, x, beta, xDesc, dx)
    return dx
end

meanpool_mode(count_include_pad::Bool) = count_include_pad ?
    CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING :
    CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING

function meanpool!(y::DenseCuArray{T}, x::DenseCuArray{T}, pdims::PoolDims;
                   count_include_pad::Bool=true) where T<:CUDNNFloat
    symmetric_padding(pdims) ||
        return cuDNN.meanpool!(y, x; pool_kwargs(pdims)..., count_include_pad)
    d = cudnnPoolingDescriptor(pdims, x, meanpool_mode(count_include_pad))
    cudnnPoolingForward!(y, x, d)
end

function ∇meanpool!(dx::DenseCuArray{T}, dy::DenseCuArray{T}, y::DenseCuArray{T}, x::DenseCuArray{T}, pdims::PoolDims;
                    count_include_pad::Bool=true, alpha=1, beta=0, kwargs...) where T<:CUDNNFloat
    symmetric_padding(pdims) ||
        return cuDNN.∇meanpool!(dx, dy, y, x; pool_kwargs(pdims)..., count_include_pad, alpha, beta)
    xDesc, yDesc = cudnnTensorDescriptor.((x, y))
    d = cudnnPoolingDescriptor(pdims, x, meanpool_mode(count_include_pad))
    alpha, beta = scalingParameter(T,alpha), scalingParameter(T,beta)
    cudnnPoolingBackward(handle(), d, alpha, yDesc, y, yDesc, dy, xDesc, x, beta, xDesc, dx)
    return dx
end

### Since CUDA.jl does not support 1D pooling, we have to convert to 2d

add1d(x) = reshape(x, 1, size(x)...)

function fix_pooldims_1d(pdims::PoolDims{1,K,S,P,D}) where {K,S,P,D}
    PoolDims{2, K + 1, S + 1, P + 2, D + 1}((1, NNlib.input_size(pdims)...),
                                            (1, NNlib.kernel_size(pdims)...),
                                            NNlib.channels_in(pdims),
                                            (1, NNlib.stride(pdims)...),
                                            (0, 0, NNlib.padding(pdims)...),
                                            (1, NNlib.dilation(pdims)...),
                                            NNlib.ceil_mode(pdims))
end

function maxpool!(y::DenseCuArray{T,3}, x::DenseCuArray{T,3}, pdims::PoolDims) where T<:CUDNNFloat
    maxpool!(add1d(y), add1d(x), fix_pooldims_1d(pdims))
    return y
end

function meanpool!(y::DenseCuArray{T,3}, x::DenseCuArray{T,3}, pdims::PoolDims;
                   count_include_pad::Bool=true) where T<:CUDNNFloat
    meanpool!(add1d(y), add1d(x), fix_pooldims_1d(pdims); count_include_pad)
    return y
end

function ∇maxpool!(dx::DenseCuArray{T,3}, dy::DenseCuArray{T,3}, y::DenseCuArray{T,3}, x::DenseCuArray{T,3}, pdims::PoolDims; kwargs...) where T<:CUDNNFloat
    ∇maxpool!(add1d(dx), add1d(dy), add1d(y), add1d(x), fix_pooldims_1d(pdims); kwargs...)
    return dx
end

function ∇meanpool!(dx::DenseCuArray{T,3}, dy::DenseCuArray{T,3}, y::DenseCuArray{T,3}, x::DenseCuArray{T,3}, pdims::PoolDims;
                    count_include_pad::Bool=true, kwargs...) where T<:CUDNNFloat
    ∇meanpool!(add1d(dx), add1d(dy), add1d(y), add1d(x), fix_pooldims_1d(pdims); count_include_pad, kwargs...)
    return dx
end
