using cuDNN: cudnnPoolingMode_t, CUDNN_POOLING_MAX,
             CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
             CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
             cudnnPoolingForward!, pooldims, cudnnPoolingBackward

import NNlib: maxpool!, ∇maxpool!, meanpool!, ∇meanpool!
import cuDNN: cudnnPoolingDescriptor

function cudnnPoolingDescriptor(pdims::PoolDims, x::DenseCuArray{T}, mode::cudnnPoolingMode_t) where T
    window, padding, stride = NNlib.kernel_size(pdims), nnlibPadding(pdims), NNlib.stride(pdims)
    nanOpt = CUDNN_NOT_PROPAGATE_NAN
    cudnnPoolingDescriptor(mode, nanOpt, Cint(ndims(x)-2), pooldims(window,size(x)), pooldims(padding,size(x)), pooldims(stride,size(x)))
end

function maxpool!(y::DenseCuArray{T}, x::DenseCuArray{T}, pdims::PoolDims) where T<:CUDNNFloat
    d = cudnnPoolingDescriptor(pdims, x, CUDNN_POOLING_MAX)
    cudnnPoolingForward!(y, x, d)
end

function ∇maxpool!(dx::DenseCuArray{T}, dy::DenseCuArray{T}, y::DenseCuArray{T}, x::DenseCuArray{T}, pdims::PoolDims) where T<:CUDNNFloat
    xDesc, yDesc = cudnnTensorDescriptor.((x, y))
    d = cudnnPoolingDescriptor(pdims, x, CUDNN_POOLING_MAX)
    alpha, beta = scalingParameter(T,1), scalingParameter(T,0)
    cudnnPoolingBackward(handle(), d, alpha, yDesc, y, yDesc, dy, xDesc, x, beta, xDesc, dx)
    return dx
end

meanpool_mode(count_include_pad::Bool) = count_include_pad ?
    CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING :
    CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING

function meanpool!(y::DenseCuArray{T}, x::DenseCuArray{T}, pdims::PoolDims;
                   count_include_pad::Bool=true) where T<:CUDNNFloat
    d = cudnnPoolingDescriptor(pdims, x, meanpool_mode(count_include_pad))
    cudnnPoolingForward!(y, x, d)
end

function ∇meanpool!(dx::DenseCuArray{T}, dy::DenseCuArray{T}, y::DenseCuArray{T}, x::DenseCuArray{T}, pdims::PoolDims;
                    count_include_pad::Bool=true) where T<:CUDNNFloat
    xDesc, yDesc = cudnnTensorDescriptor.((x, y))
    d = cudnnPoolingDescriptor(pdims, x, meanpool_mode(count_include_pad))
    alpha, beta = scalingParameter(T,1), scalingParameter(T,0)
    cudnnPoolingBackward(handle(), d, alpha, yDesc, y, yDesc, dy, xDesc, x, beta, xDesc, dx)
    return dx
end

# Complex mean pooling (fixes https://github.com/FluxML/NNlib.jl/issues/610).
# Mean pooling is linear, so we pool the real and imaginary parts independently
# with cuDNN and recombine. This matches the CPU path, where complex `meanpool`
# already works. These handle any spatial rank: the inner real `meanpool!`/
# `∇meanpool!` calls dispatch to the 1D (rank-3) cuDNN methods below as needed.
# (`maxpool` has no canonical complex extension — `max` is undefined for complex
# numbers, and the CPU path errors too — so it is intentionally not supported.)
function meanpool!(y::DenseCuArray{T}, x::DenseCuArray{T}, pdims::PoolDims;
                   count_include_pad::Bool=true) where T<:CUDNNComplexFloat
    xr, xi = reim(x)
    yr = meanpool!(similar(y, real(T)), xr, pdims; count_include_pad)
    yi = meanpool!(similar(y, real(T)), xi, pdims; count_include_pad)
    @. y = complex(yr, yi)
    return y
end

function ∇meanpool!(dx::DenseCuArray{T}, dy::DenseCuArray{T}, y::DenseCuArray{T}, x::DenseCuArray{T},
                    pdims::PoolDims; count_include_pad::Bool=true) where T<:CUDNNComplexFloat
    dyr, dyi = reim(dy)
    yr, yi = reim(y)
    xr, xi = reim(x)
    dxr = ∇meanpool!(similar(dx, real(T)), dyr, yr, xr, pdims; count_include_pad)
    dxi = ∇meanpool!(similar(dx, real(T)), dyi, yi, xi, pdims; count_include_pad)
    @. dx = complex(dxr, dxi)
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
                                            (1, NNlib.dilation(pdims)...))
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

function ∇maxpool!(dx::DenseCuArray{T,3}, dy::DenseCuArray{T,3}, y::DenseCuArray{T,3}, x::DenseCuArray{T,3}, pdims::PoolDims) where T<:CUDNNFloat
    ∇maxpool!(add1d(dx), add1d(dy), add1d(y), add1d(x), fix_pooldims_1d(pdims))
    return dx
end

function ∇meanpool!(dx::DenseCuArray{T,3}, dy::DenseCuArray{T,3}, y::DenseCuArray{T,3}, x::DenseCuArray{T,3}, pdims::PoolDims;
                    count_include_pad::Bool=true) where T<:CUDNNFloat
    ∇meanpool!(add1d(dx), add1d(dy), add1d(y), add1d(x), fix_pooldims_1d(pdims); count_include_pad)
    return dx
end


