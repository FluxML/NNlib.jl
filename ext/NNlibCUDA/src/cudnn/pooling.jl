using CUDA.CUDNN: cudnnPoolingMode_t, CUDNN_POOLING_MAX, 
                  CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
                  cudnnPoolingForward!, pooldims, cudnnPoolingBackward
          
import NNlib: maxpool!, ∇maxpool!, meanpool!, ∇meanpool!
import CUDA.CUDNN: cudnnPoolingDescriptor

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

function meanpool!(y::DenseCuArray{T}, x::DenseCuArray{T}, pdims::PoolDims) where T<:CUDNNFloat
    d = cudnnPoolingDescriptor(pdims, x, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING)
    cudnnPoolingForward!(y, x, d)
end

function ∇meanpool!(dx::DenseCuArray{T}, dy::DenseCuArray{T}, y::DenseCuArray{T}, x::DenseCuArray{T}, pdims::PoolDims) where T<:CUDNNFloat
    xDesc, yDesc = cudnnTensorDescriptor.((x, y))
    d = cudnnPoolingDescriptor(pdims, x, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING)
    alpha, beta = scalingParameter(T,1), scalingParameter(T,0)
    cudnnPoolingBackward(handle(), d, alpha, yDesc, y, yDesc, dy, xDesc, x, beta, xDesc, dx)
    return dx
end

### Since CUDA.jl does not support 1D pooling, we have to convert to 2d

add1d(x) = reshape(x, 1, size(x)...)

fix_pooldims_1d(pdims::PoolDims{1,K,S,P,D}) where {K,S,P,D} =
        PoolDims{2,(1,K...),(1,S...),(0,0,P...),(1,D...)}((1,pdims.I...), pdims.C_in)

function maxpool!(y::DenseCuArray{T,3}, x::DenseCuArray{T,3}, pdims::PoolDims) where T<:CUDNNFloat
    maxpool!(add1d(y), add1d(x), fix_pooldims_1d(pdims))
    return y
end

function meanpool!(y::DenseCuArray{T,3}, x::DenseCuArray{T,3}, pdims::PoolDims) where T<:CUDNNFloat
    meanpool!(add1d(y), add1d(x), fix_pooldims_1d(pdims))
    return y
end

function ∇maxpool!(dx::DenseCuArray{T,3}, dy::DenseCuArray{T,3}, y::DenseCuArray{T,3}, x::DenseCuArray{T,3}, pdims::PoolDims) where T<:CUDNNFloat
    ∇maxpool!(add1d(dx), add1d(dy), add1d(y), add1d(x), fix_pooldims_1d(pdims))
    return dx
end

function ∇meanpool!(dx::DenseCuArray{T,3}, dy::DenseCuArray{T,3}, y::DenseCuArray{T,3}, x::DenseCuArray{T,3}, pdims::PoolDims) where T<:CUDNNFloat
    ∇meanpool!(add1d(dx), add1d(dy), add1d(y), add1d(x), fix_pooldims_1d(pdims))
    return dx
end


