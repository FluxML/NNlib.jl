using CUDA.CUDNN: cudnnPoolingMode_t, CUDNN_POOLING_MAX, 
                  CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
                  cudnnPoolingForward!, pooldims, cudnnPoolingBackward
          
import CUDA.CUDNN: cudnnPoolingDescriptor

function cudnnPoolingDescriptor(pdims::PoolDims, x::DenseCuArray{T}, mode::cudnnPoolingMode_t) where T
    pdims, x = fix1d(pdims), fix1d(x)
    window, padding, stride = NNlib.kernel_size(pdims), nnlibPadding(pdims), NNlib.stride(pdims)
    nanOpt = CUDNN_NOT_PROPAGATE_NAN
    cudnnPoolingDescriptor(mode, nanOpt, Cint(max(2,ndims(x)-2)), pooldims(window,size(x)), pooldims(padding,size(x)), pooldims(stride,size(x)))
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

function maxpool!(y::DenseCuArray{T}, x::DenseCuArray{T}, k; pad=map(_->0,k), stride=k) where {T<:CUDNNFloat}
    pdims = PoolDims(x, k; padding=pad, stride=stride)
    return maxpool!(y, x, pdims)
end

function meanpool!(y::DenseCuArray{T}, x::DenseCuArray{T}, k; pad=map(_->0,k), stride=k) where {T<:CUDNNFloat}
    pdims = PoolDims(x, k; padding=pad, stride=stride)
    return meanpool!(y, x, pdims)
end

# Deprecated methods
function cudnnPoolingForward(y::DenseCuArray{T,N}, x::DenseCuArray{T,N}, pdims::NNlib.PoolDims;
                             alpha=1, beta=0, mode=CUDNN_POOLING_MAX) where {T,N}
    # @warn "`cudnnPoolingForward(y,x,d::PoolDims)` is deprecated, please use one of the methods in `@doc cudnnPoolingForward`." maxlog=1
    cudnnPoolingForward!(y, x; window=NNlib.kernel_size(pdims), padding=nnlibPadding(pdims), stride=NNlib.stride(pdims), mode, alpha, beta)
end

