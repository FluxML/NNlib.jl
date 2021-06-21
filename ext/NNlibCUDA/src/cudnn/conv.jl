
# Deprecated methods
using NNlib: DenseConvDims
import NNlib: stride, padding, dilation, flipkernel, spatial_dims, kernel_size,
    conv!, ∇conv_filter!, ∇conv_data!,
    maxpool!, meanpool!, ∇maxpool!, ∇meanpool!, PoolDims

using CUDA.CUDNN: scalingParameter, CUDNN_CONVOLUTION, convdims, 
                  cudnnConvolutionDescriptor, cudnnConvolutionBwdDataAlgoPerf,
                  cudnnConvolutionForward!, cudnnConvolutionBwdFilterAlgoPerf,
                  cudnnConvolutionBackwardData, cudnnConvolutionBackwardFilter,
                  cudnnConvolutionBackwardBias

const CUDNNFloat = Union{Float16,Float32,Float64}

# Since CUDNN does not support 1D convolution, Conv in Flux will give a CUDNNError if the size is 1-dimensional.
fix1d(x) = x
fix1d(x::DenseCuArray{T, 3}) where T = reshape(x, 1, size(x, 1), size(x, 2), size(x, 3))
fix1d(cdims::DenseConvDims{1,K,C_in,C_out,S,P,D,F}) where {K,C_in,C_out,S,P,D,F} =
    DenseConvDims{2,(1,K...),C_in,C_out,(1,S...),(0,0,P...),(1,D...),F}((1,cdims.I...))
fix1d(pdims::PoolDims{1,K,S,P,D}) where {K,S,P,D,F} =
    PoolDims{2,(1,K...),(1,S...),(0,0,P...),(1,D...)}((1,pdims.I...), pdims.C_in)

# Convolution

function cudnnConvolutionDescriptor(cdims::DenseConvDims, x::DenseCuArray{T}) where T
    cdims, x = fix1d(cdims), fix1d(x)
    mode=(NNlib.flipkernel(cdims) ? CUDNN_CROSS_CORRELATION : CUDNN_CONVOLUTION)
    cudnnConvolutionDescriptor(convdims(nnlibPadding(cdims),size(x),0),
                               convdims(NNlib.stride(cdims),size(x),1),
                               convdims(NNlib.dilation(cdims),size(x),1),
                               mode,
                               cudnnDataType(T),
                               math_mode(),
                               CUDNN_DEFAULT_REORDER,
                               Cint(1))
end

function conv!(y::DenseCuArray{T}, x::DenseCuArray{T}, w::DenseCuArray{T}, cdims::DenseConvDims;
               alpha=1, beta=0, algo=-1) where T<:CUDNNFloat
    if cudnnversion() < v"6"
        all(x -> x == 1, dilation(cdims)) || error("Only dilation = 1 is supported in cuDNN version < 6")
    end
    if algo != -1
        @warn "algo option has been deprecated, the fastest algo is computed automatically" maxlog=1
    end
    d = cudnnConvolutionDescriptor(cdims, x)
    cudnnConvolutionForward!(y, w, x, d; alpha, beta, z=y)
end

function NNlib.conv_bias_act!(y::DenseCuArray{T}, x::DenseCuArray{T}, w::DenseCuArray{T}, 
                            cdims::DenseConvDims, bias::DenseCuArray{T}, σ=identity;
                            z::DenseCuArray{T}=y, alpha=1, beta=0, algo=-1) where T<:CUDNNFloat
    if cudnnversion() < v"6"
        all(x -> x == 1, dilation(cdims)) || error("Only dilation = 1 is supported in cuDNN version < 6")
    end
    if algo != -1
        @warn "The algo option has been deprecated, the fastest algo is computed automatically" maxlog=1
    end    
    d = cudnnConvolutionDescriptor(cdims, x)
    # only relu and identity are supported by cudnnConvolutionForward!
    activation = (σ == NNlib.relu ? CUDNN_ACTIVATION_RELU : CUDNN_ACTIVATION_IDENTITY)
    cudnnConvolutionForward!(y, w, x, d; z, bias, activation, alpha, beta)
    if activation === CUDNN_ACTIVATION_IDENTITY && σ ∉ (nothing, identity)
        y = σ.(y)
    end
    return y
end

function ∇conv_data!(dx::DenseCuArray{T}, dy::DenseCuArray{T}, w::DenseCuArray{T},
                     cdims::DenseConvDims; alpha=1, beta=0, algo=-1) where T<:CUDNNFloat
    if cudnnversion() < v"6"
        all(x -> x == 1, dilation(cdims)) || error("Only dilation = 1 is supported in cuDNN version < 6")
    end
    if algo != -1
        @warn "The algo option has been deprecated, the fastest algo is computed automatically" maxlog=1
    end    
    alpha, beta = scalingParameter(T,alpha), scalingParameter(T,beta);
    xDesc, yDesc, wDesc = cudnnTensorDescriptor(dx), cudnnTensorDescriptor(dy), cudnnFilterDescriptor(w)
    convDesc = cudnnConvolutionDescriptor(cdims, dx)
    p = cudnnConvolutionBwdDataAlgoPerf(wDesc, w, yDesc, dy, convDesc, xDesc, dx)
    with_workspace(p.memory) do workspace
        cudnnConvolutionBackwardData(handle(), alpha, wDesc, w, yDesc, dy, convDesc, p.algo, workspace, sizeof(workspace), beta, xDesc, dx)
    end
    return dx
end

function ∇conv_filter!(dw::DenseCuArray{T}, x::DenseCuArray{T}, dy::DenseCuArray{T},
                       cdims::DenseConvDims; alpha=1, beta=0, algo=-1) where T<:CUDNNFloat
    if cudnnversion() < v"6"
        all(x -> x == 1, dilation(cdims)) || error("Only dilation = 1 is supported in cuDNN version < 6")
    end
    if algo != -1
        @warn "The algo option has been deprecated, the fastest algo is computed automatically" maxlog=1
    end    
    alpha, beta = scalingParameter(T,alpha), scalingParameter(T,beta);
    xDesc, yDesc, wDesc = cudnnTensorDescriptor(x), cudnnTensorDescriptor(dy), cudnnFilterDescriptor(dw)
    convDesc = cudnnConvolutionDescriptor(cdims, x)
    p = cudnnConvolutionBwdFilterAlgoPerf(xDesc, x, yDesc, dy, convDesc, wDesc, dw);
    with_workspace(p.memory) do workspace
        cudnnConvolutionBackwardFilter(handle(), alpha, xDesc, x, yDesc, dy, convDesc, p.algo, workspace, sizeof(workspace), beta, wDesc, dw);
    end
    return dw
end


function ∇conv_bias!(db::DenseCuArray{T}, dy::DenseCuArray{T}; alpha=1, beta=0) where T<:CUDNNFloat
    alpha,beta = scalingParameter(T,alpha), scalingParameter(T,beta)
    bDesc, yDesc = cudnnTensorDescriptor.((db,dy))
    cudnnConvolutionBackwardBias(handle(), alpha, yDesc, dy, beta, bDesc, db)
    return db
end

# Compatibility shims until users upgrade to new NNlib format
function conv!(y::DenseCuArray{T}, x::DenseCuArray{T}, w::DenseCuArray{T}; pad=0, stride=1, flipkernel=0, dilation=1, kwargs...) where {T<:CUDNNFloat}
    cdims = DenseConvDims(x, w; padding=pad, stride=stride, flipkernel=(flipkernel!=0), dilation=dilation)
    return conv!(y, x, w, cdims; kwargs...)
end

function ∇conv_filter!(dw::DenseCuArray{T}, dy::DenseCuArray{T}, x::DenseCuArray{T}; pad=0, stride=1, flipkernel=0, dilation=1, kwargs...) where {T<:CUDNNFloat}
    cdims = DenseConvDims(x, dw; padding=pad, stride=stride, flipkernel=(flipkernel!=0), dilation=dilation)
    # NOTE!!!  This compat shim re-arranges the argument order!
    return ∇conv_filter!(dw, x, dy, cdims; kwargs...)
end


function cudnnConvolutionForward(y::DenseCuArray{T,N}, x::DenseCuArray{T,N}, w::DenseCuArray{T,N},
                                 cdims::DenseConvDims; algo=0, alpha=1, beta=0) where {T,N}
    # @warn "`cudnnConvolutionForward(y,x,w,c::DenseConvDims)` is deprecated, please use one of the methods in `@doc cudnnConvolutionForward!`." maxlog=1
    cudnnConvolutionForward!(y, w, x; alpha, beta, padding=nnlibPadding(cdims), stride=NNlib.stride(cdims), dilation=NNlib.dilation(cdims), mode=(NNlib.flipkernel(cdims) ? CUDNN_CROSS_CORRELATION : CUDNN_CONVOLUTION))
end

function cudnnConvolutionBiasActivationForward(y::DenseCuArray{T,N}, x::DenseCuArray{T,N}, w::DenseCuArray{T,N}, z::DenseCuArray{T,N}, bias::DenseCuArray{T,N},
                                               cdims::DenseConvDims; algo=0, alpha1=1, alpha2=1,
                                               activationMode=CUDNN_ACTIVATION_RELU, activationCoeff=0.0, activationReluNanOpt=CUDNN_NOT_PROPAGATE_NAN) where {T,N}
    # @warn "`cudnnConvolutionBiasActivationForward` is deprecated, please use one of the methods in `@doc cudnnConvolutionForward!`." maxlog=1
    cudnnConvolutionForward!(y, w, x; bias, activation=activationMode, z, alpha=alpha1, beta=alpha2, padding=nnlibPadding(cdims), stride=NNlib.stride(cdims), dilation=NNlib.dilation(cdims), mode=(NNlib.flipkernel(cdims) ? CUDNN_CROSS_CORRELATION : CUDNN_CONVOLUTION))
end
