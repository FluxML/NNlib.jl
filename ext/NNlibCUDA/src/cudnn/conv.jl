
using NNlib: DenseConvDims
import NNlib: conv!, ∇conv_filter!, ∇conv_data!, conv_bias_act!

using CUDA.CUDNN: scalingParameter, CUDNN_CONVOLUTION, convdims, 
                  cudnnConvolutionDescriptor, cudnnConvolutionBwdDataAlgoPerf,
                  cudnnConvolutionForward!, cudnnConvolutionBwdFilterAlgoPerf,
                  cudnnConvolutionBackwardData, cudnnConvolutionBackwardFilter,
                  cudnnConvolutionBackwardBias

const CUDNNFloat = Union{Float16,Float32,Float64}

function cudnnConvolutionDescriptor(cdims::DenseConvDims, x::DenseCuArray{T}) where T
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

function conv_bias_act!(y::DenseCuArray{T}, x::DenseCuArray{T}, w::DenseCuArray{T}, 
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
