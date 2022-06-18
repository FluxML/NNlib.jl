
using NNlib: DenseConvDims
import NNlib: conv!, ∇conv_filter!, ∇conv_data!, conv_bias_act!

using CUDA.CUDNN: scalingParameter, CUDNN_CONVOLUTION, convdims,
                  cudnnConvolutionDescriptor, cudnnConvolutionBwdDataAlgoPerf,
                  cudnnConvolutionForward!, cudnnConvolutionBwdFilterAlgoPerf,
                  cudnnConvolutionBackwardData, cudnnConvolutionBackwardFilter,
                  cudnnConvolutionBackwardBias

const CUDNNFloat = Union{Float16,Float32,Float64}

function cudnnConvolutionDescriptorAndPaddedInput(cdims::DenseConvDims, x::DenseCuArray{T}) where T
    # The main purpose of this function is to catch asymmetric padding which cudnn does not support
    # If we find asymmetric padding we'll make a copy of x which is manually padded so that we can
    # call cudnn with symmetric padding.
    pad = NNlib.padding(cdims)
    sdims = NNlib.spatial_dims(cdims)
    all(i -> pad[i] .== pad[i+1], 1:2:2sdims) && return (cudnnConvolutionDescriptor(cdims, x), x, identity)

    # Naive implementation, is there a faster way?
    # How much we need to pad x manually: The absolute difference between pad_left and pad_right, pad_top
    # and pad_bottom etc. respectively. We keep the sign here though because we use it below to figure out
    # which side of x to pad. Oh, and we use a CartesianIndex as we will mainly use this to index in x
    pad_manual = CartesianIndex(ntuple(i -> i > sdims ? 0 : pad[2(i-1)+1] - pad[2(i-1)+2], ndims(x)))

    # How much we can let cudnn pad: The smallest padding amount between pad_left and pad_right, pad_top 
    # and pad_bottom etc. respectively
    pad_cudnn = ntuple(i -> min(pad[2(i-1)+1], pad[2(i-1)+2]), sdims) 

    x_padded_size = ntuple(i -> i <= sdims ? size(x, i) + abs(pad_manual[i]) : size(x ,i), ndims(x))
    x_padded = similar(x, x_padded_size)
    fill!(x_padded, 0)
    # This is a bit yucky, but we are basically figuring out where in x_padded we shall insert x
    # Haven't benchmarked if this has any advantages over a more readable solution, e.g. writing dim 
    # by dim to an array in a loop
    xIs = CartesianIndices(x)
    xI_first = first(xIs)
    xI_last = last(xIs)
    xIs_pad = max(xI_first, xI_first + pad_manual) : max(xI_last, xI_last + pad_manual)
    x_padded[xIs_pad] = x 
    
    return cudnnConvolutionDescriptor(cdims, x_padded, pad_cudnn), x_padded, _x -> _x[xIs_pad]
end

function cudnnConvolutionDescriptor(cdims::DenseConvDims, x::DenseCuArray{T}, pad = nnlibPadding(cdims)) where T
    mode=(NNlib.flipkernel(cdims) ? CUDNN_CROSS_CORRELATION : CUDNN_CONVOLUTION)
    cudnnConvolutionDescriptor(convdims(pad, size(x),0),
                               convdims(NNlib.stride(cdims),size(x),1),
                               convdims(NNlib.dilation(cdims),size(x),1),
                               mode,
                               cudnnDataType(T),
                               math_mode(),
                               CUDNN_DEFAULT_REORDER,
                               Cint(NNlib.groupcount(cdims)))
end

function conv!(y::DenseCuArray{T}, x::DenseCuArray{T}, w::DenseCuArray{T}, cdims::DenseConvDims;
               alpha=1, beta=0, algo=-1) where T<:CUDNNFloat
    if cudnnversion() < v"6"
        all(x -> x == 1, dilation(cdims)) || error("Only dilation = 1 is supported in cuDNN version < 6")
    end
    if algo != -1
        @warn "algo option has been deprecated, the fastest algo is computed automatically" maxlog=1
    end
    d, x, _ = cudnnConvolutionDescriptorAndPaddedInput(cdims, x)
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
    d, x, _ = cudnnConvolutionDescriptorAndPaddedInput(cdims, x)
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
    convDesc, dx, depad = cudnnConvolutionDescriptorAndPaddedInput(cdims, dx)
    xDesc, yDesc, wDesc = cudnnTensorDescriptor(dx), cudnnTensorDescriptor(dy), cudnnFilterDescriptor(w)
    p = cudnnConvolutionBwdDataAlgoPerf(wDesc, w, yDesc, dy, convDesc, xDesc, dx, beta!=0)
    with_workspace(p.memory) do workspace
        cudnnConvolutionBackwardData(handle(), alpha, wDesc, w, yDesc, dy, convDesc, p.algo, workspace, sizeof(workspace), beta, xDesc, dx)
    end
    return depad(dx) 
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
    convDesc, x, _ = cudnnConvolutionDescriptorAndPaddedInput(cdims, x)
    xDesc, yDesc, wDesc = cudnnTensorDescriptor(x), cudnnTensorDescriptor(dy), cudnnFilterDescriptor(dw)
    p = cudnnConvolutionBwdFilterAlgoPerf(xDesc, x, yDesc, dy, convDesc, wDesc, dw, beta!=0);
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
