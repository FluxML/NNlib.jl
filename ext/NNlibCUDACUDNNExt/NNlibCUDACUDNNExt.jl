module NNlibCUDACUDNNExt

using NNlib
using cuDNN
using CUDA
using Random, Statistics

using cuDNN: handle, with_workspace, cudnnTensorDescriptor, cudnnFilterDescriptor,
             cudnnDataType, math_mode, CUDNN_DEFAULT_REORDER, CUDNN_CROSS_CORRELATION,
             CUDNN_NOT_PROPAGATE_NAN, CUDNN_TENSOR_NCHW, dim4

cudnnversion() = cuDNN.version()

function nnlibPadding(dims)
    pd = NNlib.padding(dims)
    if !all(pd[1:2:end] .== pd[2:2:end])
        @warn "cuDNN does not support asymmetric padding; defaulting to symmetric choice" maxlog=1
    end
    return pd[1:2:end]
end

include("conv.jl")
include("pooling.jl")
include("softmax.jl")
include("activations.jl")
include("batchnorm.jl")

end # module