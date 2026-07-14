module NNlibCUDACUDNNExt

using NNlib
using cuDNN
using CUDA
using Random, Statistics

using cuDNN: handle, cudnnTensorDescriptor

include("conv.jl")
include("pooling.jl")
include("softmax.jl")
include("attention.jl")
include("batchnorm.jl")

end # module
