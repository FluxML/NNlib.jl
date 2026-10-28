module NNlibCUDACUDNNExt

using NNlib
using cuDNN
using CUDA
using BFloat16s: BFloat16
using Random, Statistics

const CUDNN_6_5 = pkgversion(cuDNN) >= v"6.5"

include("conv.jl")
include("pooling.jl")
include("softmax.jl")
include("activations.jl")
include("attention.jl")
include("batchnorm.jl")
include("precompile.jl")

end # module
