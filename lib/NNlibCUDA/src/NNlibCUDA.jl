module NNlibCUDA

using NNlib
using CUDA
using CUDA: @cufunc
using Random, Statistics

include("upsample.jl")
include("activations.jl")
include("batchedmul.jl")
include("cudnn/cudnn.jl")
include("cudnn/conv.jl")
include("cudnn/pooling.jl")
include("cudnn/softmax.jl")
include("cudnn/activations.jl")
include("cudnn/batchnorm.jl")

end # module
