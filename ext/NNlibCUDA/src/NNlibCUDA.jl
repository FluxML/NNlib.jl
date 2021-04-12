module NNlibCUDA

using NNlib
using CUDA
using Random, Statistics

const IntOrIntTuple = Union{Integer, NTuple{N,<:Integer} where N}
const MAX_THREADS = 1024

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
