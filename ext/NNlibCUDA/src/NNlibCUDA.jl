module NNlibCUDA

using NNlib
using CUDA, cuDNN
using Random, Statistics

const IntOrIntTuple = Union{Integer, NTuple{N,<:Integer} where N}

include("sampling.jl")
include("activations.jl")
include("batchedadjtrans.jl")
include("batchedmul.jl")
include("ctc.jl")
include("fold.jl")
include("scatter.jl")
include("utils.jl")
include("cudnn/cudnn.jl")
include("cudnn/conv.jl")
include("cudnn/pooling.jl")
include("cudnn/softmax.jl")
include("cudnn/activations.jl")
include("cudnn/batchnorm.jl")

end # module
