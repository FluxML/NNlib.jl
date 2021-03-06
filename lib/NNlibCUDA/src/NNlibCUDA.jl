module NNlibCUDA

using NNlib
using CUDA
using CUDA: @cufunc
using Random, Statistics

include("upsample.jl")
include("activations.jl")
include("batchedmul.jl")

end # module
