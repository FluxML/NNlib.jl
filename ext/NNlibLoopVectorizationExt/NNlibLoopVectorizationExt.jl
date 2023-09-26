module NNlibLoopVectorizationExt

using NNlib
using LoopVectorization
using Random, Statistics

include("conv.jl")
include("pooling.jl")

end # module