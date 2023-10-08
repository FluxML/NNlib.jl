module NNlibLoopVectorizationExt

using NNlib
using LoopVectorization
using Random, Statistics
using OffsetArrays, Static

include("conv.jl")
include("pooling.jl")
include("activations.jl")

end # module