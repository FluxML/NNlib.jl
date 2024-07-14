module NNlibCUDAExt

using NNlib
using CUDA
using Random, Statistics

include("sampling.jl")
include("activations.jl")
include("batchedadjtrans.jl")
include("batchedmul.jl")
include("ctc.jl")
include("scatter.jl")
include("utils.jl")

end # module
