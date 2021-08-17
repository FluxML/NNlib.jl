using Test
using NNlib
using Zygote
using NNlibCUDA
using ForwardDiff: Dual
using Statistics: mean
using CUDA
CUDA.allowscalar(false)

include("test_utils.jl")
include("activations.jl")
include("batchedmul.jl")
include("upsample.jl")
include("conv.jl")
include("pooling.jl")
include("softmax.jl")
include("batchnorm.jl")
include("scatter.jl")
include("gather.jl")
