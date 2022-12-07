using Test
using NNlib
using Zygote
using NNlibCUDA
using ForwardDiff: Dual
using Statistics: mean
using CUDA
CUDA.allowscalar(false)

@testset "NNlibCUDA" begin
include("test_utils.jl")
include("activations.jl")
include("batchedadjtrans.jl")
include("batchedmul.jl")
include("upsample.jl")
include("conv.jl")
include("ctc.jl")
include("fold.jl")
include("pooling.jl")
include("softmax.jl")
include("batchnorm.jl")
include("scatter.jl")
include("gather.jl")
include("sampling.jl")
end
