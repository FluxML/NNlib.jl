using Test
using NNlib
using Zygote
using NNlibCUDA
using ForwardDiff: Dual
using CUDA
CUDA.allowscalar(false)

include("test_utils.jl")

if CUDA.functional()
    include("activations.jl")
    include("batchedmul.jl")
    include("upsample.jl")
    include("conv.jl")
    include("pooling.jl")
    include("softmax.jl")
    include("batchnorm.jl")
else
    @warn "needs working CUDA installation to perform tests"
end
