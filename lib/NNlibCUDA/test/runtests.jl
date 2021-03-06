using Test
using NNlib
using CUDA
using CUDA: @allowscalar
CUDA.allowscalar(false)

# TODO remove ftest.
# GPUArrays has a testsuite that isn't part of the main package.
# Include it directly.
import GPUArrays
gpuarrays = pathof(GPUArrays)
gpuarrays_root = dirname(dirname(gpuarrays))
include(joinpath(gpuarrays_root, "test", "testsuite.jl"))
testf(f, xs...; kwargs...) = TestSuite.compare(f, CuArray, xs...; kwargs...)

if CUDA.has_cuda()
    include("activations.jl")
    include("batchedmul.jl")
    include("upsample.jl")
end
