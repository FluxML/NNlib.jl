using NNlib, Test, Statistics, Random
using ChainRulesCore, ChainRulesTestUtils
using Base.Broadcast: broadcasted
import FiniteDifferences
using FiniteDifferences: FiniteDifferenceMethod, central_fdm
import Zygote
using Zygote: gradient
using StableRNGs

const rng = StableRNG(123)

include("test_utils.jl")

@testset "Activation Functions" begin
    include("activations.jl")
end

@testset "Batched Multiplication" begin
    include("batchedmul.jl")
end

@testset "Convolution" begin
    include("conv.jl")
    include("conv_bias_act.jl")
end

@testset "Inference" begin
    include("inference.jl")
end

@testset "Pooling" begin
    include("pooling.jl")
end

@testset "Padding" begin
    include("padding.jl")
end

@testset "Softmax" begin
    include("softmax.jl")
end

@testset "Upsampling" begin
    include("upsample.jl")
end

@testset "Gather" begin
    include("gather.jl")
end

@testset "Scatter" begin
    include("scatter.jl")
end

@testset "Utilities" begin
    include("utils.jl")
end

using CUDA

if VERSION >= v"1.6" && CUDA.functional()
    if get(ENV, "NNLIB_TEST_CUDA", "false") == "true"
        import Pkg
        Pkg.develop(url = "https://github.com/FluxML/NNlibCUDA.jl")
        using NNlibCUDA
        @testset "CUDA" begin
            Pkg.test("NNlibCUDA")
        end
    else
        @info "Skipping CUDA tests, set NNLIB_TEST_CUDA=true to run them"
    end
else
    @info "Insufficient version or CUDA not found; Skipping CUDA tests"
end
