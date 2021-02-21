using NNlib, Test, Statistics, Random
using ChainRulesCore, ChainRulesTestUtils
using Base.Broadcast: broadcasted
import FiniteDifferences
using FiniteDifferences: FiniteDifferenceMethod, central_fdm
import Zygote
using Zygote: gradient
using StableRNGs
using CUDA
CUDA.allowscalar(false)

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
