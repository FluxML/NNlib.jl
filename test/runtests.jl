using NNlib, Test, Statistics
using ChainRulesTestUtils
import FiniteDifferences
import Zygote
using Zygote: gradient
using StableRNGs

include("test_utils.jl")

@testset "Activation Functions" begin
    include("activation.jl")
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

@testset "Softmax" begin
    include("softmax.jl")
end

@testset "Zygote" begin
    include("zygote.jl")
end
