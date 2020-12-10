using NNlib, Test, Statistics

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
