using NNlib, Test, Statistics
using ChainRulesTestUtils
import FiniteDifferences
import Zygote

"""
Compare numerical and automatic gradient.
`f` has to be a scalar valued function. 
"""
function autodiff_test(f, x)
    fdm = FiniteDifferences.central_fdm(5, 1)
    g_ad = Zygote.gradient(f, x)[1]
    g_fd = FiniteDifferences.grad(fdm, f, x)[1] 
    @test g_ad ≈ g_fd
end

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
