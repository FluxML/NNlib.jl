using NNlib, NNlibCUDA, CUDA, Test
using Zygote, ChainRulesCore

@testset "dropout + CUDA" begin
    # Basics
    x1 = CUDA.randn(3, 4)
    @test size(@inferred dropout(x1, 0.1)) == (3, 4)
    @test size(@inferred dropout(x1, 0.2; dims=2)) == (3, 4)
    @test size(@inferred dropout(x1, 0.3; dims=(1,2))) == (3, 4)

    rng =  CUDA.default_rng()
    @test size(@inferred dropout(rng, x1, 0.1)) == (3, 4)
    @test size(@inferred dropout(rng, x1, 0.1; dims=2)) == (3, 4)

    # Values
    d45 = dropout(CUDA.ones(100, 100, 100), 0.45)
    @test mean(d45) ≈ 1 atol=1e-2
    dpi2 = dropout(CUDA.fill(1f0 * pi, 1000), 0.2)
    @test sort(unique(Array(dpi2))) ≈ [0, 5pi/4]
    d33 = dropout(CUDA.fill(3f0, 10, 1000), 0.3, dims=2)
    @test sort(unique(vec(Array(d33)))) ≈ [0, 3/(1-0.3)]

    # Gradient rule
    y, back = rrule(dropout, rng, hcat(CUDA.ones(1000), CUDA.zeros(1000)), 0.45)
    dx = back(CUDA.fill(3f0, 1000, 2))[3]
    @test !all(iszero, dx[:,2])  # this is why we save the random choices
    @test sort(unique(vec(Array(dx)))) ≈ [0, 3/(1-0.45)]

    @testset "Zygote" begin
        @test Zygote.gradient(x -> sum(dropout(x, 0.3)), x1)[1] isa CuArray{Float32}
        @test Zygote.gradient(x -> sum(dropout(rng, x, 0.3)), x1)[1] isa CuArray{Float32}
        @test Zygote.gradient(x -> sum(dropout(x, 0.3, dims=1)), x1)[1] isa CuArray{Float32}
    end
end
