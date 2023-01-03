using NNlib, Test, Statistics, Random
using Zygote, StableRNGs, ChainRulesCore

@testset "dropout" begin
    # Basics
    x1 = randn(Float32, 3, 4)
    @test size(@inferred dropout(x1, 0.1)) == (3, 4)
    @test size(@inferred dropout(x1, 0.2; dims=2)) == (3, 4)
    @test size(@inferred dropout(x1, 0.3; dims=(1,2))) == (3, 4)
    @test eltype(dropout(x1, 0.1)) == Float32
    @test eltype(dropout(x1, 0.1; dims=1)) == Float32
    @test eltype(dropout(x1, 0.1; dims=(1,2))) == Float32

    rng =  Random.default_rng()
    @test size(@inferred dropout(rng, x1, 0.1)) == (3, 4)
    @test size(@inferred dropout(rng, x1, 0.1; dims=2)) == (3, 4)

    # Values
    d45 = dropout(trues(100, 100, 100), 0.45)
    @test mean(d45) ≈ 1 atol=1e-2
    dpi2 = dropout(fill(pi, 1000), 0.2)
    @test sort(unique(dpi2)) ≈ [0, 5pi/4]
    d33 = dropout(fill(3, 10, 1000), 0.3, dims=2)
    @test sort(unique(vec(d33))) ≈ [0, 3/(1-0.3)]

    # Gradient rule
    y, back = rrule(dropout, rng, hcat(trues(1000), falses(1000)), 0.45)
    dx = back(fill(3, 1000, 2))[3]
    @test !all(iszero, dx[:,2])  # this is why we save the random choices
    @test sort(unique(vec(dx))) ≈ [0, 3/(1-0.45)]

    @testset "Zygote" begin
        @test Zygote.gradient(x -> sum(dropout(x, 0.3)), x1)[1] isa Matrix{Float32}
        @test Zygote.gradient(x -> sum(dropout(rng, x, 0.3)), x1)[1] isa Matrix{Float32}
        @test Zygote.gradient(x -> sum(dropout(x, 0.3, dims=1)), x1)[1] isa Matrix{Float32}

        f1(x) = sum(dropout(x, 0.5))
        @test_broken Zygote.hessian(f1, [1.0,2.0,3.0]) == zeros(3, 3)  # forward over reverse
        @test Zygote.hessian_reverse(f1, [1.0,2.0,3.0]) == zeros(3, 3)
    end
end

