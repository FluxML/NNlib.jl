using ChainRulesCore: NoTangent, ZeroTangent, unthunk

@testset "grouped_mul validation" begin
    W, X, B = randn(3, 2, 3), randn(2, 7), randn(7, 4)
    @test_throws DimensionMismatch grouped_mul(W, X, [2, 4])
    @test_throws DimensionMismatch grouped_mul(W, B, [2, 4, 6])
    @test_throws DimensionMismatch grouped_mul(X, X, [2, 4, 6])
    @test_throws DimensionMismatch grouped_mul!(zeros(3, 6), W, X, [2, 4, 6])
    @test_throws DimensionMismatch grouped_mul!(zeros(2, 4, 2), X, B, [2, 4, 6])
    for offsets in ([-1, 2, 3], [2, 1, 3], [2, 4, 8])
        @test_throws ArgumentError grouped_mul(W, X, offsets)
        @test_throws ArgumentError grouped_mul(X, B, offsets)
    end
    @test all(iszero, grouped_mul(zeros(3, 2, 0), X, Int[]))
    @test size(grouped_mul(X, B, Int[])) == (2, 4, 0)
    @test all(iszero, grouped_mul(zeros(3, 0, 3), zeros(0, 7), [0, 2, 4]))
    @test all(iszero, grouped_mul(zeros(3, 0), zeros(0, 4), [0, 0, 0]))
    @test size(grouped_mul(zeros(0, 2, 3), X, [0, 2, 4])) == (0, 7)
    @test size(grouped_mul(zeros(2, 0, 3), zeros(0, 0), [0, 0, 0])) == (2, 0)
    @test eltype(grouped_mul(Float32.(W), X, [0, 2, 4])) == Float64
    @test grouped_mul(ones(Int, 3, 2, 3), ones(Int, 2, 7), [0, 2, 4]) ==
        [2 2 2 2 0 0 0; 2 2 2 2 0 0 0; 2 2 2 2 0 0 0]
end

@testset "grouped_mul rrules ($T)" for T in (Float64, ComplexF64)
    offsets = [0, 1, 1, 3]
    W, X, B = randn(rng, T, 2, 3, 4), randn(rng, T, 3, 4), randn(rng, T, 4, 2)
    for (a, b) in ((W, X), (X, B), (batched_transpose(W), randn(rng, T, 2, 4)),
                   (batched_adjoint(W), randn(rng, T, 2, 4)), (adjoint(B), adjoint(X)))
        ChainRulesTestUtils.test_rrule(grouped_mul, a, b,
            ChainRulesTestUtils.PrimalAndTangent(offsets, NoTangent()))
        y, back = rrule(grouped_mul, a, b, offsets)
        @test back(ZeroTangent()) == (NoTangent(), ZeroTangent(), ZeroTangent(), NoTangent())
        tangents = back(ones(T, size(y)))
        @test tangents[4] isa NoTangent
        @test size(unthunk(tangents[2])) == size(a)
        @test size(unthunk(tangents[3])) == size(b)
    end
end
