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

    x2 = LinearAlgebra.Diagonal(randn(Float32, 10))  # Just to check it runs on weird matrices.
    @test dropout(x2, 0.3) isa Matrix{Float32}  # does not infer, but that's OK?
    
    # Values
    @test dropout(x1, 0) == x1
    @test dropout(x1.+0im, 0) == x1
    @test dropout(x1, 1) == zero.(x1)
    @test dropout(x1.+im, 1) == zero.(x1)

    d45 = dropout(trues(100, 100, 100), 0.45)
    @test mean(d45) ≈ 1 atol=1e-2
    dpi2 = dropout(fill(pi, 1000), 0.2)
    @test sort(unique(dpi2)) ≈ [0, 5pi/4]
    d33 = dropout(fill(3, 10, 1000), 0.3, dims=2)
    @test sort(unique(vec(d33))) ≈ [0, 3/(1-0.3)]

    # Complex -- not worth too much optimisation, but should work!
    x2 = [1.0+0im,2.0+1im,3.0+3im]  # from Flux's tests
    @test dropout(x2, 0.5) isa Vector{ComplexF64}
    @test dropout(x2, 0.5; dims=1) isa Vector{ComplexF64}

    # Gradient rule
    y, back = rrule(dropout, rng, hcat(trues(1000), falses(1000)), 0.45)
    dx = back(fill(3, 1000, 2))[3]
    @test !all(iszero, dx[:,2])  # this is why we save the random choices
    @test sort(unique(vec(dx))) ≈ [0, 3/(1-0.45)]

    y2, back2 = rrule(dropout, rng, x2, 0.5)
    @test y2 isa Vector{ComplexF64}
    @test back2(one.(y2))[3] isa Vector{ComplexF64}

    @testset "Zygote" begin
        @test Zygote.gradient(x -> sum(dropout(x, 0.3)), x1)[1] isa Matrix{Float32}
        @test Zygote.gradient(x -> sum(dropout(rng, x, 0.3)), x1)[1] isa Matrix{Float32}
        @test Zygote.gradient(x -> sum(dropout(x, 0.3, dims=1)), x1)[1] isa Matrix{Float32}

        # p=0 & p=1
        @test Zygote.gradient(x -> sum(dropout(x, 0)), x1)[1] == ones(3,4)
        @test Zygote.gradient(x -> sum(dropout(x, 1)), x1)[1] == zeros(3,4)

        # Second order
        f1(x) = sum(dropout(x, 0.5))
        @test_broken Zygote.hessian(f1, [1.0,2.0,3.0]) == zeros(3, 3)  # forward over reverse
        @test Zygote.hessian_reverse(f1, [1.0,2.0,3.0]) == zeros(3, 3)
    end

    @testset "Mooncake" begin
        # Given the mask, dropout is linear in x, so the value is `sum(x .* grad)`.
        for f in (x -> sum(dropout(x, 0.3)), x -> sum(dropout(rng, x, 0.3, dims=1)))
            y, (dx,) = withgradient(f, AutoMooncake(), x1)
            @test y ≈ sum(x1 .* dx)
        end
    end

    # Bang
    y1 = fill!(similar(x1), NaN)
    @test dropout!(y1, x1, 0.0) == x1
    @test y1 == x1
    @test dropout!(rng, y1, x1, 1) == zero(x1)
    @test y1 == zero(x1)

    # Errors
    @test_throws ArgumentError dropout(x1, -1)
    @test_throws ArgumentError dropout(x1, 2)
    @test_throws ArgumentError dropout!(y1, x1, 3)
end

# https://github.com/FluxML/NNlib.jl/issues/807
@testset "dropout with a non-Real eltype" begin
    # Like Reactant's `TracedRNumber`: a `Number`, but not a `Real`.
    struct NotReal <: Number
        x::Float32
    end
    Base.float(::Type{NotReal}) = NotReal
    Base.real(::Type{NotReal}) = NotReal
    Base.real(a::NotReal) = a
    Base.promote_rule(::Type{NotReal}, ::Type{<:Real}) = NotReal
    Base.:-(a::NotReal, b::NotReal) = NotReal(a.x - b.x)
    Base.:*(a::NotReal, b::NotReal) = NotReal(a.x * b.x)
    Base.:/(a::NotReal, b::NotReal) = NotReal(a.x / b.x)
    Base.:<(a::NotReal, b::NotReal) = a.x < b.x
    Base.rand(rng::Random.AbstractRNG, ::Random.SamplerType{NotReal}) = NotReal(rand(rng, Float32))

    x = NotReal.(ones(Float32, 100, 50))
    for dims in (:, 2)
        y = dropout(x, 0.5; dims)
        @test y isa Matrix{NotReal}
        vals = map(a -> a.x, y)
        @test sort(unique(vals)) == [0, 2]
        @test 0.3 < mean(iszero, vals) < 0.7
    end
    @test map(a -> a.x, dropout!(similar(x), x, 0.25)) ⊆ Float32[0, 4/3]
end

@static if NNLIB_TEST_ENZYME

@testset "EnzymeRules: dropout " begin
    rng = Random.default_rng()

    x1 = randn(Float32, 3000, 4000)
    dx1 = zeros(Float32, 3000, 4000)

    dout = randn(Float32, 3000, 4000)

    p = 0.2f0

    forward, reverse = Enzyme.autodiff_thunk(ReverseSplitWithPrimal, typeof(Const(dropout)), Duplicated, typeof(Const(rng)), typeof(Duplicated(x1, dx1)), typeof(Const(0.2f0)))

    tape, primal, shadow = forward(Const(dropout), Const(rng), Duplicated(x1, dx1), Const(p))

    shadow .= dout

    reverse(Const(dropout), Const(rng), Duplicated(x1, dx1), Const(p), tape)

    # The dropout mask isn't exposed through Enzyme's (internal) tape layout, but the
    # dropped entries are exactly the zeros of the primal output (`x1` has no zeros).
    kept = primal .!= 0

    @test dx1[.!kept] ≈ zero(x1)[.!kept]

    val = convert(Float32, 1/(1-p))

    @test dx1[kept] ≈ (val * dout)[kept]
end

end