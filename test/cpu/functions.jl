@testset "glu" begin
    x = [1 2 3; 4 5 6; 7 8 9; 10 11 12]	 
    @test ceil.(NNlib.glu(x, 1)) == [1 2 3; 4 5 6]
    @test_throws AssertionError NNlib.glu(x, 2)

    y = randn(StableRNG(17), Float32, 6, 4, 3)
    @test NNlib.glu(y, 1) ≈ y[1:3, :, :] .* sigmoid.(y[4:6, :, :])
    @test NNlib.glu(y, 2) ≈ y[:, 1:2, :] .* sigmoid.(y[:, 3:4, :])
    @test @inferred(NNlib.glu(y)) isa Array{Float32,3}
    @test @inferred(NNlib.glu(y, rand(1:2))) isa Array{Float32,3}
end

@testset "swiglu" begin
    x = randn(StableRNG(17), 6, 4, 3)
    @test NNlib.swiglu(x) ≈ swish.(x[1:3, :, :]) .* x[4:6, :, :]
    @test NNlib.swiglu(x, 1) == NNlib.swiglu(x)
    @test NNlib.swiglu(x, 2) ≈ swish.(x[:, 1:2, :]) .* x[:, 3:4, :]
    @test @inferred(NNlib.swiglu(x, rand(1:2))) isa Array{Float64,3}
    @test_throws AssertionError NNlib.swiglu(x, 3)
    @test_throws AssertionError NNlib.swiglu(x, 4)

    # the gate is the first half, unlike in `glu`
    y = [0.0, 1.0, 2.0, 3.0]
    @test NNlib.swiglu(y) ≈ [0.0, swish(1.0) * 3.0]
    @test NNlib.swiglu(y) != NNlib.glu(y)

    for T in (Float32, Float16)
        xt = T.(x)
        @test @inferred(NNlib.swiglu(xt)) isa Array{T,3}
        @test NNlib.swiglu(xt) ≈ T.(NNlib.swiglu(Float64.(xt))) rtol=sqrt(eps(T))
    end
end

@testset "AutoDiff" begin
    local rng = StableRNG(17)
    @test test_gradients(NNlib.glu, rand(rng, 4, 3))
    @test test_gradients(NNlib.swiglu, rand(rng, 4, 3))
    @test test_gradients(x -> NNlib.swiglu(x, 2), rand(rng, 3, 4))
end
