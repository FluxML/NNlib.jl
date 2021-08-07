using NNlib, Zygote, ForwardDiff

ACTIVATION_FUNCTIONS = 
    [@eval($a) for a in NNlib.ACTIVATIONS]

@testset "dense_bias_act" begin
    w = randn(3,4)
    x = randn(4,5)
    b = randn(3)
    wx = w * x

    @testset "activation $fun" for fun in [identity, relu, tanh, cbrt]
        @test dense_bias_act(fun, w, x, b) ≈ fun.((w * x) .+ b)
        @test dense_bias_act(fun, w, x, zero(b)) ≈ fun.((w * x))
        @test dense_bias_act(fun, w, x, false) ≈ fun.((w * x))
        @test dense_bias_act(fun, w, x) ≈ fun.((w * x))

        gx = ForwardDiff.gradient(x -> sum(dense_bias_act(fun, w, x, b)), x)
        @test gx ≈ Zygote.gradient(x -> sum(dense_bias_act(fun, w, x, b)), x)[1]

        jx = ForwardDiff.jacobian(x -> dense_bias_act(fun, w, x, b), x)
        # This is a test that it's safe to evaluate the pullback more than once:
        @test jx ≈ Zygote.jacobian(x -> dense_bias_act(fun, w, x, b), x)[1]
    end
end

@testset "bias_act!" begin
    x = randn(2,5)
    b = randn(2)
    @test bias_act!(identity, copy(x), b) ≈ (x .+ b)
    @test bias_act!(relu, copy(x), b) ≈ relu.(x .+ b)
    @test bias_act!(tanh, copy(x), b) ≈ tanh.(x .+ b)

    @testset "gradient with $fun" for fun in vcat([identity, tanh, cbrt], ACTIVATION_FUNCTIONS)
        # Only some of these go the fast path, `cbrt` is an example of a function NNlib knows nothing about.
        fun == rrelu && continue # this one is randomised!

        @test bias_act!(fun, copy(x), b) ≈ fun.(x .+ b)
        @test bias_act!(fun, copy(x), false) ≈ fun.(x)
        @test bias_act!(fun, copy(x)) ≈ fun.(x)

        gx = ForwardDiff.gradient(x -> sum(bias_act!(fun, copy(x), b)), x)
        @test gx ≈ Zygote.gradient(x -> sum(bias_act!(fun, copy(x), b)), x)[1]

        gx2 = ForwardDiff.gradient(x -> sum(bias_act!(fun, copy(x))), x)
        @test gx2 ≈ Zygote.gradient(x -> sum(bias_act!(fun, copy(x))), x)[1]
        @test gx2 ≈ Zygote.gradient(x -> sum(bias_act!(fun, copy(x), false)), x)[1]

        gb = ForwardDiff.gradient(b -> sum(bias_act!(fun, copy(x), b)), b)
        @test gb ≈ Zygote.gradient(b -> sum(bias_act!(fun, copy(x), b)), b)[1]

        @test Zygote.gradient(b -> sum(bias_act!(fun, copy(x), b)), false) == (nothing,)
        @test Zygote.gradient(b -> sum(bias_act!(fun, copy(x), b)), b .> 0) == (nothing,)
    end
end