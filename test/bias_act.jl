using NNlib, Zygote, Test
using Zygote: ForwardDiff

ACTIVATION_FUNCTIONS = 
    [@eval($a) for a in NNlib.ACTIVATIONS]

@testset "bias_act!" begin
    x = randn(3,4)
    b = randn(3)
    @test bias_act!(identity, copy(x), b) ≈ (x .+ b)
    @test bias_act!(relu, copy(x), b) ≈ relu.(x .+ b)
    @test bias_act!(tanh, copy(x), b) ≈ tanh.(x .+ b)

    @testset "gradient with $fun" for fun in vcat([identity, tanh, cbrt], 
                                                    ACTIVATION_FUNCTIONS,
                                                    [x->x, x -> 1/(x^2+2), x -> leakyrelu(x, 0.33)])
        # Only some of these go the fast path, `cbrt` is an example of a function NNlib knows nothing about.
        fun == rrelu && continue # this one is randomised!

        @test bias_act!(fun, copy(x), b) ≈ fun.(x .+ b)
        @test bias_act!(fun, copy(x), false) ≈ fun.(x)

        gx = ForwardDiff.gradient(x -> sum(bias_act!(fun, copy(x), b)), x)
        @test gx ≈ Zygote.gradient(x -> sum(bias_act!(fun, copy(x), b)), x)[1]

        gx2 = ForwardDiff.gradient(x -> sum(bias_act!(fun, copy(x), false)), x)
        @test gx2 ≈ Zygote.gradient(x -> sum(bias_act!(fun, copy(x), false)), x)[1]

        gb = ForwardDiff.gradient(b -> sum(bias_act!(fun, copy(x), b)), b)
        @test gb ≈ Zygote.gradient(b -> sum(bias_act!(fun, copy(x), b)), b)[1]

        @test Zygote.gradient(b -> sum(bias_act!(fun, copy(x), b)), false) == (nothing,)
        @test Zygote.gradient(b -> sum(bias_act!(fun, copy(x), b)), b .> 0) == (nothing,)
    end

    @testset "gradient for fast_broadcast_plus!" begin
        # Gradient definition is just to disable mutation inside 2nd order AD
        gx = ForwardDiff.gradient(x -> sum(NNlib.fast_broadcast_plus!(cbrt, copy(x), b)), x)
        @test gx ≈ Zygote.gradient(x -> sum(NNlib.fast_broadcast_plus!(cbrt, copy(x), b)), x)[1]
    end
end
