using NNlib, Zygote, Test
using Zygote: ForwardDiff

ACTIVATION_FUNCTIONS =
    [@eval($a) for a in NNlib.ACTIVATIONS]

@testset "bias_act!" begin
    x = randn(3,4)
    b = randn(3)
    @test @inferred(bias_act!(identity, x, false)) === x  # pass-through
    @test @inferred(bias_act!(identity, copy(x), b)) ≈ (x .+ b)
    @test @inferred(bias_act!(relu, copy(x), b)) ≈ relu.(x .+ b)
    @test @inferred(bias_act!(tanh, copy(x), b)) ≈ tanh.(x .+ b)
    @test @inferred(bias_act!(tanh, copy(x), false)) ≈ tanh.(x)

    # Check that it does overwrite:
    x32 = rand(Float32, 3, 4)
    x32copy = copy(x32)
    @test @inferred(bias_act!(cbrt, x32, b)) ≈ cbrt.(x32copy .+ b)
    @test x32  ≈ cbrt.(x32copy .+ b)
    x32 = rand(Float32, 3, 4)
    x32copy = copy(x32)
    @test @inferred(bias_act!(tanh, x32, false)) ≈ tanh.(x32copy)
    @test x32  ≈ tanh.(x32copy)

    # Check that it doesn't try to overwrite non-float arrays:
    xint = rand(-3:3, 3, 4)
    bint = rand(-2:2, 3)
    @test bias_act!(identity, copy(xint), bint) ≈ xint .+ bint
    @test bias_act!(tanh, copy(xint), bint) ≈ tanh.(xint .+ bint)
    @test bias_act!(tanh, copy(xint), false) ≈ tanh.(xint)

    # Reject bias===true so that Bool means one thing:
    @test_throws Exception bias_act!(identity, rand(3), true)
    @test_throws Exception bias_act!(cbrt, rand(3), true)
    @test_throws Exception bias_act!(cbrt, rand(1:3, 3), true)

    @testset "gradient with $fun" for fun in vcat([identity, tanh, cbrt],
                                                    ACTIVATION_FUNCTIONS,
                                                    [x->x, x -> 1/(x^2+2), x -> leakyrelu(x, 0.33)])
        # Only some of these go the fast path, `cbrt` is an example of a function NNlib knows nothing about.
        fun == rrelu && continue # this one is randomised!

        @test bias_act!(fun, copy(x), b) ≈ fun.(x .+ b)
        @test bias_act!(fun, copy(x), false) ≈ fun.(x)

        gx = ForwardDiff.gradient(x -> sum(bias_act!(fun, copy(x), b)), x)
        gxplus = ForwardDiff.gradient(x -> sum(bias_act!(fun, copy(x), b)), x .+ eps())
        gxminus = ForwardDiff.gradient(x -> sum(bias_act!(fun, copy(x), b)), x .- eps())
        if !(gx ≈ gxplus ≈ gxminus)
            @warn "skipping gradient tests due to discontinuity" fun x b
            continue
        end
        @test gx ≈ Zygote.gradient(x -> sum(bias_act!(fun, copy(x), b)), x)[1]

        gx2 = ForwardDiff.gradient(x -> sum(bias_act!(fun, copy(x), false)), x)
        gx2plus = ForwardDiff.gradient(x -> sum(bias_act!(fun, copy(x), false)), x .- eps())
        gx2minus = ForwardDiff.gradient(x -> sum(bias_act!(fun, copy(x), false)), x .- eps())
        if !(gx2 ≈ gx2plus ≈ gx2minus)
            @warn "skipping gradient tests due to discontinuity" fun x
            continue
        end
        @test gx2 ≈ Zygote.gradient(x -> sum(bias_act!(fun, copy(x), false)), x)[1]

        gb = ForwardDiff.gradient(b -> sum(bias_act!(fun, copy(x), b)), b)
        @test gb ≈ Zygote.gradient(b -> sum(bias_act!(fun, copy(x), b)), b)[1]

        @test Zygote.gradient(b -> sum(bias_act!(fun, copy(x), b)), false) == (nothing,)
        @test Zygote.gradient(b -> sum(bias_act!(fun, copy(x), b)), b .> 0) == (nothing,)
    end

    @testset "gradient for fast_broadcast!" begin
        # Gradient definition is just to disable mutation inside 2nd order AD
        gx = ForwardDiff.gradient(x -> sum(NNlib._fast_broadcast!(cbrt∘(+), copy(x), b)), x)
        @test gx ≈ Zygote.gradient(x -> sum(NNlib._fast_broadcast!(cbrt∘(+), copy(x), b)), x)[1]

        # relu should take the fast path
        g2 = ForwardDiff.gradient(x) do x
            sum(abs2, Zygote.gradient(x -> sum(abs2, bias_act!(relu, copy(x), b)), x)[1])
        end
        @test_broken gx ≈ Zygote.gradient(x) do x
            sum(abs2,Zygote. gradient(x -> sum(abs2, bias_act!(relu, copy(x), b)), x)[1])
        end
        # Can't differentiate foreigncall expression $(Expr(:foreigncall, :(:jl_eqtable_get), Any, svec(Any, Any, Any), 0, :(:ccall), %5, %3, %4)).
        # [5] (::typeof(∂(accum_global)))(Δ::Nothing)
        @test g2 ≈ Zygote.gradient(x, b) do x, b
            sum(abs2, Zygote.gradient((x, b) -> sum(abs2, bias_act!(relu, copy(x), b)), x, b)[1])
        end[1]

       g3 = ForwardDiff.gradient(x) do x
            sum(abs2, Zygote.gradient((x, b) -> sum(abs2, bias_act!(swish, copy(x), b)), x, b)[1])
        end
        @test g3 ≈ Zygote.gradient(x, b) do x, b
            sum(abs2, Zygote.gradient((x, b) -> sum(abs2, bias_act!(swish, copy(x), b)), x, b)[1])
        end[1]

        # Anon function sure to take the generic path
        g4 = ForwardDiff.gradient(x) do x
            sum(abs2, Zygote.gradient((x, b) -> sum(abs2, bias_act!(y -> cbrt(y/3), copy(x), b)), x, b)[1])
        end
        @test g4 ≈ Zygote.gradient(x, b) do x, b
            sum(abs2, Zygote.gradient((x, b) -> sum(abs2, bias_act!(y -> cbrt(y/3), copy(x), b)), x, b)[1])
        end[1]
    end
end

