using CuArrays
using NNlib: @fix

"""https://github.com/FluxML/NNlib.jl/issues/37"""
@testset "fix#37" begin
    f1(x) = @fix x.*log.(x)

    function f2(x)
        return @fix x.*log.(x)
    end

    function f3(x)
        y = @fix x.*log.(x)
        y
    end

    @testset "$(summary(d))" for d in [[2.0], [2.f0], collect(1.0:10)]
        @testset "$f" for f in [f1, f2, f3]
            @test f(d) ≈ Array(f(CuArray(d)))
        end
    end
end
