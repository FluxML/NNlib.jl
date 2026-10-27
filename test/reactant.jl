# Run with `NNLIB_TEST_REACTANT=true`, which needs Reactant in the test environment (see
# `.github/workflows/ci.yml`).
using Reactant: Reactant, @jit
using Enzyme: Enzyme
using Zygote: Zygote

# Reactant traces `BFloat16` as `TracedRNumber{BFloat16}`, for which `NNlibReactantExt`
# keeps the half-precision contract of the untraced functions.
@testset "Reactant: half precision" begin
    x = BFloat16.(randn(Float32, 8, 5, 3))
    g, b = randn(Float32, 8), randn(Float32, 8)
    rx, rg, rb = Reactant.to_rarray.((x, g, b))
    close(a, b) = isapprox(Float32.(Array(a)), Float32.(b); rtol=2f-2)

    @testset "layernorm" begin
        y = @jit NNlib.layernorm(rg, rb, rx)
        @test eltype(y) == BFloat16
        @test close(y, NNlib.layernorm(g, b, x))
        @test_throws ArgumentError @jit NNlib.layernorm(Reactant.to_rarray(BFloat16.(g)), rb, rx)
    end

    @testset "rmsnorm" begin
        y = @jit NNlib.rmsnorm(rg, rx)
        @test eltype(y) == BFloat16
        @test close(y, NNlib.rmsnorm(g, x))
    end

    @testset "logsumexp" begin
        y = @jit NNlib.logsumexp(rx; dims=1)
        @test eltype(y) == Float32
        @test close(y, NNlib.logsumexp(x; dims=1))
    end

    @testset "layernorm gradient" begin
        f(g, b, x) = sum(abs2, Float32.(NNlib.layernorm(g, b, x)))
        grad(g, b, x) = Enzyme.gradient(Enzyme.Reverse, Enzyme.Const(f), g, b, x)
        dg, db, dx = @jit grad(rg, rb, rx)
        @test eltype(dg) == Float32
        @test eltype(dx) == BFloat16
        for (d, d0) in zip((dg, db, dx), Zygote.gradient(f, g, b, x))
            @test close(d, d0)
        end
    end
end
