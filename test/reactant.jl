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

    @testset "batchnorm" begin
        gc, bc = randn(Float32, 5), randn(Float32, 5)
        y = @jit NNlib.batchnorm(Reactant.to_rarray(gc), Reactant.to_rarray(bc), rx)
        @test eltype(y) == BFloat16
        @test close(y, NNlib.batchnorm(gc, bc, x))
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

# Under Reactant, `batchnorm` with the statistics of the batch lowers to
# `stablehlo.batch_norm_training`, and the other normalizations take the variance about a
# detached mean (see `NNlibReactantExt`).
@testset "Reactant: normalization" begin
    x = randn(Float32, 6, 5, 4, 3)
    g, b = 1 .+ randn(Float32, 4) ./ 10, randn(Float32, 4) ./ 10
    rx, rg, rb = Reactant.to_rarray.((x, g, b))
    close(a, b) = isapprox(Array(a), b; rtol=1f-4, atol=1f-5)

    @testset "batchnorm" begin
        μ, σ² = zeros(Float32, 4), ones(Float32, 4)
        rμ, rσ² = Reactant.to_rarray.((μ, σ²))
        y = @jit NNlib.batchnorm(rg, rb, rx, rμ, rσ², 0.1f0)
        @test close(y, NNlib.batchnorm(g, b, x, μ, σ², 0.1f0))
        @test close(rμ, μ)
        @test close(rσ², σ²)
        @test close(@jit(NNlib.batchnorm(rg, rb, rx, rμ, rσ², 0.1f0; training=false)),
                    NNlib.batchnorm(g, b, x, μ, σ², 0.1f0; training=false))
        @test close(@jit(NNlib.batchnorm(nothing, nothing, rx)), NNlib.batchnorm(nothing, nothing, x))
    end

    @testset "$name gradient" for (name, f) in (
            ("batchnorm", (g, b, x) -> sum(abs2, NNlib.batchnorm(g, b, x) .- x)),
            ("instancenorm", (g, b, x) -> sum(abs2, NNlib.instancenorm(g, b, x) .- x)),
            ("groupnorm", (g, b, x) -> sum(abs2, NNlib.groupnorm(g, b, x, 2) .- x)))
        grad(g, b, x) = Enzyme.gradient(Enzyme.Reverse, Enzyme.Const(f), g, b, x)
        for (d, d0) in zip(@jit(grad(rg, rb, rx)), Zygote.gradient(f, g, b, x))
            @test close(d, d0)
        end
    end
end
