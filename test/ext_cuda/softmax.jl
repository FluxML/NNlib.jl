@testset "softmax" begin
    for (sz, dims) in [((5,), :), ((5,), 1), ((5,5), :), ((5,5), 1), ((5,5), 2), ((5,5,5,5), (2,3)), ((5,5,5,5), (2,4))]
        x = randn(Float64, sz)
        dy = randn(Float64, sz)

        y = softmax(x, dims=dims)
        gputest(softmax, x, dims=dims)
        gputest(NNlib.∇softmax_data, dy, y; dims=dims)

        y2 = logsoftmax(x, dims=dims)
        gputest(logsoftmax, x, dims=dims)
        gputest(NNlib.∇logsoftmax_data, dy, y2; dims=dims)

        # From NNlib 0.8.3, ∇softmax! is not used in the gradient.
        # But NNlibCUDA still knows how to call cuDNN routines, let's test they agree:
        @test NNlib.∇softmax_data(dy, y; dims=dims) ≈ collect(∇softmax!(similar(cu(x)), cu(dy), cu(x), cu(y); dims=dims)) atol=1e-4
        @test NNlib.∇logsoftmax_data(dy, y2; dims=dims) ≈ collect(∇logsoftmax!(similar(cu(x)), cu(dy), cu(x), cu(y2); dims=dims)) atol=1e-4
        # (Note that ∇softmax! does not depend on x, it's just there to disambiguate from an even older signature.)
    end
end

@testset "softmax with masked input (fixes #506)" begin
    # The cuDNN FAST algorithm overflows to NaN on masked/large-magnitude inputs,
    # so softmax must always use the ACCURATE algorithm regardless of CUDA.math_mode.
    # This reproduces the attention-mask scenario from the issue (fill with -1000).
    orig = CUDA.math_mode()
    try
        for mode in (CUDA.DEFAULT_MATH, CUDA.FAST_MATH)
            CUDA.math_mode!(mode)

            # (a) large finite negative mask, as in #506. Outputs stay finite, so we
            #     can compare against the CPU reference directly.
            x = randn(Float32, 128, 32)
            x[1:64, :] .= -1f3        # masked entries
            gx = cu(x)

            gy = softmax(gx; dims=1)
            @test all(isfinite, collect(gy))
            @test collect(gy) ≈ softmax(x; dims=1) atol=1e-4

            gly = logsoftmax(gx; dims=1)
            @test all(isfinite, collect(gly))
            @test collect(gly) ≈ logsoftmax(x; dims=1) atol=1e-4

            # (b) hard -Inf mask. softmax must give exactly-0 (finite) at masked
            #     positions; logsoftmax gives -Inf there (never NaN). Compare only
            #     at the unmasked positions to avoid Inf-Inf=NaN inside `isapprox`.
            xi = randn(Float32, 128, 32)
            mask = falses(128, 32)
            mask[1:64, :] .= true
            xi[mask] .= -Inf32
            gxi = cu(xi)

            gyi = softmax(gxi; dims=1)
            @test all(isfinite, collect(gyi))
            @test all(collect(gyi)[mask] .== 0)
            @test collect(gyi) ≈ softmax(xi; dims=1) atol=1e-4

            glyi = collect(logsoftmax(gxi; dims=1))
            @test !any(isnan, glyi)
            @test glyi[.!mask] ≈ logsoftmax(xi; dims=1)[.!mask] atol=1e-4
        end
    finally
        CUDA.math_mode!(orig)
    end
end
