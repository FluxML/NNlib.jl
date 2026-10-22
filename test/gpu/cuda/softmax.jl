@testset "softmax" begin
    for (sz, dims) in [((5,), :), ((5,), 1), ((5,5), :), ((5,5), 1), ((5,5), 2), ((5,5,5,5), (2,3)), ((5,5,5,5), (2,4))]
        x = randn(Float64, sz)
        dy = randn(Float64, sz)

        y = softmax(x, dims=dims)
        @test test_gradients(x -> softmax(x; dims), x; test_gpu=true)
        gputest(NNlib.∇softmax, dy, y; dims=dims)

        y2 = logsoftmax(x, dims=dims)
        @test test_gradients(x -> logsoftmax(x; dims), x; test_gpu=true)
        gputest(NNlib.∇logsoftmax, dy, y2; dims=dims)

        @test NNlib.∇softmax(dy, y; dims=dims) ≈ collect(∇softmax!(similar(cu(x)), cu(dy), cu(y); dims=dims)) atol=1e-4
        @test NNlib.∇logsoftmax(dy, y2; dims=dims) ≈ collect(∇logsoftmax!(similar(cu(x)), cu(dy), cu(y2); dims=dims)) atol=1e-4
    end
end

@testset "Second derivatives" begin
    # Reverse-over-reverse differentiates ∇softmax/∇logsoftmax through their own rules;
    # forward-over-reverse sends `Dual` arrays to the generic kernels rather than cuDNN.
    # We use a polynomial loss rather than `sum(sin, ...)` as in the CPU test:
    # Zygote's forward-over-reverse `Dual` broadcast for `sin` does not compile
    # to a GPU kernel, which is a limitation unrelated to softmax.
    for f in (softmax, logsoftmax)
        x = randn(Float64, 4, 3)
        loss(z) = sum(abs2, Zygote.gradient(w -> sum(abs2, f(w; dims=1)), z)[1])
        gcpu = Zygote.gradient(loss, x)[1]
        ggpu = Zygote.gradient(loss, CuArray(x))[1]
        @test Array(ggpu) ≈ gcpu rtol=1e-5

        v = randn(Float64, 4, 3)
        hvp(X, V) = ForwardDiff.derivative(ε -> Zygote.gradient(w -> sum(abs2, f(w; dims=1)), X .+ ε .* V)[1], 0.0)
        @test Array(hvp(CuArray(x), CuArray(v))) ≈ hvp(x, v) rtol=1e-5
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

@testset "half precision sums in Float32 (#813)" begin
    # `logsumexp`, and the `dims` cuDNN doesn't take, run the generic kernels.
    err(a, b) = maximum(abs, Float64.(collect(a)) .- b)
    V = 50257
    for T in (Float16, BFloat16)
        T == BFloat16 && CUDA.capability(CUDA.device()) < v"8.0" && continue
        x = T.(cu(0.01f0 .* randn(Float32, 2, V)))  # round on the GPU, see bfloat16.jl
        xd = Float64.(collect(x))
        tol = eps(T(log(V)))  # an ulp of the outputs of `logsoftmax` and `logsumexp`
        for d in (2, :)
            @test eltype(logsumexp(x; dims = d)) == Float32  # not rounded to T (#826)
            @test err(logsumexp(x; dims = d), logsumexp(xd; dims = d)) <= 64eps(Float32(log(V)))
        end
        @test err(logsoftmax(x; dims = 2), logsoftmax(xd; dims = 2)) <= tol
        @test err(sum(Float32.(softmax(x; dims = 2)); dims = 2), 1) < eps(T) / 2
    end
    # Past 65504, a Float16 sum overflows.
    @test collect(logsumexp(CUDA.zeros(Float16, 128256, 1); dims = 1)) == fill(log(128256f0), 1, 1)
end
