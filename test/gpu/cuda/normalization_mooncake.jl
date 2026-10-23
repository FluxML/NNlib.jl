@testset "Normalization under Mooncake" begin
    # Mooncake runs the `rrule`s (`NNlibMooncakeExt`), with the fused kernels and cuDNN, so
    # its gradients are Zygote's.
    T = Float32
    x4 = CUDA.randn(T, 4, 5, 6, 3); x3 = CUDA.randn(T, 5, 6, 3); x2 = CUDA.randn(T, 6, 8)
    c = CUDA.randn(T, 6); d = CUDA.randn(T, 6)
    rm = CUDA.randn(T, 6); rv = CUDA.rand(T, 6) .+ 1
    @testset "$name" for (name, f, args) in (
            ("batchnorm", (g, b, x) -> batchnorm(g, b, x), (c, d, x4)),
            ("batchnorm nothing", x -> batchnorm(nothing, nothing, x; eps=1f-3), (x4,)),
            ("batchnorm 3-D", (g, b, x) -> batchnorm(g, b, x), (c, d, x3)),
            ("batchnorm 2-D", (g, b, x) -> batchnorm(g, b, x), (c, d, x2)),
            ("batchnorm 2-D nothing", x -> batchnorm(nothing, nothing, x), (x2,)),
            ("batchnorm training", (g, b, x) -> batchnorm(g, b, x, rm, rv, 0.1f0; training=true), (c, d, x4)),
            ("batchnorm inference", (g, b, x) -> batchnorm(g, b, x, rm, rv, 0.1f0; training=false), (c, d, x4)),
            ("batchnorm 2-D inference", (g, b, x) -> batchnorm(g, b, x, rm, rv, 0.1f0; training=false), (c, d, x2)),
            ("instancenorm", (g, b, x) -> instancenorm(g, b, x), (c, d, x4)),
            ("instancenorm nothing", x -> instancenorm(nothing, nothing, x; eps=1f-3), (x4,)),
            ("instancenorm training", (g, b, x) -> instancenorm(g, b, x, rm, rv, 0.1f0; track_stats=true), (c, d, x4)),
            ("instancenorm inference", (g, b, x) -> instancenorm(g, b, x, rm, rv, 0.1f0; training=false), (c, d, x4)),
            ("groupnorm", (g, b, x) -> groupnorm(g, b, x, 3), (c, d, x4)),
            ("groupnorm nothing", x -> groupnorm(nothing, nothing, x, 2; eps=1f-3), (x4,)),
            ("layernorm", (g, b, x) -> layernorm(g, b, x), (c, d, x2)),
            ("layernorm dims", (g, b, x) -> layernorm(g, b, x; dims=1:2), (x2[1:4, 1:5], x2[2:5, 1:5], x4)),
            ("layernorm generic", (g, b, x) -> layernorm(g, b, x; dims=3), (reshape(c, 1, 1, 6), reshape(d, 1, 1, 6), x4)),
            ("normalise", x -> NNlib.normalise(x; dims=1), (x2,)),
            ("rmsnorm", (g, x) -> rmsnorm(g, x), (c, x2)),
            ("rmsnorm nothing", x -> rmsnorm(nothing, x; dims=1:2), (x4,)))
        w = CUDA.randn(T, size(f(args...)))
        loss(xs...) = sum(w .* f(xs...))
        for (m, z) in zip(withgradient(loss, AutoMooncake(), args...).grad, Zygote.gradient(loss, args...))
            @test Array(m) ≈ Array(z) rtol=1e-4 atol=1e-5
        end
    end

    # A gradient call evaluates the function once, and updates the statistics once.
    @testset "running stats, $name" for (name, norm, x) in (("batchnorm", batchnorm, x4),
            ("batchnorm 2-D", batchnorm, x2), ("instancenorm", instancenorm, x4))
        w = CUDA.randn(T, size(x))
        loss(g, b, x, rm, rv) = sum(w .* norm(g, b, x, rm, rv, 0.1f0; training=true, track_stats=true))
        rm0 = CUDA.zeros(T, 6); rv0 = CUDA.ones(T, 6)
        loss(c, d, x, rm0, rv0)
        rm1 = CUDA.zeros(T, 6); rv1 = CUDA.ones(T, 6)
        Mooncake.value_and_gradient!!(Mooncake.build_rrule(loss, c, d, x, rm1, rv1), loss, c, d, x, rm1, rv1)
        @test Array(rm1) ≈ Array(rm0)
        @test Array(rv1) ≈ Array(rv0)
    end
end
