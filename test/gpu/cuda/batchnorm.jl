@testset "Batchnorm" begin
    @testset "Mooncake" begin
        rng = Random.MersenneTwister(42)
        N, B = 3, 4
        g  = CUDA.ones(Float32, N)
        b  = CUDA.zeros(Float32, N)
        x4 = CUDA.randn(Float32, 1, 1, N, B)
        # Use nothing for running stats to avoid in-place mutation across test_rule calls.
        _bn(g, b, x) = sum(batchnorm(g, b, x, nothing, nothing, 0.1f0; training=true))
        test_rule(rng, _bn, g, b, x4; is_primitive=false, mode=Mooncake.ReverseMode)
    end
    v = CUDA.rand(Float32, 2)
    m = CUDA.rand(Float32, 2, 5)

    @testset for training in (true, false), track_stats in (true, false)
        kws = (training=training, track_stats=track_stats)

        # Normal
        batchnorm(v, v, m, v, v, 1.0; kws...)
        ∇batchnorm(v, v, m, m, v, v, 1.0; kws...)

        # No affine
        batchnorm(nothing, nothing, m, v, v, 1.0; kws...)
        ∇batchnorm(nothing, nothing, m, m, v, v, 1.0; kws...)

        # No tracking
        batchnorm(v, v, m, nothing, nothing, 1.0; kws...)
        ∇batchnorm(v, v, m, m, nothing, nothing, 1.0; kws...)

        # Both or neither tracked or affine params must be set
        for (α, β) in ((v, nothing), (nothing, v))
            @test_throws MethodError batchnorm(α, β, m, v, v, 1.0; kws...)
            @test_throws MethodError ∇batchnorm(α, β, m, m, v, v, 1.0; kws...)
            @test_throws ArgumentError batchnorm(v, v, m, α, β, 1.0; kws...)
        end
    end 
    @testset "test mode" begin
        y_no_track_stats = batchnorm(v, v, m, nothing, nothing, 1.0; training=false, track_stats=false)
        running_mean = mean(m, dims=[2])
        running_var = Statistics.var(m, mean=running_mean, dims=[2], corrected=false)
        y_track_stats = batchnorm(v, v, m, running_mean, running_var, 1.0; training=false, track_stats=true)
        # batchnorm without tracked stats should equal bathnorm with tracked stats where the
        # stats are calculated only on the input.
        @test y_no_track_stats ≈ y_track_stats
    end

    @testset "Float16" begin
        x = CUDA.rand(Float16, 1, 1, 3, 4)
        g = CUDA.rand(Float16, 3)
        b = CUDA.rand(Float16, 3)
        running_mean = CUDA.zeros(Float16, 3)
        running_var = CUDA.ones(Float16, 3)
        ext = Base.get_extension(NNlib, :NNlibCUDACUDNNExt)
        cache = ext.BNCache()
        y = batchnorm(g, b, x, running_mean, running_var, 0.1f0; training=true,
                      track_stats=true, cache)

        x32 = Float32.(Array(x))
        mean = sum(x32; dims=(1, 2, 4)) ./ 4
        var = sum(abs2, x32 .- mean; dims=(1, 2, 4)) ./ 4
        g32 = reshape(Float32.(Array(g)), 1, 1, 3, 1)
        b32 = reshape(Float32.(Array(b)), 1, 1, 3, 1)
        ref = @. g32 * (x32 - mean) / sqrt(var + 1f-5) + b32
        @test eltype(y) == Float16
        @test Float32.(Array(y)) ≈ ref rtol=2f-3 atol=2f-3
        @test eltype(cache.mean) == Float32
        @test eltype(cache.ivar) == Float32
        @test eltype(running_mean) == Float16
        @test eltype(running_var) == Float16

        yi = batchnorm(g, b, x, running_mean, running_var, 0.1f0; training=false,
                       track_stats=true)
        running_mean32 = reshape(Float32.(Array(running_mean)), 1, 1, 3, 1)
        running_var32 = reshape(Float32.(Array(running_var)), 1, 1, 3, 1)
        inference_ref = @. g32 * (x32 - running_mean32) /
                           sqrt(running_var32 + 1f-5) + b32
        @test Float32.(Array(yi)) ≈ inference_ref rtol=2f-3 atol=2f-3

        dy = CUDA.rand(Float16, size(x))
        dg, db, dx = ∇batchnorm(g, b, x, dy, running_mean, running_var, 0.1f0;
                                   training=true, track_stats=true, cache)
        @test eltype(dg) == Float16
        @test eltype(db) == Float16
        @test eltype(dx) == Float16

        dy32 = Float32.(Array(dy))
        invvar = @. inv(sqrt(var + 1f-5))
        xhat = @. (x32 - mean) * invvar
        db_ref = sum(dy32; dims=(1, 2, 4))
        dg_ref = sum(dy32 .* xhat; dims=(1, 2, 4))
        dx_ref = @. g32 * invvar / 4 * (4 * dy32 - db_ref - xhat * dg_ref)
        @test Float32.(Array(dg)) ≈ vec(dg_ref) rtol=3f-3 atol=3f-3
        @test Float32.(Array(db)) ≈ vec(db_ref) rtol=3f-3 atol=3f-3
        @test Float32.(Array(dx)) ≈ dx_ref rtol=3f-3 atol=3f-3
    end
end
