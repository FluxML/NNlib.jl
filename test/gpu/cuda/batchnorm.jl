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
            @test_throws ArgumentError batchnorm(α, β, m, v, v, 1.0; kws...)
            @test_throws ArgumentError ∇batchnorm(α, β, m, m, v, v, 1.0; kws...)
            @test_throws ArgumentError batchnorm(v, v, m, α, β, 1.0; kws...)
        end
    end 
    @testset "3D input (issue #753)" begin
        # cuDNN batchnorm supports only 4D/5D descriptors; a 3D (W, C, N) input is
        # reshaped to 4D (1, W, C, N) on both the forward and backward paths. Guards
        # against the regression where the backward path errored with BAD_PARAM.
        g3 = rand(Float32, 4)
        b3 = rand(Float32, 4)
        x3 = rand(Float32, 5, 4, 8)
        _bn3(g, b, x) = batchnorm(g, b, x, nothing, nothing, 0.1f0; training=true)
        gputest(_bn3, g3, b3, x3; rtol=1e-3, atol=1e-4)
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
end
