using Statistics

@testset "Batchnorm" begin
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
        running_var = var(m, mean=running_mean, dims=[2], corrected=false)
        y_track_stats = batchnorm(v, v, m, running_mean, running_var, 1.0; training=false, track_stats=true)
        # batchnorm without tracked stats should equal bathnorm with tracked stats where the
        # stats are calculated only on the input.
        @test y_no_track_stats ≈ y_track_stats
    end
end
