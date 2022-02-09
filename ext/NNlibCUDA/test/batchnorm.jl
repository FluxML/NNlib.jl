@testset "Batchnorm" begin
    v = CUDA.rand(Float32, 2)
    m = CUDA.rand(Float32, 2, 5)

    @testset for training in (true, false), track_stats in (true, false)
        kws = (training=training, track_stats=track_stats)

        # Normal
        NNlibCUDA.batchnorm(v, v, m, v, v, 1.0; kws...)
        NNlibCUDA.∇batchnorm(v, v, m, m, v, v, 1.0; kws...)

        # No affine
        NNlibCUDA.batchnorm(nothing, nothing, m, v, v, 1.0; kws...)
        NNlibCUDA.∇batchnorm(nothing, nothing, m, m, v, v, 1.0; kws...)

        # No tracking
        NNlibCUDA.batchnorm(v, v, m, nothing, nothing, 1.0; kws...)
        NNlibCUDA.∇batchnorm(v, v, m, m, nothing, nothing, 1.0; kws...)

        # Both or neither tracked or affine params must be set
        for (α, β) in ((v, nothing), (nothing, v))
            @test_throws MethodError NNlibCUDA.batchnorm(α, β, m, v, v, 1.0; kws...)
            @test_throws MethodError NNlibCUDA.∇batchnorm(α, β, m, m, v, v, 1.0; kws...)
            @test_throws ArgumentError NNlibCUDA.batchnorm(v, v, m, α, β, 1.0; kws...)
        end
    end 
end
