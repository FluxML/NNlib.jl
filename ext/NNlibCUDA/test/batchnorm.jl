@testset "Batchnorm" begin
    v = CUDA.rand(Float32, 2)
    m = CUDA.rand(Float32, 2, 5)
    for training in (false, true)
        NNlibCUDA.batchnorm(v, v, m, v, v, 1.0; training=training)
        NNlibCUDA.∇batchnorm(v, v, m, m, v, v, 1.0; training=training)
    end

    @testset "track_stats=false" begin
        for training in (false, true)
            NNlibCUDA.batchnorm(v, v, m, nothing, nothing, 1.0; training=training)
            NNlibCUDA.∇batchnorm(v, v, m, m, nothing, nothing, 1.0; training=training)
        end 
    end
end
