@testset "Compare CPU & GPU" begin
    for (T, atol) in ((Float16, 1.0f-2), (Float32, 1.0f-5))
        @testset "ndims: $(ndims(x))" for x in (randn(T, 16), randn(T, ntuple(_ -> 2, 5)...), randn(T, ntuple(_ -> 2, 6)...))
            @test test_gradients(x -> NNlib.relu.(x), x; test_gpu=true, atol)
            @test test_gradients(x -> NNlib.relu6.(x), x; test_gpu=true, atol)
            @test test_gradients(x -> NNlib.softplus.(x), x; test_gpu=true, atol)
            @test test_gradients(x -> tanh.(x), x; test_gpu=true, atol)
            @test test_gradients(x -> identity.(x), x; test_gpu=true, atol)
        end
    end
end
