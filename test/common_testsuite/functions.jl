function functions_testsuite(Backend)
    device(x) = adapt(Backend(), x)
    test_gpu = Backend != CPU
    types = nameof(Backend) === :MetalBackend ? (Float32, Float16) : (Float32, Float64, Float16)

    @testset "glu/swiglu ($T)" for T in types
        x = device(T.(randn(StableRNG(17), Float32, 6, 4, 3)))
        xc = Float64.(cpu(x))
        for dim in (1, 2)
            y = NNlib.swiglu(x, dim)
            @test eltype(y) == T
            @test cpu(y) ≈ NNlib.swiglu(xc, dim) rtol=sqrt(eps(T))
            @test cpu(NNlib.glu(x, dim)) ≈ NNlib.glu(xc, dim) rtol=sqrt(eps(T))
        end
    end

    @testset "gradients" begin
        x = randn(StableRNG(17), Float32, 4, 3)
        @test test_gradients(NNlib.glu, x; test_gpu)
        @test test_gradients(NNlib.swiglu, x; test_gpu)
        @test test_gradients(x -> NNlib.swiglu(x, 2), permutedims(x); test_gpu)
    end
end
