@testset "activation broadcast" begin
    for f in NNlib.ACTIVATIONS
        if f ∉ [:rrelu]
            @eval gputest(x -> $f.(x), rand(Float64, 5))
        end
    end
end

@testset "forward diff" begin
    for f in NNlib.ACTIVATIONS
        if f ∉ [:rrelu]
            @eval gputest(x -> $f.(x), Dual.(rand(5), 1))
        end
    end
end

# Broadcasting over complex CuArray works without NNlibCUDA, this test checks that
# NNlibCUDA does not cause such operations to take a fast path which does not support
# complex numbers (e.g. CUDNN)
@testset "complex" begin
    f(x) = tanh.(x)
    cs = rand(ComplexF64, 5)
    @test f(cs) ≈ collect(f(CuArray(cs)))
end

@testset "softplus" begin 
  # softplus does not give `Inf` for large arguments
   x = CuArray([1000.])
   @test all(softplus.(x) .== x)
end

@testset "input is preserved" begin
    x = CUDA.ones(1)
    @test Array(x) == [1f0]
    tanh.(x)
    @test Array(x) == [1f0]
    y = tanh.(x)
    @test Array(x) == [1f0]
    @test Array(y) == [tanh(1f0)]
    x .= tanh.(y)
    @test Array(y) == [tanh(1f0)]
    @test Array(x) == [tanh(tanh(1f0))]
end
