@testset "activation broadcast" begin
    for name in NNlib.ACTIVATIONS
        # println("Testing forward diff for activation: ", name)
        f = @eval $name
        @test test_gradients(x -> f.(x), rand(5); test_gpu=true)
    end
end

# Up to large inputs, where the fast `tanh` of Metal gives NaN, and in half precision,
# which is computed in Float32.
@testset "values: $T" for T in (Float32, Float16)
    x = T.(range(-50f0, 50f0, length=201))
    for name in NNlib.ACTIVATIONS
        name == :rrelu && continue
        f = @eval $name
        y = f.(MtlArray(x))
        @test eltype(y) == T
        @test all(isapprox.(Float32.(Array(y)), f.(Float32.(x)); rtol=T == Float32 ? 1f-4 : 4f-3, atol=1f-4))
    end
end

@testset "forward diff" begin
    for name in NNlib.ACTIVATIONS
        # println("Testing forward diff for activation: ", name)
        f = @eval $name
        @test gputest(DEVICE, x -> f.(x), Dual.(rand(Float32, 5), 1))
    end
end
