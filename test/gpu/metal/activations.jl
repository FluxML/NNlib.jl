@testset "activation broadcast" begin
    for name in NNlib.ACTIVATIONS
        # println("Testing forward diff for activation: ", name)
        f = @eval $name
        @test test_gradients(x -> f.(x), rand(5); test_gpu=true)
    end
end

@testset "forward diff" begin
    for name in NNlib.ACTIVATIONS
        # println("Testing forward diff for activation: ", name)
        f = @eval $name
        @test gputest(DEVICE, x -> f.(x), Dual.(rand(Float32, 5), 1))
    end
end
