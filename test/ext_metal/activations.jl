@testset "activation broadcast" begin
    for name in NNlib.ACTIVATIONS
        # println("Testing forward diff for activation: ", name)
        f = @eval $name
        @test gputest(DEVICE, x -> f.(x), rand(5))
    end
end

@testset "forward diff" begin
    for name in NNlib.ACTIVATIONS
        # println("Testing forward diff for activation: ", name)
        f = @eval $name
        @test gputest(DEVICE, x -> f.(x), Dual.(rand(Float32, 5), 1))
    end
end
