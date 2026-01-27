@testset "activation broadcast" begin
    broken_f = (:hardσ, :leakyrelu) 
    for name in NNlib.ACTIVATIONS
        # println("Testing forward diff for activation: ", name)
        f = @eval $name
        @test gputest(DEVICE, x -> f.(x), rand(5)) broken=name ∈ broken_f
    end
end

@testset "forward diff" begin
    broken_f = (:hardσ, :leakyrelu) 
    for name in NNlib.ACTIVATIONS
        # println("Testing forward diff for activation: ", name)
        f = @eval $name
        @test gputest(DEVICE, x -> f.(x), Dual.(rand(Float32, 5), 1)) broken=name ∈ broken_f
    end
end
