@testset "activation broadcast" begin
    for f in NNlib.ACTIVATIONS
        if f ∉ [:rrelu]
            @eval gputest(x -> $f.(x), rand(Float32, 5))
        end
    end
end
