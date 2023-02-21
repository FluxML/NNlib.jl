@testset "Compare CPU & GPU" begin
    for (T, atol) in ((Float16, 1f-2), (Float32, 1f-5))
        x = randn(T, 16)
        gputest(x -> NNlib.relu.(x), x; atol)
        gputest(x -> NNlib.relu6.(x), x; atol)
        gputest(x -> NNlib.softplus.(x), x; atol)
        gputest(x -> tanh.(x), x; atol)
        gputest(x -> identity.(x), x; atol)
    end
end
