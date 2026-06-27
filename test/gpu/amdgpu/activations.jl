@testset "Compare CPU & GPU" begin
    for (T, atol) in ((Float16, 1.0f-2), (Float32, 1.0f-5))
        @testset "ndims: $(ndims(x))" for x in (randn(T, 16), randn(T, ntuple(_ -> 2, 5)...), randn(T, ntuple(_ -> 2, 6)...))
            gputest(x -> NNlib.relu.(x), x; atol)
            gputest(x -> NNlib.relu6.(x), x; atol)
            gputest(x -> NNlib.softplus.(x), x; atol)
            gputest(x -> tanh.(x), x; atol)
            gputest(x -> identity.(x), x; atol)
        end
    end
end
