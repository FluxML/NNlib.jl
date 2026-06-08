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

@testset "NaN propagation" begin
    # MIOpen's activation path used to swallow NaNs (returning e.g. 0 for relu),
    # diverging from the CPU. Make sure the native broadcast propagates them (#509).
    for f in (NNlib.relu, NNlib.relu6, NNlib.softplus, tanh, NNlib.σ)
        @test all(isnan, f.(ROCArray([NaN32, NaN32])))
    end
end
