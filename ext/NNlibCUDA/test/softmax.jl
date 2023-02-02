@testset "softmax" begin
    for (sz, dims) in [((5,), :), ((5,), 1), ((5,5), :), ((5,5), 1), ((5,5), 2), ((5,5,5,5), (2,3)), ((5,5,5,5), (2,4))]
        x = randn(Float64, sz)
        dy = randn(Float64, sz)

        y = softmax(x, dims=dims)
        gputest(softmax, x, dims=dims)
        gputest(NNlib.∇softmax_data, dy, y; dims=dims)

        y2 = logsoftmax(x, dims=dims)
        gputest(logsoftmax, x, dims=dims)
        gputest(NNlib.∇logsoftmax_data, dy, y2; dims=dims)

        # From NNlib 0.8.3, ∇softmax! is not used in the gradient.
        # But NNlibCUDA still knows how to call cuDNN routines, let's test they agree:
        @test NNlib.∇softmax_data(dy, y; dims=dims) ≈ collect(∇softmax!(similar(cu(x)), cu(dy), cu(x), cu(y); dims=dims)) atol=1e-4
        @test NNlib.∇logsoftmax_data(dy, y2; dims=dims) ≈ collect(∇logsoftmax!(similar(cu(x)), cu(dy), cu(x), cu(y2); dims=dims)) atol=1e-4
        # (Note that ∇softmax! does not depend on x, it's just there to disambiguate from an even older signature.)
    end
end
