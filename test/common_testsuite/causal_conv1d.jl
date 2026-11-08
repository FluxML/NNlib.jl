function causal_conv1d_testsuite(Backend)
    device(x) = adapt(Backend(), x)
    T = nameof(Backend) === :MetalBackend ? Float32 : Float64   # Metal has no Float64
    test_gpu = Backend != CPU
    rng = StableRNG(17)

    # the convolution with loops, and the last `K - 1` inputs
    function conv_reference(x, w, bias, state)
        K, C = size(w)
        n, b = size(x, 2), size(x, 3)
        xs = cat(something(state, zeros(eltype(x), C, K - 1, b)), x; dims=2)
        y = [sum(w[j, c] * xs[c, t + j - 1, i] for j in 1:K) + (bias === nothing ? 0 : bias[c])
             for c in 1:C, t in 1:n, i in 1:b]
        return y, xs[:, (end - K + 2):end, :]
    end
    conv(x, w; state=nothing, kws...) =
        cpu.(NNlib.causal_conv1d(device(x), device(w); state=device(state), kws...))

    @testset "forward, K = $K" for K in (1, 2, 4)
        x, w = randn(rng, T, 6, 9, 2), randn(rng, T, K, 6)
        bias, state = randn(rng, T, 6), randn(rng, T, 6, K - 1, 2)
        for bias in (nothing, bias), state in (nothing, state)
            y, new_state = conv(x, w; bias=device(bias), state)
            y_ref, state_ref = conv_reference(x, w, bias, state)
            @test y ≈ y_ref
            @test new_state ≈ state_ref
        end

        # in pieces, or token by token, from the state of the previous piece
        y, new_state = conv(x, w; state)
        y1, state1 = conv(x[:, 1:4, :], w; state)
        y2, state2 = conv(x[:, 5:9, :], w; state=state1)
        @test cat(y1, y2; dims=2) ≈ y
        @test state2 ≈ new_state
        s = state
        for t in 1:9
            yt, s = conv(x[:, t:t, :], w; state=s)
            @test yt ≈ y[:, t:t, :]
        end
        @test s ≈ new_state

        # in a batch padded on the right, the state at the end of each sequence
        for lengths in ([4, 9], device([4, 9]), [0, 2])
            y, new_state = conv(x, w; state, lengths)
            for (i, n) in enumerate(cpu(lengths))
                y_ref, state_ref = conv_reference(x[:, 1:n, i:i], w, nothing, state[:, :, i:i])
                @test y[:, 1:n, i:i] ≈ y_ref
                @test new_state[:, :, i:i] ≈ state_ref
            end
        end
        @test conv(x, w; state, lengths=9)[2] ≈ conv(x, w; state)[2]

        # a single token, past the end of the first sequence
        y1, state1 = conv(x[:, 1:1, :], w; bias=device(bias), state, lengths=[0, 1])
        y_ref, state_ref = conv_reference(x[:, 1:1, :], w, bias, state)
        @test y1 ≈ y_ref
        @test state1 ≈ cat(state[:, :, 1:1], state_ref[:, :, 2:2]; dims=3)
        @test conv(x[:, 1:1, :], w)[1] ≈ conv_reference(x[:, 1:1, :], w, nothing, nothing)[1]
    end

    @testset "types and errors" begin
        x, w = device(randn(rng, Float32, 6, 9, 2)), device(randn(rng, Float32, 4, 6))
        y, s = @inferred NNlib.causal_conv1d(x, w; bias=device(randn(rng, Float32, 6)))
        @test eltype(y) == eltype(s) == Float32 && size(y) == (6, 9, 2) && size(s) == (6, 3, 2)
        y16, _ = NNlib.causal_conv1d(Float16.(x), Float16.(w))
        @test eltype(y16) == Float16
        @test cpu(y16) ≈ cpu(NNlib.causal_conv1d(x, w)[1]) rtol=1e-2
        @test_throws DimensionMismatch NNlib.causal_conv1d(x, device(randn(rng, Float32, 4, 5)))
        @test_throws DimensionMismatch NNlib.causal_conv1d(x, w; bias=device(randn(rng, Float32, 5)))
        @test_throws DimensionMismatch NNlib.causal_conv1d(x, w; state=device(randn(rng, Float32, 6, 4, 2)))
        @test_throws ArgumentError NNlib.causal_conv1d(x, w; lengths=[1, 2, 3])
    end

    @testset "gradients" begin
        x, w = randn(rng, T, 5, 7, 2), randn(rng, T, 4, 5)
        bias, state = randn(rng, T, 5), randn(rng, T, 5, 3, 2)
        compare = test_gpu ? [AutoZygote(), AutoEnzyme()] : [AutoZygote(), AutoEnzyme(), AutoMooncake()]
        conv_and_state(x, w, bias, state; lengths=nothing) =
            hcat(NNlib.causal_conv1d(x, w; bias, state, lengths)...)
        @test test_gradients(conv_and_state, x, w, bias, state; test_gpu, compare)
        @test test_gradients((x, w, bias, state) -> conv_and_state(x, w, bias, state; lengths=[3, 7]),
                             x, w, bias, state; test_gpu, compare=AutoZygote())
    end
end
