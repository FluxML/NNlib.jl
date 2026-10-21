function rotary_embedding_testsuite(Backend)
    device(x) = adapt(Backend(), x)
    T = nameof(Backend) === :MetalBackend ? Float32 : Float64   # Metal has no Float64
    test_gpu = Backend != CPU

    # `x` rotated with loops, at the `positions` of size `(seq_len, batch_size...)`
    function rotary_reference(x, c, s, positions; interleaved=false)
        y = similar(x)
        half = size(x, 1) ÷ 2
        for I in CartesianIndices(size(x)[2:end]), k in 1:half
            p = positions[I[2], Tuple(I)[3:end]...]
            i, j = interleaved ? (2k - 1, 2k) : (k, k + half)
            a, b = x[i, I], x[j, I]
            y[i, I] = a * c[k, p] - b * s[k, p]
            y[j, I] = b * c[k, p] + a * s[k, p]
        end
        return y
    end

    @testset "rotary_embedding_table" begin
        c, s = rotary_embedding_table(8, 20)
        @test c isa Matrix{Float32} && s isa Matrix{Float32}
        @test size(c) == size(s) == (4, 20)
        @test c[:, 1] == ones(4) && s[:, 1] == zeros(4)
        θ = 10_000.0 .^ (-(0:2:6) ./ 8)
        @test c ≈ cos.(θ .* (0:19)') && s ≈ sin.(θ .* (0:19)')
        c, s = rotary_embedding_table(8, 20, Float64; base=500)
        @test c isa Matrix{Float64}
        @test s[2, 3] ≈ sin(2 * 500^(-2 / 8))
        @test_throws ArgumentError rotary_embedding_table(7, 20)
    end

    @testset "apply_rotary_embedding" begin
        c, s = rotary_embedding_table(8, 30, T)
        dc, ds = device(c), device(s)
        for interleaved in (false, true), batch_size in ((), (3,), (2, 3))
            x = randn(T, 8, 2, 5, batch_size...)
            dx = device(x)
            nbatch = prod(batch_size)
            positions = repeat(1:5, 1, batch_size...)
            rope(x; kws...) = cpu(apply_rotary_embedding(dx, dc, ds; interleaved, kws...))
            @test rope(x) ≈ rotary_reference(x, c, s, positions; interleaved)
            @test rope(x; offset=7) ≈ rotary_reference(x, c, s, positions .+ 7; interleaved)
            offset = rand(0:20, nbatch)
            expected = rotary_reference(x, c, s, positions .+ reshape(offset, 1, batch_size...); interleaved)
            @test rope(x; offset) ≈ expected
            @test rope(x; offset=device(offset)) ≈ expected
            P = rand(1:30, 5, batch_size...)
            expected = rotary_reference(x, c, s, P; interleaved)
            @test rope(x; positions=P) ≈ expected
            @test rope(x; positions=device(P)) ≈ expected
            @test rope(x; positions=reshape(P, 5, :)) ≈ expected
            @test rope(x; positions=P[1:5]) ≈ rotary_reference(x, c, s, repeat(P[1:5], 1, batch_size...); interleaved)
        end

        # large enough for the kernel of the CPU to split the tokens between tasks
        c, s = rotary_embedding_table(64, 400, T)
        x = randn(T, 64, 4, 300, 2)
        P = rand(1:400, 300, 2)
        for interleaved in (false, true), (kws, positions) in (((;), repeat(1:300, 1, 2)),
                ((; offset=[3, 50]), (1:300) .+ [3 50]), ((; positions=P), P), ((; positions=P[:, 1]), P[:, [1, 1]]))
            y = cpu(apply_rotary_embedding(device(x), device(c), device(s); interleaved, kws...))
            @test y ≈ rotary_reference(x, c, s, positions; interleaved)
        end
        c, s = rotary_embedding_table(8, 30, T)
        dc, ds = device(c), device(s)

        # the attention logits depend on the difference of the positions only
        q, k = device(randn(T, 8, 1, 1)), device(randn(T, 8, 1, 1))
        logit(m, n) = sum(cpu(apply_rotary_embedding(q, dc, ds; offset=m) .*
                              apply_rotary_embedding(k, dc, ds; offset=n)))
        @test logit(9, 4) ≈ logit(5, 0) ≈ logit(25, 20)
        @test !(logit(9, 4) ≈ logit(9, 5))

        x = device(randn(T, 8, 2, 5, 3))
        @inferred apply_rotary_embedding(x, dc, ds)
        @inferred apply_rotary_embedding(x, dc, ds; interleaved=true, positions=device(rand(1:30, 5, 3)))
        @test_throws DimensionMismatch apply_rotary_embedding(device(randn(T, 6, 2, 5)), dc, ds)
        @test_throws DimensionMismatch apply_rotary_embedding(x, dc, ds[:, 1:20])
        @test_throws DimensionMismatch apply_rotary_embedding(x, dc, ds; offset=[1, 2])
        @test_throws DimensionMismatch apply_rotary_embedding(x, dc, ds; positions=ones(Int, 4, 3))
        @test_throws ArgumentError apply_rotary_embedding(x, dc, ds; offset=1, positions=ones(Int, 5))
        @test_throws ArgumentError apply_rotary_embedding(device(randn(T, 8, 5)), dc, ds)
        @test_throws BoundsError apply_rotary_embedding(x, dc, ds; offset=26)
        @test_throws ArgumentError apply_rotary_embedding(x, dc, ds; positions=fill(31, 5))
    end

    @testset "with attention" begin
        c, s = device.(rotary_embedding_table(16, 64, Float32))
        rope(x; kws...) = apply_rotary_embedding(x, c, s; kws...)

        # decoding with a KV cache gives the last outputs of the whole sequence
        q = device(randn(Float32, 16, 4, 13, 2))
        k = rope(device(randn(Float32, 16, 2, 13, 2)))
        v = device(randn(Float32, 16, 2, 13, 2))
        y = cpu(sdpa(rope(q), k, v; is_causal=true))
        @test cpu(sdpa(rope(q[:, :, 11:13, :]; offset=10), k, v; is_causal=true, q_offset=10)) ≈
              y[:, :, 11:13, :]
        @test cpu(sdpa(rope(q[:, :, 11:13, :]; offset=10), k, v; is_causal=:bottom_right)) ≈
              y[:, :, 11:13, :]

        # the positions of packed documents start over at each of them
        segments = [5, 12, 3]
        x = device(randn(Float32, 16, 4, 20))
        r = rope(x; positions=reduce(vcat, [1:n for n in segments]))
        y = cpu(sdpa(r, r, x; is_causal=true, q_segments=segments))
        for (n, last) in zip(segments, cumsum(segments))
            d = x[:, :, (last - n + 1):last]
            @test y[:, :, (last - n + 1):last] ≈ cpu(sdpa(rope(d), rope(d), d; is_causal=true))
        end
    end

    @testset "gradients" begin
        c, s = rotary_embedding_table(8, 30, T)
        dc, ds = device(c), device(s)
        compare = test_gpu ? [AutoZygote(), AutoEnzyme()] : [AutoZygote(), AutoEnzyme(), AutoMooncake()]
        x = randn(T, 8, 2, 5, 3)
        for interleaved in (false, true), kws in ((;), (; offset=4), (; offset=[1, 2, 3]),
                                                  (; positions=rand(1:30, 5, 3)))
            @test test_gradients(x -> apply_rotary_embedding(x, c, s; interleaved, kws...), x;
                                 f_gpu = x -> apply_rotary_embedding(x, dc, ds; interleaved, kws...),
                                 test_gpu, compare)
        end

        # Enzyme without runtime activity, for any of `x` and the tables constant
        if NNLIB_TEST_ENZYME && !test_gpu
            w = randn(T, size(x))
            loss(x, c, s) = sum(w .* apply_rotary_embedding(x, c, s; offset=2))
            gx = Zygote.gradient(x -> loss(x, c, s), x)[1]
            gc, gs = (FiniteDifferences.grad(FiniteDifferences.central_fdm(5, 1), f, a)[1]
                      for (f, a) in ((c -> loss(x, c, s), c), (s -> loss(x, c, s), s)))
            for (ax, at) in ((Duplicated, Const), (Duplicated, Duplicated), (Const, Duplicated))
                X = ax === Const ? Const(x) : Duplicated(x, zero(x))
                C, S = at === Const ? (Const(c), Const(s)) : (Duplicated(c, zero(c)), Duplicated(s, zero(s)))
                Enzyme.autodiff(Enzyme.Reverse, Const(loss), Active, X, C, S)
                ax === Const || @test X.dval ≈ gx
                at === Const || @test C.dval ≈ gc && S.dval ≈ gs
            end
        end

        # the pullback is the opposite rotation
        dx = device(x)
        y, back = rrule(NNlib.rotate_pairs, dx, dc, ds, 3, nothing, false)
        @test cpu(back(y)[2]) ≈ x
    end
end
