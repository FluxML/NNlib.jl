# The materialized implementation: an `fdrop` other than `identity` keeps a call on it.
materialized_attention(args...; kws...) = scaled_dot_product_attention(args...; fdrop = x -> x, kws...)

# The gradients of `sum(w .* f(xs...; kws...))` with respect to the `xs`.
attention_gradients(f, w, xs...; kws...) = gradient((xs...) -> sum(w .* f(xs...; kws...)), xs...)

@testset "is_fused_attention" begin
    q = rand(Float32, 8, 2, 3, 2)
    @test NNlib.is_fused_attention(q, q, q)
    @test NNlib.is_fused_attention(q, q, q; is_causal=true, kv_lengths=[3, 2])
    @test NNlib.is_fused_attention(q, q, q, rand(Float32, 3, 3); mask=trues(3, 3))
    @test NNlib.is_fused_attention(Float64.(q), Float64.(q), Float64.(q), rand(3, 3))
    # dropout acts on the weights, which are not formed
    @test !NNlib.is_fused_attention(q, q, q; fdrop = x -> x)
    # mixed element types, a bias that promotes the output, arrays that are not dense
    @test !NNlib.is_fused_attention(q, q, Float64.(q))
    @test !NNlib.is_fused_attention(q, q, q, rand(3, 3))
    @test !NNlib.is_fused_attention(view(q, :, :, :, 1:2), q, q)
    @test !NNlib.is_fused_attention(q, q, q; mask=view(trues(3, 3), :, :))
    @test_throws ArgumentError NNlib.is_fused_attention(q, q[:, :, :, 1], q)
end

@testset "fused implementation" begin
    @testset "$T" for T in (Float32, Float64, Float16)
        # the half-precision reference accumulates in `Float16`
        rtol = T == Float16 ? 2f-2 : sqrt(eps(T))
        # more keys than a tile holds, an odd number of queries
        head_dim, v_head_dim, nheads, lenq, lenkv, bs = 10, 7, 4, 37, 301, 2
        q = randn(T, head_dim, nheads, lenq, bs) ./ 2
        k = randn(T, head_dim, nheads, lenkv, bs) ./ 2
        v = randn(T, v_head_dim, nheads, lenkv, bs)
        @test NNlib.is_fused_attention(q, k, v)

        y = @inferred scaled_dot_product_attention(q, k, v)
        @test y isa Array{T,4}
        @test size(y) == (v_head_dim, nheads, lenq, bs)
        @test y ≈ materialized_attention(q, k, v) rtol=rtol
        @test y ≈ first(scaled_dot_product_attention_with_weights(q, k, v)) rtol=rtol
        @test scaled_dot_product_attention(q, k, v; scale=0.3) ≈ materialized_attention(q, k, v; scale=0.3) rtol=rtol
        @test scaled_dot_product_attention(q, k, v; is_causal=true) ≈
              materialized_attention(q, k, v; is_causal=true) rtol=rtol
        @test NNlib.@disallow_spawns(scaled_dot_product_attention(q, k, v)) ≈ y

        lengths = (; q_lengths=[37, 12], kv_lengths=[150, 301])
        yl = scaled_dot_product_attention(q, k, v; lengths...)
        @test yl ≈ materialized_attention(q, k, v; lengths...) rtol=rtol
        @test all(iszero, yl[:, :, 13:end, 2])

        mask = rand(Bool, lenkv, lenq, 1, bs)
        mask[1, :, :, :] .= true
        bias = randn(T, lenkv, lenq)
        @test scaled_dot_product_attention(q, k, v; mask) ≈ materialized_attention(q, k, v; mask) rtol=rtol
        @test scaled_dot_product_attention(q, k, v; mask=BitArray(mask)) ≈
              materialized_attention(q, k, v; mask) rtol=rtol
        @test scaled_dot_product_attention(q, k, v, bias) ≈ materialized_attention(q, k, v, bias) rtol=rtol
        @test scaled_dot_product_attention(q, k, v, bias; mask, is_causal=true, lengths...) ≈
              materialized_attention(q, k, v, bias; mask, is_causal=true, lengths...) rtol=rtol

        # a causal mask shifted by an offset, which ends inside the tiles of keys
        for q_offset in (lenkv - lenq, [100, 3])
            @test scaled_dot_product_attention(q, k, v; is_causal=true, q_offset) ≈
                  materialized_attention(q, k, v; is_causal=true, q_offset) rtol=rtol
            @test scaled_dot_product_attention(q, k, v, bias; mask, is_causal=true, q_offset, lengths...) ≈
                  materialized_attention(q, k, v, bias; mask, is_causal=true, q_offset, lengths...) rtol=rtol
        end
        @test scaled_dot_product_attention(q, k, v; is_causal=:bottom_right, lengths...) ≈
              materialized_attention(q, k, v; is_causal=:bottom_right, lengths...) rtol=rtol

        # sliding windows, which start and end inside the tiles of keys
        for window in (1, 50, 200), is_causal in (true, false)
            kws = (; is_causal, window, q_offset=lenkv - lenq)
            @test scaled_dot_product_attention(q, k, v; kws...) ≈ materialized_attention(q, k, v; kws...) rtol=rtol
        end
        @test scaled_dot_product_attention(q, k, v, bias; is_causal=:bottom_right, window=40, lengths...) ≈
              materialized_attention(q, k, v, bias; is_causal=:bottom_right, window=40, lengths...) rtol=rtol

        # grouped-query attention
        kg, vg = k[:, 1:2, :, :], v[:, 1:2, :, :]
        @test scaled_dot_product_attention(q, kg, vg; is_causal=true) ≈
              materialized_attention(q, kg, vg; is_causal=true) rtol=rtol
        @test scaled_dot_product_attention(q, kg, vg; is_causal=true, q_offset=[100, 3]) ≈
              materialized_attention(q, kg, vg; is_causal=true, q_offset=[100, 3]) rtol=rtol
        @test scaled_dot_product_attention(q, kg, vg; window=50, q_offset=[100, 3]) ≈
              materialized_attention(q, kg, vg; window=50, q_offset=[100, 3]) rtol=rtol
    end

    # self-attention over several tiles of keys, where the causal mask ends inside them
    x = randn(Float32, 6, 2, 300, 1)
    @test scaled_dot_product_attention(x, x, x; is_causal=true) ≈
          materialized_attention(x, x, x; is_causal=true)

    # a query attending to no key gives NaN, as in the materialized implementation
    q = rand(Float32, 4, 2, 3, 1)
    k = rand(Float32, 4, 2, 5, 1)
    mask = trues(5, 3)
    mask[:, 2] .= false
    y = scaled_dot_product_attention(q, k, k; mask)
    @test all(isnan, y[:, :, 2, :])
    @test y[:, :, [1, 3], :] ≈ materialized_attention(q, k, k; mask)[:, :, [1, 3], :]

    # ... and so does a sequence without a valid key
    k2 = rand(Float32, 4, 2, 5, 2)
    q2 = rand(Float32, 4, 2, 3, 2)
    y = scaled_dot_product_attention(q2, k2, k2; kv_lengths=[0, 4])
    @test all(isnan, y[:, :, :, 1])
    @test y[:, :, :, 2] ≈ materialized_attention(q2, k2, k2; kv_lengths=[5, 4])[:, :, :, 2]

    # ... and the queries a negative offset leaves without a key
    y = scaled_dot_product_attention(q2, k2, k2; is_causal=true, q_offset=[-2, -5])
    ym = materialized_attention(q2, k2, k2; is_causal=true, q_offset=[-2, -5])
    @test all(isnan, y[:, :, 1:2, 1]) && all(isnan, y[:, :, :, 2])
    @test y[:, :, 3, 1] ≈ ym[:, :, 3, 1]

    # the tilings of the GPU kernel are plain arithmetic: each has to fit the shared
    # memory it is given, whatever the head dimension, the problem size and the eltype
    @testset "kernel tilings" begin
        for budget in (31 * 1024, 47 * 1024), bytes in (4, 8), nqueries in (10, 10^6),
                v_head_dim in (1, 5, 64, 65, 128, 200, 256, 257, 512, 600, 1024, 2048)
            c = NNlib.attention_tiling(v_head_dim, nqueries, budget ÷ bytes)
            @test c !== nothing
            @test NNlib.attention_shmem(c) * bytes <= budget
            @test c.T * c.TM * c.TN == c.BQ * c.BK      # the logits of a tile
            @test c.NR * c.NC == c.T == c.ND * c.NQ
            @test c.TD * c.ND >= v_head_dim             # the outputs of a tile of queries
            @test c.TD * c.TQ <= 64 && c.TM * c.TN <= 64
            @test c.T % c.KD == 0 && c.BK % c.KV == 0
        end
        @test NNlib.attention_tiling(4096, 10^6, 47 * 1024 ÷ 4) === nothing
        @test NNlib.attention_tiling(64, 10^6, 100) === nothing
    end
end

@testset "fused gradient" begin
    # Under Zygote the fused implementation takes its rule, and the materialized one is
    # differentiated through.
    function same_gradients(w, xs...; rtol, kws...)
        g = attention_gradients(scaled_dot_product_attention, w, xs...; kws...)
        gm = attention_gradients(materialized_attention, w, xs...; kws...)
        return all(isapprox.(g, gm; rtol))
    end

    @testset "$T" for T in (Float32, Float64)
        rtol = T == Float32 ? 1f-3 : 1e-8
        # more keys than a tile holds, a number of queries that is not a multiple of four
        head_dim, v_head_dim, nheads, lenq, lenkv, bs = 10, 7, 4, 37, 301, 2
        q = randn(T, head_dim, nheads, lenq, bs) ./ 2
        k = randn(T, head_dim, nheads, lenkv, bs) ./ 2
        v = randn(T, v_head_dim, nheads, lenkv, bs)
        w = randn(T, v_head_dim, nheads, lenq, bs)
        mask = rand(Bool, lenkv, lenq, 1, bs)
        mask[1, :, :, :] .= true
        lengths = (; q_lengths=[37, 12], kv_lengths=[150, 301])

        @test same_gradients(w, q, k, v; rtol)
        @test same_gradients(w, q, k, v; rtol, mask=BitArray(mask), is_causal=true, lengths...)
        # the code is generic in the element type, and each case compiles anew: the others in Float32 only
        if T == Float32
            @test same_gradients(w, q, k, v; rtol, is_causal=true)
            @test same_gradients(w, q, k, v; rtol, scale=0.3)
            @test same_gradients(w, q, k, v; rtol, lengths...)
            @test same_gradients(w, q, k, v; rtol, mask)
            @test same_gradients(w, q, k[:, 1:2, :, :], v[:, 1:2, :, :]; rtol, is_causal=true)
            @test NNlib.@disallow_spawns same_gradients(w, q, k, v; rtol, is_causal=true)
            @test same_gradients(w, q, k, v; rtol, is_causal=true, q_offset=lenkv - lenq)
            @test same_gradients(w, q, k, v; rtol, mask, is_causal=true, q_offset=[100, 3], lengths...)
            @test same_gradients(w, q, k[:, 1:2, :, :], v[:, 1:2, :, :]; rtol, is_causal=true, q_offset=[100, 3])
            @test same_gradients(w, q, k, v; rtol, is_causal=:bottom_right, lengths...)
            @test same_gradients(w, q, k, v; rtol, is_causal=true, window=50, q_offset=lenkv - lenq)
            @test same_gradients(w, q, k, v; rtol, window=50, q_offset=[100, 3], lengths...)
            @test same_gradients(w, q, k[:, 1:2, :, :], v[:, 1:2, :, :]; rtol, is_causal=:bottom_right, window=30,
                                 lengths...)

            # with respect to the scale, which can be zero
            for scale in (T(0.3), zero(T))
                ds = gradient(s -> sum(w .* scaled_dot_product_attention(q, k, v; scale=s)), scale)[1]
                @test ds ≈ gradient(s -> sum(w .* materialized_attention(q, k, v; scale=s)), scale)[1] rtol=rtol
            end
        end

        dq, dk, dv = attention_gradients(scaled_dot_product_attention, w, q, k, v; lengths...)
        @test dq isa Array{T,4} && dk isa Array{T,4} && dv isa Array{T,4}
        # no gradient reaches the padding
        @test all(iszero, dq[:, :, 13:end, 2])
        @test all(iszero, dk[:, :, 151:end, 1]) && all(iszero, dv[:, :, 151:end, 1])
    end

    # Every pair of a range of queries and a range of keys is taken once, and the pairs of
    # a round share no range.
    for (nrq, nrk) in ((1, 1), (3, 3), (2, 5), (5, 2))
        pairs = [NNlib.attention_pair(x, round, nrq, nrk)
                 for round in 0:(max(nrq, nrk) - 1), x in 0:(min(nrq, nrk) - 1)]
        @test sort(vec(pairs)) == sort(vec([(t, u) for t in 0:(nrq - 1), u in 0:(nrk - 1)]))
        @test all(row -> allunique(first.(row)) && allunique(last.(row)), eachrow(pairs))
    end

    # Several ranges of queries and of keys, when there are threads to pair them in rounds.
    @testset "$lenq queries, $lenkv keys" for (lenq, lenkv) in ((700, 700), (300, 1100), (1100, 300), (128, 1100))
        q = randn(Float32, 8, 1, lenq, 1) ./ 2
        k = randn(Float32, 8, 1, lenkv, 1) ./ 2
        v = randn(Float32, 5, 1, lenkv, 1)
        w = randn(Float32, 5, 1, lenq, 1)
        @test same_gradients(w, q, k, v; rtol=1f-3)
        @test same_gradients(w, q, k, v; rtol=1f-3, is_causal=true, q_lengths=[lenq - 5], kv_lengths=[lenkv - 9])
        @test same_gradients(w, q, k, v; rtol=1f-3, is_causal=true, q_offset=max(lenkv - lenq, 0) + 70)
        # windows that leave no query without a key
        if lenq <= lenkv
            @test same_gradients(w, q, k, v; rtol=1f-3, is_causal=:bottom_right, window=150)
            @test same_gradients(w, q, k, v; rtol=1f-3, window=150, q_offset=lenkv - lenq)
        end
    end

    # the same array as query, key and value
    x = randn(Float32, 6, 2, 300, 1)
    w = randn(Float32, 6, 2, 300, 1)
    @test gradient(x -> sum(w .* scaled_dot_product_attention(x, x, x; is_causal=true)), x)[1] ≈
          gradient(x -> sum(w .* materialized_attention(x, x, x; is_causal=true)), x)[1] rtol=1f-3

    q = randn(Float32, 8, 2, 20, 2) ./ 2
    k = randn(Float32, 8, 2, 30, 2) ./ 2
    v = randn(Float32, 6, 2, 30, 2)
    w = randn(Float32, 6, 2, 20, 2)

    # half precision is accumulated in single precision
    half = map(x -> Float16.(x), (w, q, k, v))
    g16 = attention_gradients(scaled_dot_product_attention, half...; is_causal=true)
    g32 = attention_gradients(scaled_dot_product_attention, w, q, k, v; is_causal=true)
    @test all(x -> x isa Array{Float16,4}, g16)
    @test all(isapprox.(g16, g32; rtol=2f-2))

    # a cotangent that is not an `Array`
    @test gradient(q -> sum(scaled_dot_product_attention(q, k, v)), q)[1] ≈
          gradient(q -> sum(materialized_attention(q, k, v)), q)[1] rtol=1f-3

    # A query attending to no key has a NaN output, and takes no part in the gradients:
    # those of the other queries are what they are without it.
    mask = trues(30, 20)
    mask[:, 2] .= false
    dq, dk, dv = attention_gradients(scaled_dot_product_attention, w, q, k, v; mask)
    rest = setdiff(1:20, 2)
    dqr, dkr, dvr = attention_gradients(scaled_dot_product_attention, w[:, :, rest, :], q[:, :, rest, :], k, v)
    @test all(iszero, dq[:, :, 2, :])
    @test dq[:, :, rest, :] ≈ dqr
    @test dk ≈ dkr && dv ≈ dvr

    # ... so do those a negative offset leaves without a key
    dq, dk, dv = attention_gradients(scaled_dot_product_attention, w, q, k, v; is_causal=true, q_offset=-2)
    dqr, dkr, dvr = attention_gradients(scaled_dot_product_attention, w[:, :, 3:end, :], q[:, :, 3:end, :], k, v;
                                        is_causal=true, q_offset=0)
    @test all(iszero, dq[:, :, 1:2, :])
    @test dq[:, :, 3:end, :] ≈ dqr
    @test dk ≈ dkr && dv ≈ dvr

    # With a bias the materialized implementation is the one differentiated: the fused
    # one has no gradient with respect to it.
    bias = randn(Float32, 30, 20)
    @test NNlib.is_fused_attention(q, k, v, bias)
    @test gradient(b -> sum(w .* scaled_dot_product_attention(q, k, v, b)), bias)[1] ≈
          gradient(b -> sum(w .* materialized_attention(q, k, v, b)), bias)[1]
    @test_throws ArgumentError gradient(b -> sum(NNlib.fused_attention(q, k, v, b, nothing, nothing, false, 0, nothing,
                                                                       nothing, nothing, nothing)), bias)
end

if NNLIB_TEST_ENZYME

# Enzyme has rules of its own for the fused implementation: reverse mode calls its
# gradient, and forward mode differentiates the materialized implementation.
@testset "fused gradient: EnzymeRules" begin
    q = randn(3, 2, 4, 2) ./ 2
    k = randn(3, 2, 5, 2) ./ 2
    v = randn(2, 2, 5, 2)
    rest(Tscale) = ((nothing, EnzymeCore.Const), (nothing, EnzymeCore.Const), (0.3, Tscale),
                    (true, EnzymeCore.Const), ([1, 0], EnzymeCore.Const), (nothing, EnzymeCore.Const),
                    ([4, 2], EnzymeCore.Const), (nothing, EnzymeCore.Const), (3, EnzymeCore.Const))
    D, BD, C = EnzymeCore.Duplicated, EnzymeCore.BatchDuplicated, EnzymeCore.Const

    for (Tret, Tq, Tk, Tv, Tscale) in ((D, D, D, D, EnzymeCore.Active), (D, C, D, D, C), (D, D, C, C, EnzymeCore.Active),
                                       (BD, BD, BD, BD, C), (C, D, D, D, C))
        EnzymeTestUtils.test_reverse(NNlib.fused_attention, Tret, (q, Tq), (k, Tk), (v, Tv), rest(Tscale)...;
                                     atol=1e-6, rtol=1e-6)
    end
    for (Tret, Tq, Tk, Tv) in ((D, D, D, D), (D, D, C, D), (BD, BD, BD, BD), (C, D, D, D))
        EnzymeTestUtils.test_forward(NNlib.fused_attention, Tret, (q, Tq), (k, Tk), (v, Tv), rest(C)...;
                                     atol=1e-6, rtol=1e-6)
    end

    # through the public function, the three arguments being the same array
    x = randn(3, 2, 4, 2)
    w = randn(3, 2, 4, 2)
    dx = zero(x)
    Enzyme.autodiff(Enzyme.Reverse, Const(x -> sum(w .* scaled_dot_product_attention(x, x, x; is_causal=true))),
                    Active, Duplicated(x, dx))
    @test dx ≈ gradient(x -> sum(w .* materialized_attention(x, x, x; is_causal=true)), x)[1]

    # Constant arguments next to active ones, as a cache of keys and values, in the fused
    # implementation, in the materialized one (with a bias) and with several batch dimensions.
    q, k, v = randn(3, 4, 2, 2), randn(3, 2, 4, 2), randn(2, 2, 4, 2)
    xs5 = (randn(3, 2, 2, 2, 2), randn(3, 2, 4, 2, 2), randn(2, 2, 4, 2, 2))
    xs3 = (randn(3, 4, 5), randn(3, 2, 6), randn(2, 2, 6))
    @testset "$name, active $act" for (name, xs, bias, kws) in (
                ("fused", (q, k, v), nothing, (; is_causal=:bottom_right, kv_lengths=[3, 4])),
                ("materialized", (q, k, v), randn(4, 2), (; is_causal=true)),
                ("5 dimensions", xs5, nothing, (;)),
                ("packed", xs3, nothing, (; is_causal=true, q_segments=[2, 3], kv_segments=[4, 2]))),
            act in ((true, false, false), (false, true, true))
        shadows = map(zero, xs)
        Enzyme.autodiff(Enzyme.Reverse, Const((xs...) -> sum(scaled_dot_product_attention(xs..., bias; kws...))),
                        Active, map((x, s, a) -> a ? Duplicated(x, s) : Const(x), xs, shadows, act)...)
        grads = gradient((xs...) -> sum(materialized_attention(xs..., bias; kws...)), xs...)
        @test all(i -> !act[i] || shadows[i] ≈ grads[i], 1:3)
    end
end

end # NNLIB_TEST_ENZYME
