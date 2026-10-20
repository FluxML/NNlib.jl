# cuDNN's fused kernel handles half-precision inputs whose head dimension is a multiple
# of 8, with no bias, dropout or arbitrary mask. NNlib's own fused kernel takes the rest,
# in every precision, except under dropout: the materialized implementation forms the
# weights dropout acts on.
const sdpa_fused = NNlib.is_fused_attention

# The materialized implementation: an `fdrop` other than `identity` keeps a call on it.
materialized_attention(args...; kws...) = scaled_dot_product_attention(args...; fdrop = x -> x, kws...)

half(T, x) = T.(CuArray(x))    # round on the device (JuliaMath/BFloat16s.jl#107)

@testset "scaled dot-product attention ($T)" for T in (Float16, BFloat16)
    T == BFloat16 && CUDA.capability(CUDA.device()) < v"8.0" && continue
    q = rand(Float32, 32, 2, 16, 3) ./ 4
    k = rand(Float32, 32, 2, 12, 3) ./ 4
    v = rand(Float32, 32, 2, 12, 3) ./ 4
    qd, kd, vd = half(T, q), half(T, k), half(T, v)

    @test sdpa_fused(qd, kd, vd)
    y = scaled_dot_product_attention(qd, kd, vd)
    @test y isa CuArray{T,4}
    @test Float32.(Array(y)) ≈ scaled_dot_product_attention(q, k, v) rtol=2f-2
    @test Float32.(Array(scaled_dot_product_attention(qd, kd, vd; scale=0.25f0))) ≈
          scaled_dot_product_attention(q, k, v; scale=0.25f0) rtol=2f-2

    @testset "causal" begin
        @test sdpa_fused(qd, qd, qd; is_causal=true)
        @test Float32.(Array(scaled_dot_product_attention(qd, qd, qd; is_causal=true))) ≈
              scaled_dot_product_attention(q, q, q; is_causal=true) rtol=2f-2
        @test Float32.(Array(scaled_dot_product_attention(qd, kd, vd; is_causal=true))) ≈
              scaled_dot_product_attention(q, k, v; is_causal=true) rtol=2f-2

        # cuDNN aligns the causal mask with the first key: an offset takes NNlib's kernel
        ext = Base.get_extension(NNlib, :NNlibCUDACUDNNExt)
        cudnn(q_offset) = ext.cudnn_sdpa_output(qd, kd, vd, nothing, identity, nothing, true,
                                                NNlib.attention_offsets(qd, q_offset), nothing, nothing)
        @test cudnn(0) !== nothing
        @test cudnn(2) === nothing && cudnn([2, 0, 5]) === nothing
        @test sdpa_fused(qd, kd, vd; is_causal=true, q_offset=[2, 0, 5])
        @test Float32.(Array(scaled_dot_product_attention(qd, kd, vd; is_causal=true, q_offset=[2, 0, 5]))) ≈
              scaled_dot_product_attention(q, k, v; is_causal=true, q_offset=[2, 0, 5]) rtol=2f-2
    end

    @testset "grouped-query attention" begin
        qg = rand(Float32, 32, 4, 16, 3) ./ 4
        qgd = half(T, qg)
        @test sdpa_fused(qgd, kd, vd)
        @test Float32.(Array(scaled_dot_product_attention(qgd, kd, vd))) ≈
              scaled_dot_product_attention(qg, k, v) rtol=2f-2
    end

    @testset "with weights" begin
        # the weights are never computed by the fused kernel
        y, α = scaled_dot_product_attention_with_weights(qd, kd, vd)
        yref, αref = scaled_dot_product_attention_with_weights(q, k, v)
        @test y isa CuArray{T,4}
        @test α isa CuArray{T,4}
        @test Float32.(Array(y)) ≈ yref rtol=2f-2
        @test Float32.(Array(α)) ≈ αref rtol=2f-2 atol=1f-3
    end

    @testset "sequence lengths" begin
        q_lengths, kv_lengths = [16, 9, 1], [12, 5, 7]
        @testset "$(keys(kws))" for kws in ((; kv_lengths), (; q_lengths),
                (; q_lengths, kv_lengths), (; q_lengths, kv_lengths, is_causal=true))
            @test sdpa_fused(qd, kd, vd; kws...)
            @test Float32.(Array(scaled_dot_product_attention(qd, kd, vd; kws...))) ≈
                  scaled_dot_product_attention(q, k, v; kws...) rtol=2f-2
        end
        # the queries past `q_lengths` are zero
        y = Array(scaled_dot_product_attention(qd, kd, vd; q_lengths, kv_lengths))
        @test all(iszero, y[:, :, 10:end, 2])
        # lengths on the device, and for all the sequences
        @test Array(scaled_dot_product_attention(qd, kd, vd; kv_lengths=CuArray(kv_lengths))) ==
              Array(scaled_dot_product_attention(qd, kd, vd; kv_lengths))
        @test Array(scaled_dot_product_attention(qd, kd, vd; kv_lengths=5)) ==
              Array(scaled_dot_product_attention(qd, kd, vd; kv_lengths=[5, 5, 5]))
        # together with a mask they take NNlib's kernel
        mask = CUDA.ones(Bool, 12, 16)
        @test sdpa_fused(qd, kd, vd; mask, q_lengths, kv_lengths)
        @test Float32.(Array(scaled_dot_product_attention(qd, kd, vd; mask, q_lengths, kv_lengths))) ≈
              scaled_dot_product_attention(q, k, v; q_lengths, kv_lengths) rtol=2f-2
    end

    @testset "batch dimensions $bs" for bs in ((), (2, 3))
        qb = rand(Float32, 32, 2, 16, bs...) ./ 4
        qbd = half(T, qb)
        y = scaled_dot_product_attention(qbd, qbd, qbd)
        @test size(y) == size(qb)
        @test Float32.(Array(y)) ≈ scaled_dot_product_attention(qb, qb, qb) rtol=2f-2
        lengths = rand(1:16, max(1, prod(bs)))
        bs == () || (lengths = reshape(lengths, bs))
        @test sdpa_fused(qbd, qbd, qbd; kv_lengths=lengths)
        @test Float32.(Array(scaled_dot_product_attention(qbd, qbd, qbd; kv_lengths=lengths))) ≈
              scaled_dot_product_attention(qb, qb, qb; kv_lengths=lengths) rtol=2f-2
    end

    # what cuDNN has no engine for
    @testset "NNlib's kernel" begin
        mask = make_causal_mask(qd; dims=3)
        @test sdpa_fused(qd, qd, qd; mask)
        @test Float32.(Array(scaled_dot_product_attention(qd, qd, qd; mask))) ≈
              scaled_dot_product_attention(q, q, q; mask=Array(mask)) rtol=2f-2

        bias = randn(Float32, 12, 16)
        biasd = half(T, bias)
        @test sdpa_fused(qd, kd, vd, biasd)
        @test Float32.(Array(scaled_dot_product_attention(qd, kd, vd, biasd))) ≈
              scaled_dot_product_attention(q, k, v, bias) rtol=2f-2

        @test !sdpa_fused(qd, kd, vd; fdrop = x -> x)
        @test Float32.(Array(scaled_dot_product_attention(qd, kd, vd; fdrop = x -> x))) ≈
              scaled_dot_product_attention(q, k, v) rtol=2f-2

        # a value head dimension different from the query's
        v8 = rand(Float32, 8, 2, 12, 3) ./ 4
        v8d = half(T, v8)
        @test sdpa_fused(qd, kd, v8d)
        @test Float32.(Array(scaled_dot_product_attention(qd, kd, v8d))) ≈
              scaled_dot_product_attention(q, k, v8) rtol=2f-2

        # a head dimension that is not a multiple of 8
        q60 = rand(Float32, 60, 2, 16, 1) ./ 4
        q60d = half(T, q60)
        @test sdpa_fused(q60d, q60d, q60d)
        @test Float32.(Array(scaled_dot_product_attention(q60d, q60d, q60d))) ≈
              scaled_dot_product_attention(q60, q60, q60) rtol=2f-2
    end

    @testset "gradient" begin
        loss(kws) = (q, k, v) -> sum(scaled_dot_product_attention(q, k, v; kws...))
        for kws in ((;), (; scale=0.25f0))
            refgrads = gradient(loss(kws), q, k, v)
            grads = gradient(loss(kws), qd, kd, vd)
            for (g, refg) in zip(grads, refgrads)
                @test g isa CuArray{T,4}
                @test Float32.(Array(g)) ≈ refg rtol=5f-2
            end
        end

        refgrad = gradient(q -> sum(scaled_dot_product_attention(q, q, q; is_causal=true)), q)[1]
        grad = gradient(q -> sum(scaled_dot_product_attention(q, q, q; is_causal=true)), qd)[1]
        @test grad isa CuArray{T,4}
        @test Float32.(Array(grad)) ≈ refgrad rtol=5f-2

        lengths = (; q_lengths=[16, 9, 1], kv_lengths=[12, 5, 7])
        for kws in (lengths, (; lengths..., is_causal=true), (; lengths..., is_causal=true, q_offset=[2, 0, 5]))
            refgrads = gradient(loss(kws), q, k, v)
            grads = gradient(loss(kws), qd, kd, vd)
            for (g, refg) in zip(grads, refgrads)
                @test g isa CuArray{T,4}
                @test Float32.(Array(g)) ≈ refg rtol=5f-2
            end
            # no gradient reaches the padding
            @test all(iszero, Array(grads[1])[:, :, 10:end, 2])
            @test all(iszero, Array(grads[2])[:, :, 6:end, 2])
            @test all(iszero, Array(grads[3])[:, :, 6:end, 2])
        end

        qg = rand(Float32, 32, 4, 16, 3) ./ 4
        refgrads = gradient(loss((;)), qg, k, v)
        grads = gradient(loss((;)), half(T, qg), kd, vd)
        for (g, refg) in zip(grads, refgrads)
            @test g isa CuArray{T,4}
            @test Float32.(Array(g)) ≈ refg rtol=5f-2
        end

        # the pullback can be called more than once
        y, back = Zygote.pullback(loss((;)), qd, kd, vd)
        g1 = map(Array, back(one(y)))
        g2 = map(Array, back(one(y)))
        @test g1 == g2

        q60d = half(T, rand(Float32, 60, 2, 16, 1) ./ 4)
        @test gradient(q -> sum(scaled_dot_product_attention(q, q, q)), q60d)[1] isa
              CuArray{T,4}
    end
end

@testset "scaled dot-product attention (Float32)" begin
    q = rand(Float32, 32, 2, 16, 3)
    k = rand(Float32, 32, 2, 12, 3)
    v = rand(Float32, 32, 2, 12, 3)
    gputest(scaled_dot_product_attention, q, k, v; rtol=1f-4, atol=1f-5)
    gputest((q, k, v) -> scaled_dot_product_attention(q, k, v; is_causal=true), q, k, v;
            rtol=1f-4, atol=1f-5)
    gputest(first ∘ scaled_dot_product_attention_with_weights, q, k, v; rtol=1f-4, atol=1f-5)
    gputest(last ∘ scaled_dot_product_attention_with_weights, q, k, v; rtol=1f-4, atol=1f-5)

    @test NNlib.is_fused_attention(CuArray(q), CuArray(k), CuArray(v))

    # lengths given on the CPU are moved to the device of the inputs
    lengths = (; q_lengths=[16, 9, 1], kv_lengths=[12, 5, 7])
    gputest((q, k, v) -> scaled_dot_product_attention(q, k, v; lengths...), q, k, v;
            rtol=1f-4, atol=1f-5)
    gputest((q, k, v) -> scaled_dot_product_attention(q, k, v; lengths..., is_causal=true),
            q, k, v; rtol=1f-4, atol=1f-5)
    gputest((q, k, v) -> last(scaled_dot_product_attention_with_weights(q, k, v; lengths...)),
            q, k, v; rtol=1f-4, atol=1f-5)

    # offsets of the queries, in the fused and the materialized implementations
    gputest((q, k, v) -> scaled_dot_product_attention(q, k, v; lengths..., is_causal=true, q_offset=[2, 0, 5]),
            q, k, v; rtol=1f-4, atol=1f-5)
    for q_offset in (3, [2, 0, 5])
        gputest((q, k, v) -> last(scaled_dot_product_attention_with_weights(q, k, v; is_causal=true, q_offset)),
                q, k, v; rtol=1f-4, atol=1f-5)
    end

    # a bias and a mask following the batch dimensions of the inputs
    q2, k2, v2 = map(x -> repeat(x; outer=(1, 1, 1, 1, 2)), (q, k, v))
    mask = rand(Bool, 12, 16, 1, 3, 1)
    mask[1, :, :, :, :] .= true
    gputest((q, k, v) -> scaled_dot_product_attention(q, k, v; mask=q isa CuArray ? CuArray(mask) : mask),
            q2, k2, v2; rtol=1f-4, atol=1f-5)
    gputest(scaled_dot_product_attention, q2, k2, v2, randn(Float32, 12, 16, 1, 3, 2);
            rtol=1f-4, atol=1f-5)
    gputest(scaled_dot_product_attention, q2, k2, v2, randn(Float32, 12, 16, 1, 1, 2);
            rtol=1f-4, atol=1f-5)
end

@testset "fused kernel ($T)" for T in (Float32, Float64)
    rtol = sqrt(eps(T))
    # a ragged problem: sequence lengths that are not multiples of the tiles
    head_dim, v_head_dim, nheads, lenq, lenkv, bs = 12, 44, 6, 100, 77, 3
    q = CUDA.randn(T, head_dim, nheads, lenq, bs) ./ 2
    k = CUDA.randn(T, head_dim, nheads, lenkv, bs) ./ 2
    v = CUDA.randn(T, v_head_dim, nheads, lenkv, bs)
    @test NNlib.is_fused_attention(q, k, v)
    @test !NNlib.is_fused_attention(q, k, v; fdrop = x -> x)

    y = @inferred scaled_dot_product_attention(q, k, v)
    @test y isa CuArray{T,4}
    @test size(y) == (v_head_dim, nheads, lenq, bs)
    @test Array(y) ≈ Array(materialized_attention(q, k, v)) rtol=rtol
    @test Array(scaled_dot_product_attention(q, k, v; is_causal=true, scale=0.3)) ≈
          Array(materialized_attention(q, k, v; is_causal=true, scale=0.3)) rtol=rtol

    lengths = (; q_lengths=[100, 37, 1], kv_lengths=[77, 12, 70])
    mask = CUDA.rand(Bool, lenkv, lenq, 1, bs)
    mask[1, :, :, :] .= true
    bias = CUDA.randn(T, lenkv, lenq)
    yl = Array(scaled_dot_product_attention(q, k, v, bias; mask, is_causal=true, lengths...))
    @test yl ≈ Array(materialized_attention(q, k, v, bias; mask, is_causal=true, lengths...)) rtol=rtol
    @test all(iszero, yl[:, :, 38:end, 2])
    # grouped-query attention
    kg, vg = k[:, 1:2, :, :], v[:, 1:2, :, :]
    @test Array(scaled_dot_product_attention(q, kg, vg)) ≈ Array(materialized_attention(q, kg, vg)) rtol=rtol

    # a causal mask shifted by an offset, given on the host or the device
    for q_offset in (5, [0, 40, 77], CuArray([0, 40, 77]))
        @test Array(scaled_dot_product_attention(q, k, v, bias; mask, is_causal=true, q_offset, lengths...)) ≈
              Array(materialized_attention(q, k, v, bias; mask, is_causal=true, q_offset, lengths...)) rtol=rtol
        @test Array(scaled_dot_product_attention(q, kg, vg; is_causal=true, q_offset)) ≈
              Array(materialized_attention(q, kg, vg; is_causal=true, q_offset)) rtol=rtol
    end
    # a negative one leaves the first queries without a key
    yn = Array(scaled_dot_product_attention(q, k, v; is_causal=true, q_offset=-3))
    @test all(isnan, yn[:, :, 1:3, :])
    @test yn[:, :, 4:end, :] ≈ Array(materialized_attention(q, k, v; is_causal=true, q_offset=-3))[:, :, 4:end, :] rtol=rtol

    # a query attending to no key gives NaN, as in the materialized implementation
    m0 = CUDA.ones(Bool, lenkv, lenq)
    m0[:, 2] .= false
    y0 = Array(scaled_dot_product_attention(q, k, v; mask=m0))
    @test all(isnan, y0[:, :, 2, :])
    @test y0[:, :, 3:end, :] ≈ Array(y)[:, :, 3:end, :] rtol=rtol

    # a sequence without a valid key
    ye = Array(scaled_dot_product_attention(q, k, v; kv_lengths=[0, 12, 70]))
    @test all(isnan, ye[:, :, :, 1])
    @test ye[:, :, :, 2:3] ≈ Array(materialized_attention(q, k, v; kv_lengths=[77, 12, 70]))[:, :, :, 2:3] rtol=rtol

    # a bias or a mask on another device is left to the materialized implementation
    @test !NNlib.is_fused_attention(q, k, v; mask=Array(mask))
end

# The value head dimension sets the threads of a block and the tiles in their registers,
# and a small problem takes smaller tiles of queries.
@testset "fused kernel tilings" begin
    @testset "value head dimension $v_head_dim, $nheads heads" for (v_head_dim, nheads) in
            ((5, 2), (64, 2), (64, 40), (128, 2), (128, 40), (256, 2), (600, 2), (2048, 1))
        q = CUDA.randn(Float32, 16, nheads, 150, 2)
        k = CUDA.randn(Float32, 16, nheads, 131, 2)
        v = CUDA.randn(Float32, v_head_dim, nheads, 131, 2)
        @test NNlib.is_fused_attention(q, k, v)
        for q_offset in (0, 37)
            @test Array(scaled_dot_product_attention(q, k, v; is_causal=true, q_offset)) ≈
                  Array(materialized_attention(q, k, v; is_causal=true, q_offset)) rtol=1f-4
        end
    end
    # beyond the head dimensions the kernel has a tiling for
    q = CUDA.randn(Float32, 16, 1, 4, 1)
    v = CUDA.randn(Float32, 4096, 1, 4, 1)
    @test !NNlib.is_fused_attention(q, q, v)
    @test size(scaled_dot_product_attention(q, q, v)) == (4096, 1, 4, 1)
end

# The gradient of NNlib's kernel, against that of the materialized implementation on the CPU.
@testset "fused gradient ($T)" for T in (Float32, Float64)
    rtol = T == Float32 ? 1f-3 : 1e-8
    head_dim, v_head_dim, nheads, lenq, lenkv, bs = 12, 44, 6, 100, 77, 3
    q = randn(T, head_dim, nheads, lenq, bs) ./ 2
    k = randn(T, head_dim, nheads, lenkv, bs) ./ 2
    v = randn(T, v_head_dim, nheads, lenkv, bs)
    w = randn(T, v_head_dim, nheads, lenq, bs)
    wd = CuArray(w)
    loss(f, w; kws...) = (x...) -> sum(w .* f(x...; kws...))

    function same_gradients(xs...; kws...)
        gm = gradient(loss(materialized_attention, w; kws...), xs...)
        kwsd = map(x -> x isa AbstractArray{Bool} ? CuArray(x) : x, values(kws))
        g = gradient(loss(scaled_dot_product_attention, wd; kwsd...), map(CuArray, xs)...)
        return all(x -> x isa CuArray{T,4}, g) && all(isapprox.(map(Array, g), gm; rtol))
    end

    mask = rand(Bool, lenkv, lenq, 1, bs)
    mask[1, :, :, :] .= true
    lengths = (; q_lengths=[100, 37, 1], kv_lengths=[77, 12, 70])
    @test same_gradients(q, k, v)
    @test same_gradients(q, k, v; is_causal=true)
    @test same_gradients(q, k, v; scale=0.3)
    @test same_gradients(q, k, v; lengths...)
    @test same_gradients(q, k, v; mask)
    @test same_gradients(q, k, v; mask, is_causal=true, lengths...)
    # grouped-query attention
    @test same_gradients(q, k[:, 1:2, :, :], v[:, 1:2, :, :])
    @test same_gradients(q, k[:, 1:2, :, :], v[:, 1:2, :, :]; is_causal=true, lengths...)
    # offsets of the queries
    @test same_gradients(q, k, v; is_causal=true, q_offset=[0, 40, 77])
    @test same_gradients(q, k, v; mask, is_causal=true, q_offset=5, lengths...)
    @test same_gradients(q, k[:, 1:2, :, :], v[:, 1:2, :, :]; is_causal=true, q_offset=[0, 40, 77], lengths...)

    qd, kd, vd = CuArray(q), CuArray(k), CuArray(v)
    # the same array as query, key and value
    x = randn(T, head_dim, nheads, lenq, bs) ./ 2
    wx = randn(T, head_dim, nheads, lenq, bs)
    wxd = CuArray(wx)
    @test Array(gradient(x -> sum(wxd .* scaled_dot_product_attention(x, x, x; is_causal=true)), CuArray(x))[1]) ≈
          gradient(x -> sum(wx .* materialized_attention(x, x, x; is_causal=true)), x)[1] rtol=rtol

    # with respect to the scale
    ds = gradient(s -> sum(wd .* scaled_dot_product_attention(qd, kd, vd; scale=s)), T(0.3))[1]
    @test ds ≈ gradient(s -> sum(w .* materialized_attention(q, k, v; scale=s)), T(0.3))[1] rtol=rtol

    # no gradient reaches the padding, and a query attending to no key takes no part
    dq, dk, dv = map(Array, gradient(loss(scaled_dot_product_attention, wd; lengths...), qd, kd, vd))
    @test all(iszero, dq[:, :, 38:end, 2])
    @test all(iszero, dk[:, :, 13:end, 2]) && all(iszero, dv[:, :, 13:end, 2])
    m0 = CUDA.ones(Bool, lenkv, lenq)
    m0[:, 2] .= false
    dq, dk, dv = map(Array, gradient(loss(scaled_dot_product_attention, wd; mask=m0), qd, kd, vd))
    @test all(iszero, dq[:, :, 2, :])
    @test all(isfinite, dk) && all(isfinite, dv)
    dq, dk, dv = map(Array, gradient(loss(scaled_dot_product_attention, wd; is_causal=true, q_offset=-3), qd, kd, vd))
    @test all(iszero, dq[:, :, 1:3, :])
    @test all(isfinite, dk) && all(isfinite, dv)

    # with a bias the materialized implementation is differentiated
    bias = randn(T, lenkv, lenq)
    @test Array(gradient(b -> sum(wd .* scaled_dot_product_attention(qd, kd, vd, b)), CuArray(bias))[1]) ≈
          gradient(b -> sum(w .* materialized_attention(q, k, v, b)), bias)[1] rtol=rtol
end

# Each gradient has a tiling of its own, set by the head dimension it has.
@testset "fused gradient tilings" begin
    @testset "head dimensions $head_dim and $v_head_dim, $nheads heads" for (head_dim, v_head_dim, nheads) in
            ((5, 3, 2), (64, 64, 2), (64, 64, 40), (128, 128, 2), (128, 16, 40), (16, 256, 2), (600, 16, 2),
             (16, 2048, 1), (2048, 16, 1))
        q = randn(Float32, head_dim, nheads, 150, 2) ./ 2
        k = randn(Float32, head_dim, nheads, 131, 2) ./ 2
        v = randn(Float32, v_head_dim, nheads, 131, 2)
        w = randn(Float32, v_head_dim, nheads, 150, 2)
        wd = CuArray(w)
        @test NNlib.is_fused_attention(CuArray(q), CuArray(k), CuArray(v))
        for q_offset in (0, 37)
            g = gradient((x...) -> sum(wd .* scaled_dot_product_attention(x...; is_causal=true, q_offset)),
                         CuArray(q), CuArray(k), CuArray(v))
            gm = gradient((x...) -> sum(w .* materialized_attention(x...; is_causal=true, q_offset)), q, k, v)
            @test all(isapprox.(map(Array, g), gm; rtol=1f-3))
        end
    end
    # beyond the head dimensions the kernel has a tiling for
    q = CUDA.randn(Float32, 4096, 1, 4, 1)
    v = CUDA.randn(Float32, 16, 1, 4, 1)
    @test !NNlib.is_fused_attention(q, q, v)
end

# Half precision takes NNlib's kernel where cuDNN has no engine, here for the head
# dimension, and its gradient is accumulated in single precision.
@testset "fused gradient ($T)" for T in (Float16, BFloat16)
    q = randn(Float32, 12, 4, 50, 2) ./ 2
    k = randn(Float32, 12, 4, 70, 2) ./ 2
    v = randn(Float32, 12, 4, 70, 2)
    w = randn(Float32, 12, 4, 50, 2)
    qd, kd, vd, wd = map(x -> CuArray{T}(x), (q, k, v, w))
    g = gradient((x...) -> sum(wd .* scaled_dot_product_attention(x...; is_causal=true)), qd, kd, vd)
    gm = gradient((x...) -> sum(w .* materialized_attention(x...; is_causal=true)), q, k, v)
    @test all(x -> x isa CuArray{T,4}, g)
    @test all(isapprox.(map(x -> Float32.(Array(x)), g), gm; rtol=T == Float16 ? 2f-2 : 5f-2))
end

@testset "deprecated interface" begin
    q = rand(Float32, 64, 16, 3) ./ 4
    qd = CuArray(Float16.(q))
    y, α = @test_deprecated dot_product_attention(q, q, q; nheads=2)
    yd, αd = @test_deprecated dot_product_attention(qd, qd, qd; nheads=2)
    @test size(yd) == (64, 16, 3)
    @test size(αd) == (16, 16, 2, 3)
    @test Float32.(Array(yd)) ≈ y rtol=2f-2
    @test Float32.(Array(αd)) ≈ α rtol=2f-2 atol=1f-3
end
