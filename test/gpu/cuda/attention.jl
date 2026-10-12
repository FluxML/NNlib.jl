# cuDNN's fused kernel handles half-precision inputs whose head dimension is a multiple
# of 8, with no bias, dropout or arbitrary mask. Everything else, and every `Float32`
# input, takes the generic implementation.
const sdpa_fused = NNlib.is_fused_attention

# Whether cuDNN has a fused engine for these inputs on this device: NNlib takes the fused
# kernel exactly then. Where it has none, the comparisons below check the generic path.
function sdpa_supported(q, k, v; is_causal=false, q_lengths=nothing, kv_lengths=nothing)
    padded = q_lengths !== nothing || kv_lengths !== nothing
    # cuDNN takes the lengths of both the queries and the keys, or of neither
    seqlens(x, lengths) = !padded ? nothing :
        CuArray(reshape(Int32.(something(lengths, fill(size(x, 3), size(x, 4)))), 1, 1, 1, :))
    return cuDNN.attention_supported(similar(q), q, k, v; causal=is_causal,
        seq_len_q=seqlens(q, q_lengths), seq_len_kv=seqlens(k, kv_lengths))
end

half(T, x) = T.(CuArray(x))    # round on the device (JuliaMath/BFloat16s.jl#107)
collapse(x) = reshape(x, size(x)[1:3]..., :)

@testset "scaled dot-product attention ($T)" for T in (Float16, BFloat16)
    T == BFloat16 && CUDA.capability(CUDA.device()) < v"8.0" && continue
    q = rand(Float32, 32, 2, 16, 3) ./ 4
    k = rand(Float32, 32, 2, 12, 3) ./ 4
    v = rand(Float32, 32, 2, 12, 3) ./ 4
    qd, kd, vd = half(T, q), half(T, k), half(T, v)

    @test sdpa_fused(qd, kd, vd) == sdpa_supported(qd, kd, vd)
    y = scaled_dot_product_attention(qd, kd, vd)
    @test y isa CuArray{T,4}
    @test Float32.(Array(y)) ≈ scaled_dot_product_attention(q, k, v) rtol=2f-2
    @test Float32.(Array(scaled_dot_product_attention(qd, kd, vd; scale=0.25f0))) ≈
          scaled_dot_product_attention(q, k, v; scale=0.25f0) rtol=2f-2

    @testset "causal" begin
        @test sdpa_fused(qd, qd, qd; is_causal=true) ==
              sdpa_supported(qd, qd, qd; is_causal=true)
        @test Float32.(Array(scaled_dot_product_attention(qd, qd, qd; is_causal=true))) ≈
              scaled_dot_product_attention(q, q, q; is_causal=true) rtol=2f-2
        @test Float32.(Array(scaled_dot_product_attention(qd, kd, vd; is_causal=true))) ≈
              scaled_dot_product_attention(q, k, v; is_causal=true) rtol=2f-2
    end

    @testset "grouped-query attention" begin
        qg = rand(Float32, 32, 4, 16, 3) ./ 4
        qgd = half(T, qg)
        @test sdpa_fused(qgd, kd, vd) == sdpa_supported(qgd, kd, vd)
        @test Float32.(Array(scaled_dot_product_attention(qgd, kd, vd))) ≈
              scaled_dot_product_attention(qg, k, v) rtol=2f-2
    end

    @testset "sequence lengths" begin
        q_lengths, kv_lengths = [16, 9, 1], [12, 5, 7]
        @testset "$(keys(kws))" for kws in ((; kv_lengths), (; q_lengths),
                (; q_lengths, kv_lengths), (; q_lengths, kv_lengths, is_causal=true))
            @test sdpa_fused(qd, kd, vd; kws...) == sdpa_supported(qd, kd, vd; kws...)
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
        # together with a mask they take the generic implementation
        mask = CUDA.ones(Bool, 12, 16)
        @test !sdpa_fused(qd, kd, vd; mask, q_lengths, kv_lengths)
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
        @test sdpa_fused(qbd, qbd, qbd; kv_lengths=lengths) ==
              sdpa_supported(collapse(qbd), collapse(qbd), collapse(qbd); kv_lengths=vec(lengths))
        @test Float32.(Array(scaled_dot_product_attention(qbd, qbd, qbd; kv_lengths=lengths))) ≈
              scaled_dot_product_attention(qb, qb, qb; kv_lengths=lengths) rtol=2f-2
    end

    @testset "generic fallback" begin
        mask = make_causal_mask(qd; dims=3)
        @test !sdpa_fused(qd, qd, qd; mask)
        @test Float32.(Array(scaled_dot_product_attention(qd, qd, qd; mask))) ≈
              scaled_dot_product_attention(q, q, q; mask=Array(mask)) rtol=2f-2

        bias = randn(Float32, 12, 16)
        biasd = half(T, bias)
        @test !sdpa_fused(qd, kd, vd, biasd)
        @test Float32.(Array(scaled_dot_product_attention(qd, kd, vd, biasd))) ≈
              scaled_dot_product_attention(q, k, v, bias) rtol=2f-2

        @test !sdpa_fused(qd, kd, vd; fdrop = x -> x)
        @test Float32.(Array(scaled_dot_product_attention(qd, kd, vd; fdrop = x -> x))) ≈
              scaled_dot_product_attention(q, k, v) rtol=2f-2

        # a value head dimension different from the query's
        v8 = rand(Float32, 8, 2, 12, 3) ./ 4
        v8d = half(T, v8)
        @test !sdpa_fused(qd, kd, v8d)
        @test Float32.(Array(scaled_dot_product_attention(qd, kd, v8d))) ≈
              scaled_dot_product_attention(q, k, v8) rtol=2f-2

        # a head dimension that is not a multiple of 8
        q60 = rand(Float32, 60, 2, 16, 1) ./ 4
        q60d = half(T, q60)
        @test !sdpa_fused(q60d, q60d, q60d)
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
        for kws in (lengths, (; lengths..., is_causal=true))
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
    gputest(scaled_dot_product_attention_scores, q, k; rtol=1f-4, atol=1f-5)

    @test !NNlib.is_fused_attention(CuArray(q), CuArray(k), CuArray(v))

    # lengths given on the CPU are moved to the device of the inputs
    lengths = (; q_lengths=[16, 9, 1], kv_lengths=[12, 5, 7])
    gputest((q, k, v) -> scaled_dot_product_attention(q, k, v; lengths...), q, k, v;
            rtol=1f-4, atol=1f-5)
    gputest((q, k, v) -> scaled_dot_product_attention(q, k, v; lengths..., is_causal=true),
            q, k, v; rtol=1f-4, atol=1f-5)
    gputest((q, k) -> scaled_dot_product_attention_scores(q, k; lengths...), q, k;
            rtol=1f-4, atol=1f-5)

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

@testset "deprecated dot_product_attention" begin
    q = rand(Float32, 64, 16, 3) ./ 4
    qd = CuArray(Float16.(q))
    y, α = @test_deprecated dot_product_attention(q, q, q; nheads=2)
    yd, αd = @test_deprecated dot_product_attention(qd, qd, qd; nheads=2)
    @test size(yd) == (64, 16, 3)
    @test size(αd) == (16, 16, 2, 3)
    @test Float32.(Array(yd)) ≈ y rtol=2f-2
    @test Float32.(Array(αd)) ≈ α rtol=2f-2 atol=1f-3
end
