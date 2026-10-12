# the attention weights alone
attention_weights(q, k, args...; kws...) =
    last(scaled_dot_product_attention_with_weights(q, k, k, args...; kws...))

@testset "different batchsizes" begin
    head_dim = 5
    lenq = 3
    lenkv = 4
    for batch_size in [(), 1, 2, (2,1,3)], nheads in [1, 3, 5]
        q = rand(Float32, head_dim, nheads, lenq, batch_size...)
        k = rand(Float32, head_dim, nheads, lenkv, batch_size...)
        v = rand(Float32, head_dim, nheads, lenkv, batch_size...)
        y = scaled_dot_product_attention(q, k, v)
        α = attention_weights(q, k)
        @test y isa Array{Float32}
        @test size(y) == (head_dim, nheads, lenq, batch_size...)
        @test size(α) == (lenkv, lenq, nheads, batch_size...)
        @test sum(α, dims=1) ≈ ones(1, lenq, nheads, batch_size...)
    end
end

@testset "output and weights in one pass" begin
    q = rand(Float32, 4, 2, 3, 1)
    k = rand(Float32, 4, 2, 5, 1)
    v = rand(Float32, 6, 2, 5, 1)
    y, α = @inferred scaled_dot_product_attention_with_weights(q, k, v)
    @test y == scaled_dot_product_attention(q, k, v)
    @test size(α) == (5, 3, 2, 1)

    # the output is computed from the returned weights, also with a stochastic `fdrop`
    fdrop(α) = α .* (rand(Float32, size(α)) .> 0.5f0)
    y, α = scaled_dot_product_attention_with_weights(q, k, v; fdrop)
    @test any(iszero, α)
    @test y ≈ [sum(α[j, i, h, b] * v[d, h, j, b] for j in 1:5) for d in 1:6, h in 1:2, i in 1:3, b in 1:1]

    # batch dimensions are restored in both
    qb, kb, vb = map(x -> repeat(x; outer=(1, 1, 1, 2, 3)), (q, k, v))
    y, α = @inferred scaled_dot_product_attention_with_weights(qb, kb, vb)
    @test size(y) == (6, 2, 3, 2, 3)
    @test size(α) == (5, 3, 2, 2, 3)

    @test sdpa === scaled_dot_product_attention
    @test sdpa_with_weights === scaled_dot_product_attention_with_weights

    qd, kd, vd = map(x -> Float64.(x), (q, k, v))
    # a loss on both the output and the weights
    @test test_gradients((x...) -> scaled_dot_product_attention_with_weights(x...), qd, kd, vd;
                         loss = (f, xs...) -> sum(x -> mean(abs2, x), f(xs...)))
end

@testset "weights ↔ output consistency" begin
    q = rand(Float32, 4, 2, 3, 1)
    k = v = rand(Float32, 4, 2, 5, 1)
    y, α = scaled_dot_product_attention_with_weights(q, k, v)
    # y[:, h, i, b] = Σ_j α[j, i, h, b] * v[:, h, j, b]
    yref = similar(y)
    for b in 1:1, h in 1:2, i in 1:3
        yref[:, h, i, b] = sum(α[j, i, h, b] .* v[:, h, j, b] for j in 1:5)
    end
    @test y ≈ yref
end

@testset "specific results" begin
    # (head_dim, nheads, seq_len, batch) = (2, 2, 3, 1)
    q = k = v = reshape([1:12;] ./ 12, 2, 2, 3, 1)
    y = scaled_dot_product_attention(q, k, v)
    α = attention_weights(q, k)
    ytrue = [0.429754, 0.513087, 0.613791, 0.697125, 0.46431, 0.547644, 0.647876, 0.73121, 0.49773, 0.581064, 0.680455, 0.763788]
    ytrue = reshape(ytrue, 4, 3, 1)
    αtrue = [0.313896, 0.332948, 0.353157, 0.264431, 0.328206, 0.407362, 0.219215, 0.31838, 0.462405, 0.288691, 0.331243, 0.380066, 0.241239, 0.323893, 0.434868, 0.198438, 0.311761, 0.489801]
    αtrue = reshape(αtrue, 3, 3, 2, 1)
    @test reshape(y, 4, 3, 1) ≈ ytrue atol=1e-5   # heads joined back for comparison
    @test α ≈ αtrue atol=1e-5
end

@testset "mask" begin
    q = rand(4, 2, 3, 1)
    k = rand(4, 2, 5, 1)

    mask = rand(Bool, (5, 3))
    α = attention_weights(q, k; mask)
    @test all((α[:,:,1,1].> 0) .== mask)
    @test all((α[:,:,2,1].> 0) .== mask)

    @testset "causal" begin
        x = rand(4, 2, 3, 1)
        mask = make_causal_mask(x, dims=3)
        α = attention_weights(x, x; mask)
        @test all((α[:,:,1,1].> 0) .== mask)
        @test all((α[:,:,2,1].> 0) .== mask)
    end
end

@testset "dropout" begin
    q = k = rand(5, 2, 10)
    fdrop(x, p) = (Random.rand!(similar(x)) .> p) .* x ./ (1-p)
    α = attention_weights(q, k; fdrop = x -> fdrop(x, 0.5))
    @test 0.6 > mean(>(0), α) > 0.4
end

@testset "bias" begin
    q = rand(2, 2, 5, 1)
    k = v = rand(2, 2, 3, 1)
    bias = randn(3, 5)
    y = scaled_dot_product_attention(q, k, v, bias)
    α = attention_weights(q, k, bias)
    @test size(α) == (3, 5, 2, 1)
    @test size(y) == (2, 2, 5, 1)
end

@testset "gradient" begin
    q = rand(2, 2, 5, 1)
    k = v = rand(2, 2, 3, 1)
    bias = randn(3, 5)
    @test test_gradients((x...) -> scaled_dot_product_attention(x...), q, k, v, bias)
end

@testset "scale" begin
    head_dim, nheads = 4, 2
    q = rand(Float32, head_dim, nheads, 5, 1)
    k = v = rand(Float32, head_dim, nheads, 3, 1)
    # passing the default scale explicitly reproduces the default
    α0 = attention_weights(q, k)
    α1 = attention_weights(q, k; scale=inv(sqrt(Float32(head_dim))))
    @test α1 ≈ α0
    # a different scale changes the result
    α2 = attention_weights(q, k; scale=1f0)
    @test !(α2 ≈ α0)
    # the output path accepts `scale` too
    @test size(scaled_dot_product_attention(q, k, v; scale=2f0)) == (head_dim, nheads, 5, 1)
end

@testset "is_causal" begin
    head_dim, nheads = 4, 2
    x = rand(Float32, head_dim, nheads, 4, 1)
    αcausal = attention_weights(x, x; is_causal=true)
    mask = make_causal_mask(x, dims=3)
    αmask = attention_weights(x, x; mask)
    @test αcausal ≈ αmask
    # lower triangle (key index > query index) must be masked out
    @test all(all(αcausal[j, i, :, 1] .== 0) for j in 1:4, i in 1:4 if j > i)
    # output path agrees
    @test scaled_dot_product_attention(x, x, x; is_causal=true) ≈
          scaled_dot_product_attention(x, x, x; mask)
    # `mask` and `is_causal` combine: a key is attended to where both allow it
    mask2 = rand(Bool, 4, 4)
    mask2[1, :] .= true
    αboth = attention_weights(x, x; mask=mask2, is_causal=true)
    @test all((αboth[:, :, h, 1] .> 0) == (mask2 .& mask) for h in 1:nheads)
    @test scaled_dot_product_attention(x, x, x; mask=mask2, is_causal=true) ≈
          scaled_dot_product_attention(x, x, x; mask=mask2 .& mask)
end

@testset "sequence lengths" begin
    head_dim, nheads, lenq, lenkv, bs = 4, 2, 5, 6, 3
    q = rand(Float32, head_dim, nheads, lenq, bs)
    k = rand(Float32, head_dim, nheads, lenkv, bs)
    v = rand(Float32, head_dim, nheads, lenkv, bs)
    q_lengths, kv_lengths = [5, 3, 1], [6, 2, 4]
    kvmask = reshape([j <= n for j in 1:lenkv, n in kv_lengths], lenkv, 1, 1, bs)
    qmask = reshape([i <= n for i in 1:lenq, n in q_lengths], 1, 1, lenq, bs)

    # the keys past `kv_lengths` are not attended to
    @test scaled_dot_product_attention(q, k, v; kv_lengths) ≈
          scaled_dot_product_attention(q, k, v; mask=kvmask)
    α = attention_weights(q, k; kv_lengths)
    @test (α .> 0) == (kvmask .& trues(lenkv, lenq, nheads, bs))

    # the output and the weights are zero at the queries past `q_lengths`
    y = scaled_dot_product_attention(q, k, v; q_lengths)
    @test y ≈ scaled_dot_product_attention(q, k, v) .* qmask
    @test all(iszero, y[:, :, 4:5, 2])
    @test attention_weights(q, k; q_lengths) ≈
          attention_weights(q, k) .* reshape(qmask, 1, lenq, 1, bs)

    # combined with each other, with `is_causal` and with a `mask`
    causal = NNlib.make_causal_mask(q, lenkv, lenq)
    @test scaled_dot_product_attention(q, k, v; q_lengths, kv_lengths, is_causal=true) ≈
          scaled_dot_product_attention(q, k, v; mask=causal .& kvmask) .* qmask
    mask = rand(Bool, lenkv, lenq)
    mask[1, :] .= true
    @test scaled_dot_product_attention(q, k, v; kv_lengths, mask) ≈
          scaled_dot_product_attention(q, k, v; mask=mask .& kvmask)

    # an integer is the length of every sequence, and any integer array is accepted
    @test scaled_dot_product_attention(q, k, v; kv_lengths=4) ≈
          scaled_dot_product_attention(q, k, v; kv_lengths=[4, 4, 4])
    @test scaled_dot_product_attention(q, k, v; kv_lengths=Int32.(kv_lengths)) ==
          scaled_dot_product_attention(q, k, v; kv_lengths)
    # lengths reaching the end of the sequences mean no padding
    @test scaled_dot_product_attention(q, k, v; q_lengths=lenq, kv_lengths=fill(lenkv + 2, bs)) ≈
          scaled_dot_product_attention(q, k, v)

    # one length for each batch element, whatever the batch dimensions
    qb, kb, vb = map(x -> reshape(x, size(x)[1:3]..., bs, 1), (q, k, v))
    yb = scaled_dot_product_attention(qb, kb, vb; q_lengths=reshape(q_lengths, bs, 1), kv_lengths)
    @test yb == reshape(scaled_dot_product_attention(q, k, v; q_lengths, kv_lengths), size(yb))
    @test scaled_dot_product_attention(q[:, :, :, 2], k[:, :, :, 2], v[:, :, :, 2]; kv_lengths=2) ≈
          scaled_dot_product_attention(q, k, v; kv_lengths=2)[:, :, :, 2]
    @test_throws ArgumentError scaled_dot_product_attention(q, k, v; kv_lengths=[6, 2])

    @inferred scaled_dot_product_attention(q, k, v; q_lengths, kv_lengths, is_causal=true)

    qd, kd, vd = map(x -> Float64.(x), (q, k, v))
    @test test_gradients((x...) -> scaled_dot_product_attention(x...; q_lengths, kv_lengths, is_causal=true),
                         qd, kd, vd)
end

@testset "bias and mask with several batch dimensions" begin
    head_dim, nheads, lenq, lenkv = 4, 2, 3, 5
    batch_size = (2, 3)
    q = rand(Float32, head_dim, nheads, lenq, batch_size...)
    k = rand(Float32, head_dim, nheads, lenkv, batch_size...)
    v = rand(Float32, head_dim, nheads, lenkv, batch_size...)
    flat(x) = reshape(x, size(x)[1:3]..., :)
    yref(; kws...) = reshape(scaled_dot_product_attention(flat(q), flat(k), flat(v); kws...), size(q))

    # trailing dimensions equal to the batch ones, a prefix of them, with singletons,
    # or already collapsed
    @testset "trailing dimensions $sz" for sz in ((2, 3), (2,), (2, 1), (1, 3), (1, 1), (6,))
        full(x) = sz == (6,) ? x : flat(x .+ zeros(Bool, 1, 1, 1, batch_size...))

        mask = rand(Bool, lenkv, lenq, 1, sz...)
        selectdim(mask, 1, 1) .= true
        @test scaled_dot_product_attention(q, k, v; mask) ≈ yref(mask=Bool.(full(mask)))
        α = attention_weights(q, k; mask)
        @test flat(α .> 0) == (full(mask) .> 0) .& trues(lenkv, lenq, nheads, 6)

        bias = randn(Float32, lenkv, lenq, 1, sz...)
        @test scaled_dot_product_attention(q, k, v, bias) ≈
              reshape(scaled_dot_product_attention(flat(q), flat(k), flat(v), full(bias)), size(q))
    end

    @test_throws DimensionMismatch scaled_dot_product_attention(q, k, v; mask=trues(lenkv, lenq, 1, 3, 2))
    @test_throws DimensionMismatch scaled_dot_product_attention(q, k, v, zeros(Float32, lenkv, lenq, 1, 4))
    @test_throws DimensionMismatch scaled_dot_product_attention(q, k, v; mask=trues(lenkv, lenq, 1, 2, 3, 1, 2))

    @inferred scaled_dot_product_attention(q, k, v, zeros(Float32, lenkv, lenq, 1, 2, 3); mask=trues(lenkv, lenq, 1, 2, 1))

    qd, kd, vd = map(x -> Float64.(x), (q, k, v))
    for sz in ((2, 3), (2, 1))
        @test test_gradients((x...) -> scaled_dot_product_attention(x...), qd, kd, vd, randn(lenkv, lenq, 1, sz...))
    end
end

@testset "input validation" begin
    q = rand(Float32, 4, 2, 3, 2)
    k = v = rand(Float32, 4, 2, 5, 2)
    # arrays with different numbers of dimensions
    @test_throws ArgumentError scaled_dot_product_attention(q, k[:, :, :, 1], v)
    @test_throws ArgumentError scaled_dot_product_attention(q, k, v[:, :, :, 1])
    @test_throws ArgumentError scaled_dot_product_attention_with_weights(q, k[:, :, :, 1], v)
    @test_throws ArgumentError scaled_dot_product_attention(q[:, :, 1, 1], k[:, :, 1, 1], v[:, :, 1, 1])
    # head and batch dimensions
    @test_throws ArgumentError scaled_dot_product_attention(q, k[1:2, :, :, :], v)
    @test_throws ArgumentError scaled_dot_product_attention(q, k, v[:, :, :, 1:1])
    @test_throws ArgumentError scaled_dot_product_attention_with_weights(q, k[:, :, :, 1:1], v)
end

@testset "is_fused_attention" begin
    q = rand(Float32, 8, 2, 3, 2)
    @test !NNlib.is_fused_attention(q, q, q)
    @test !NNlib.is_fused_attention(q, q, q; is_causal=true, kv_lengths=[3, 2])
    @test_throws ArgumentError NNlib.is_fused_attention(q, q[:, :, :, 1], q)
end

@testset "grouped-query attention" begin
    head_dim, nheads, nkvheads = 4, 6, 2
    q = rand(Float32, head_dim, nheads, 5, 2)
    k = rand(Float32, head_dim, nkvheads, 3, 2)
    v = rand(Float32, head_dim, nkvheads, 3, 2)

    y = scaled_dot_product_attention(q, k, v)
    α = attention_weights(q, k)
    @test size(y) == (head_dim, nheads, 5, 2)
    @test size(α) == (3, 5, nheads, 2)
    @test sum(α, dims=1) ≈ ones(Float32, 1, 5, nheads, 2)

    # GQA must match standard MHA with the kv heads expanded along the heads axis
    r = nheads ÷ nkvheads
    kf = repeat(k; inner=(1, r, 1, 1))
    vf = repeat(v; inner=(1, r, 1, 1))
    @test y ≈ scaled_dot_product_attention(q, kf, vf)
    @test α ≈ attention_weights(q, kf)

    # number of query heads must be divisible by the number of kv heads
    badk = rand(Float32, head_dim, 4, 3, 2)   # 6 % 4 != 0
    @test_throws ArgumentError scaled_dot_product_attention(q, badk, badk)

    qd = rand(head_dim, nheads, 5, 2)
    kd = rand(head_dim, nkvheads, 3, 2)
    vd = rand(head_dim, nkvheads, 3, 2)
    @test test_gradients((qq, kk, vv) -> scaled_dot_product_attention(qq, kk, vv), qd, kd, vd)
end

@testset "deprecated interface" begin
    # heads packed into the feature dimension: (features, seq_len, batch)
    nheads = 2
    q = rand(Float32, 8, 5, 3)
    k = v = rand(Float32, 8, 4, 3)
    y, α = @test_deprecated dot_product_attention(q, k, v; nheads)
    qh, kh, vh = map(x -> reshape(x, 4, nheads, size(x, 2), 3), (q, k, v))
    @test y ≈ reshape(scaled_dot_product_attention(qh, kh, vh), 8, 5, 3)
    @test α ≈ attention_weights(qh, kh)
    @test α ≈ @test_deprecated dot_product_attention_scores(qh, kh)
    @test α ≈ @test_deprecated scaled_dot_product_attention_scores(qh, kh)

    # without `dims`, the mask is still the one for the deprecated layout
    mask = @test_deprecated make_causal_mask(q)
    @test mask == make_causal_mask(q; dims=2)
    @test size(mask) == (5, 5)
    y, α = @test_deprecated dot_product_attention(q, q, q; nheads, mask)
    @test all((α[:, :, 1, 1] .> 0) .== mask)
end
