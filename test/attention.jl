@testset "different batchsizes" begin
    head_dim = 5
    lenq = 3
    lenkv = 4
    for batch_size in [(), 1, 2, (2,1,3)], nheads in [1, 3, 5]
        q = rand(Float32, head_dim, nheads, lenq, batch_size...)
        k = rand(Float32, head_dim, nheads, lenkv, batch_size...)
        v = rand(Float32, head_dim, nheads, lenkv, batch_size...)
        y = scaled_dot_product_attention(q, k, v)
        α = scaled_dot_product_attention_scores(q, k)
        @test y isa Array{Float32}
        @test size(y) == (head_dim, nheads, lenq, batch_size...)
        @test size(α) == (lenkv, lenq, nheads, batch_size...)
        @test sum(α, dims=1) ≈ ones(1, lenq, nheads, batch_size...)
    end
end

@testset "scores ↔ output consistency" begin
    q = rand(Float32, 4, 2, 3, 1)
    k = v = rand(Float32, 4, 2, 5, 1)
    α = scaled_dot_product_attention_scores(q, k)
    y = scaled_dot_product_attention(q, k, v)
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
    α = scaled_dot_product_attention_scores(q, k)
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
    α = scaled_dot_product_attention_scores(q, k; mask)
    @test all((α[:,:,1,1].> 0) .== mask)
    @test all((α[:,:,2,1].> 0) .== mask)

    @testset "causal" begin
        x = rand(4, 2, 3, 1)
        mask = make_causal_mask(x, dims=3)
        α = scaled_dot_product_attention_scores(x, x; mask)
        @test all((α[:,:,1,1].> 0) .== mask)
        @test all((α[:,:,2,1].> 0) .== mask)
    end
end

@testset "dropout" begin
    q = k = rand(5, 2, 10)
    fdrop(x, p) = (rand!(similar(x)) .> p) .* x ./ (1-p)
    α = scaled_dot_product_attention_scores(q, k; fdrop = x -> fdrop(x, 0.5))
    @test 0.6 > mean(>(0), α) > 0.4
end

@testset "bias" begin
    q = rand(2, 2, 5, 1)
    k = v = rand(2, 2, 3, 1)
    bias = randn(3, 5)
    y = scaled_dot_product_attention(q, k, v, bias)
    α = scaled_dot_product_attention_scores(q, k, bias)
    @test size(α) == (3, 5, 2, 1)
    @test size(y) == (2, 2, 5, 1)
end

@testset "gradient" begin
    q = rand(2, 2, 5, 1)
    k = v = rand(2, 2, 3, 1)
    bias = randn(3, 5)
    gradtest((x...) -> scaled_dot_product_attention(x...), q, k, v, bias)
end

@testset "scale" begin
    head_dim, nheads = 4, 2
    q = rand(Float32, head_dim, nheads, 5, 1)
    k = v = rand(Float32, head_dim, nheads, 3, 1)
    # passing the default scale explicitly reproduces the default
    α0 = scaled_dot_product_attention_scores(q, k)
    α1 = scaled_dot_product_attention_scores(q, k; scale=sqrt(Float32(head_dim)))
    @test α1 ≈ α0
    # a different scale changes the result
    α2 = scaled_dot_product_attention_scores(q, k; scale=1f0)
    @test !(α2 ≈ α0)
    # the output path accepts `scale` too
    @test size(scaled_dot_product_attention(q, k, v; scale=2f0)) == (head_dim, nheads, 5, 1)
end

@testset "is_causal" begin
    head_dim, nheads = 4, 2
    x = rand(Float32, head_dim, nheads, 4, 1)
    αcausal = scaled_dot_product_attention_scores(x, x; is_causal=true)
    mask = make_causal_mask(x, dims=3)
    αmask = scaled_dot_product_attention_scores(x, x; mask)
    @test αcausal ≈ αmask
    # lower triangle (key index > query index) must be masked out
    @test all(all(αcausal[j, i, :, 1] .== 0) for j in 1:4, i in 1:4 if j > i)
    # output path agrees
    @test scaled_dot_product_attention(x, x, x; is_causal=true) ≈
          scaled_dot_product_attention(x, x, x; mask)
    # `mask` and `is_causal` are mutually exclusive
    @test_throws ArgumentError scaled_dot_product_attention(x, x, x; mask, is_causal=true)
end

@testset "grouped-query attention" begin
    head_dim, nheads, nkvheads = 4, 6, 2
    q = rand(Float32, head_dim, nheads, 5, 2)
    k = rand(Float32, head_dim, nkvheads, 3, 2)
    v = rand(Float32, head_dim, nkvheads, 3, 2)

    y = scaled_dot_product_attention(q, k, v)
    α = scaled_dot_product_attention_scores(q, k)
    @test size(y) == (head_dim, nheads, 5, 2)
    @test size(α) == (3, 5, nheads, 2)
    @test sum(α, dims=1) ≈ ones(Float32, 1, 5, nheads, 2)

    # GQA must match standard MHA with the kv heads expanded along the heads axis
    r = nheads ÷ nkvheads
    kf = repeat(k; inner=(1, r, 1, 1))
    vf = repeat(v; inner=(1, r, 1, 1))
    @test y ≈ scaled_dot_product_attention(q, kf, vf)
    @test α ≈ scaled_dot_product_attention_scores(q, kf)

    # number of query heads must be divisible by the number of kv heads
    badk = rand(Float32, head_dim, 4, 3, 2)   # 6 % 4 != 0
    @test_throws ArgumentError scaled_dot_product_attention(q, badk, badk)

    qd = rand(head_dim, nheads, 5, 2)
    kd = rand(head_dim, nkvheads, 3, 2)
    vd = rand(head_dim, nkvheads, 3, 2)
    gradtest((qq, kk, vv) -> scaled_dot_product_attention(qq, kk, vv), qd, kd, vd)
end
