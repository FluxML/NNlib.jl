@testset "different batchsizes" begin
    n = 15
    lenq = 3
    lenkv = 4
    for batch_size in [(), 1, 2, (2,1,3)], nheads in [1, 3, 5]
        q = rand(Float32, n, lenq, batch_size...)
        k = rand(Float32, n, lenkv, batch_size...)
        v = rand(Float32, n, lenkv, batch_size...)
        y, α = dot_product_attention(q, k, v; nheads)
        @test y isa Array{Float32}
        @test size(y) == (n, lenq, batch_size...)
        @test size(α) == (lenkv, lenq, nheads, batch_size...)
        @test sum(α, dims=1) ≈ ones(1, lenq, nheads, batch_size...)
    end
end

@testset "dot_product_attention_scores" begin
    q = k = reshape([1:24;], 4, 2, 3, 1) ./ 24
    α = dot_product_attention_scores(q, k)
    q2, k2 = reshape.((q, k), 8, 3, 1)
    y, α2 = dot_product_attention(q2, k2, k2; nheads=2)
    @test α ≈ α2
end

@testset "specific results" begin
    q = k = v = reshape([1:12;], 4, 3, 1) ./ 12
    y, α = dot_product_attention(q, k, v; nheads=2)
    ytrue = [0.429754, 0.513087, 0.613791, 0.697125, 0.46431, 0.547644, 0.647876, 0.73121, 0.49773, 0.581064, 0.680455, 0.763788]
    ytrue = reshape(ytrue, 4, 3, 1)
    αtrue = [0.313896, 0.332948, 0.353157, 0.264431, 0.328206, 0.407362, 0.219215, 0.31838, 0.462405, 0.288691, 0.331243, 0.380066, 0.241239, 0.323893, 0.434868, 0.198438, 0.311761, 0.489801]
    αtrue = reshape(αtrue, 3, 3, 2, 1)
    @test y ≈ ytrue atol=1e-5
    @test α ≈ αtrue atol=1e-5
end

@testset "mask" begin
    q = rand(4, 2, 3, 1)
    k = rand(4, 2, 5, 1)

    mask = rand(Bool, (5, 3))
    α = dot_product_attention_scores(q, k; mask)
    @test all((α[:,:,1,1].> 0) .== mask)
    @test all((α[:,:,2,1].> 0) .== mask)

    @testset "causal" begin
        x = rand(4, 2, 3, 1)
        mask = make_causal_mask(x, dims=3)
        α = dot_product_attention_scores(x, x; mask)
        @test all((α[:,:,1,1].> 0) .== mask)
        @test all((α[:,:,2,1].> 0) .== mask)
    end
end

@testset "dropout" begin
    q = k = v = rand(10, 10, 10)
    fdrop(x, p) = (rand!(similar(x)) .> p) .* x ./ (1-p)
    y, α = dot_product_attention(q, k, v; nheads=2, fdrop = x -> fdrop(x, 0.5))
    @test 0.6 > mean(>(0), α) > 0.4
end

@testset "bias" begin
    q = rand(4, 5, 1)
    k = v = rand(4, 3, 1)
    bias = randn(3, 5)
    y, α = dot_product_attention(q, k, v, bias; nheads=2)
    @test size(α) == (3, 5, 2, 1)
    @test size(y) == (4, 5, 1)
end

@testset "gradient" begin
    q = rand(4, 5, 1)
    k = v = rand(4, 3, 1)
    bias = randn(3, 5)
    y, α = dot_product_attention(q, k, v, bias; nheads=2)
    gradtest((x...) -> dot_product_attention(x...; nheads=2)[1], q, k, v, bias)
end
