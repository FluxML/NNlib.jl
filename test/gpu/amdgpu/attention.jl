@testset "Compare CPU & GPU" begin
    n = 15
    lenq = 3
    lenkv = 4
    for batch_size in [(), 1, 2, (2, 1, 3)], nheads in [1, 3, 5]
        q = AMDGPU.rand(Float32, n, lenq, batch_size...)
        k = AMDGPU.rand(Float32, n, lenkv, batch_size...)
        v = AMDGPU.rand(Float32, n, lenkv, batch_size...)
        y, α = @inferred dot_product_attention(q, k, v; nheads)

        @test y isa ROCArray{Float32}
        @test size(y) == (n, lenq, batch_size...)
        @test size(α) == (lenkv, lenq, nheads, batch_size...)
        @test sum(Array(α), dims=1) ≈ ones(1, lenq, nheads, batch_size...)

        qh = rand(Float32, n, lenq, batch_size...)
        kh = rand(Float32, n, lenkv, batch_size...)
        vh = rand(Float32, n, lenkv, batch_size...)
        gputest(
            (x...) -> dot_product_attention(x...; nheads)[1], qh, kh, vh;
            atol=1f-5)
    end
end

@testset "Mask" begin
    x = AMDGPU.rand(Float32, 4, 2, 3, 1)
    mask = make_causal_mask(x, dims=3)
    @test mask isa ROCArray{Bool}
    α = dot_product_attention_scores(x, x; mask)

    α_host, mask_host = Array.((α, mask))
    @test all((α_host[:, :, 1, 1] .> 0) .== mask_host)
    @test all((α_host[:, :, 2, 1] .> 0) .== mask_host)
end

@testset "Dropout" begin
    q = k = v = AMDGPU.rand(Float32, 10, 10, 10)
    fdrop(x, p) = (rand!(similar(x)) .> p) .* x ./ (1-p)
    y, α = dot_product_attention(
        q, k, v; nheads=2, fdrop=x -> dropout(x, 0.5))
    @test 0.6 > mean(>(0), α) > 0.4
end
