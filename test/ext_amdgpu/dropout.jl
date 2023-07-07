@testset "Test API" begin
    x = AMDGPU.randn(Float32, 3, 4)
    @test size(@inferred dropout(x, 0.1)) == (3, 4)
    @test size(@inferred dropout(x, 0.2; dims=2)) == (3, 4)
    @test size(@inferred dropout(x, 0.3; dims=(1, 2))) == (3, 4)

    rng = AMDGPU.rocrand_rng()
    @test size(@inferred dropout(rng, x, 0.1)) == (3, 4)
    @test size(@inferred dropout(rng, x, 0.1; dims=2)) == (3, 4)

    # Values
    d45 = dropout(AMDGPU.ones(100, 100, 100), 0.45)
    @test mean(d45) ≈ 1 atol=1e-2
    dpi2 = dropout(AMDGPU.fill(1f0 * pi, 1000), 0.2)
    @test sort(unique(Array(dpi2))) ≈ [0, 5 * pi / 4]
    d33 = dropout(AMDGPU.fill(3f0, 10, 1000), 0.3, dims=2)
    @test sort(unique(vec(Array(d33)))) ≈ [0, 3 / (1 - 0.3)]

    @test Zygote.gradient(x -> sum(dropout(x, 0.1)), x)[1] isa ROCArray{Float32}
end
