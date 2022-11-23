using NNlib, Test

@testset "unfold wrapper" begin
    x = rand(rng, 16, 16, 3, 10)
    w = rand(rng, 5, 5, 3, 2)
    @test size(unfold(x, size(w))) == (144, 75, 10)
    @test size(unfold(x, size(w); pad=2)) == (256, 75, 10)
    @test size(unfold(x, size(w); stride=2)) == (36, 75, 10)
    @test size(unfold(x, size(w); dilation=2)) == (64, 75, 10)
end

@testset "Inverses: spatial_rank=$spatial_rank" for spatial_rank in (1, 2, 3)
    x = rand(rng, repeat([8], spatial_rank)..., 3, 2)
    w = rand(rng, repeat([3], spatial_rank)..., 3, 3)
    cdims = DenseConvDims(x, w; padding=1)
    y = unfold(x, cdims)
    z = fold(y, size(x), cdims)
    divisor = fold(unfold(ones(eltype(x), size(x)...), cdims), size(x), cdims)
    @test isapprox(z ./ divisor, x, rtol=1.0e-7)

    # introduce stride
    cdims = DenseConvDims(x, w; padding=1, stride=2)
    y = unfold(x, cdims)
    z = fold(y, size(x), cdims)
    divisor = fold(unfold(ones(eltype(x), size(x)...), cdims), size(x), cdims)
    @test isapprox(z ./ divisor, x, rtol=1.0e-7)
end

@testset "AutoDiff: spatial_rank=$spatial_rank" for spatial_rank in (1, 2, 3)
    x = rand(rng, repeat([5], spatial_rank)..., 3, 2)
    w = rand(rng, repeat([3], spatial_rank)..., 3, 3)
    cdims = DenseConvDims(x, w)
    gradtest(x -> sum(unfold(x, cdims)), x)
    
    y = unfold(x, cdims)
    gradtest(y -> sum(fold(y, size(x), cdims)), y)
end

