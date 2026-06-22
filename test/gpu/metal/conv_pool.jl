using NNlib: DenseConvDims, PoolDims, conv, conv!, maxpool, maxpool!, meanpool, meanpool!

@testset "conv forward" begin
    for T in (Float16, Float32)
        x = randn(T, 7, 6, 2, 3)
        w = randn(T, 3, 2, 2, 4)
        cdims = DenseConvDims(size(x), size(w); stride=(2, 1), padding=(1, 1, 0, 1))

        y = conv(x, w, cdims)
        gy = conv(DEVICE(x), DEVICE(w), cdims)
        @test Array(gy) ≈ y

        gout = similar(DEVICE(x), size(y))
        conv!(gout, DEVICE(x), DEVICE(w), cdims)
        @test Array(gout) ≈ y
    end
end

@testset "pool forward" begin
    for T in (Float16, Float32)
        x = randn(T, 7, 6, 2, 3)

        pdims = PoolDims(size(x), (3, 2); stride=(2, 1))
        y = maxpool(x, pdims)
        gy = maxpool(DEVICE(x), pdims)
        @test Array(gy) ≈ y

        gout = similar(DEVICE(x), size(y))
        maxpool!(gout, DEVICE(x), pdims)
        @test Array(gout) ≈ y

        pdims = PoolDims(size(x), (3, 2); stride=(2, 1), padding=(1, 1, 0, 1))
        y = meanpool(x, pdims; count_include_pad=true)
        gy = meanpool(DEVICE(x), pdims; count_include_pad=true)
        @test Array(gy) ≈ y

        gout = similar(DEVICE(x), size(y))
        meanpool!(gout, DEVICE(x), pdims; count_include_pad=true)
        @test Array(gout) ≈ y
    end
end
