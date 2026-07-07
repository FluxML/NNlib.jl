using NNlib: DenseConvDims, DepthwiseConvDims, PoolDims, conv, conv!,
             ∇conv_data, ∇conv_data!, ∇conv_filter, ∇conv_filter!,
             depthwiseconv, depthwiseconv!, ∇depthwiseconv_data,
             ∇depthwiseconv_data!, ∇depthwiseconv_filter,
             ∇depthwiseconv_filter!, maxpool, maxpool!, ∇maxpool,
             ∇maxpool!, meanpool, meanpool!, ∇meanpool, ∇meanpool!

@testset "conv forward" begin
    for T in (Float16, Float32)
        configs = (
            (randn(T, 7, 6, 2, 3), randn(T, 3, 2, 2, 4), 1),
            (randn(T, 7, 6, 4, 3), randn(T, 3, 2, 2, 6), 2),
        )

        for (x, w, groups) in configs
            cdims = DenseConvDims(size(x), size(w); stride=(2, 1),
                                  padding=(1, 1, 0, 1), groups)

            y = conv(x, w, cdims)
            gy = conv(DEVICE(x), DEVICE(w), cdims)
            @test Array(gy) ≈ y

            gout = similar(DEVICE(x), size(y))
            conv!(gout, DEVICE(x), DEVICE(w), cdims)
            @test Array(gout) ≈ y
        end
    end
end

@testset "conv backward" begin
    for T in (Float16, Float32)
        configs = (
            (randn(T, 7, 6, 2, 3), randn(T, 3, 2, 2, 4), 1),
            (randn(T, 7, 6, 4, 3), randn(T, 3, 2, 2, 6), 2),
        )

        for (x, w, groups) in configs
            cdims = DenseConvDims(size(x), size(w); stride=(2, 1),
                                  padding=(1, 1, 0, 1), groups)
            dy = randn(T, size(conv(x, w, cdims)))

            dx = ∇conv_data(dy, w, cdims)
            gdx = ∇conv_data(DEVICE(dy), DEVICE(w), cdims)
            @test Array(gdx) ≈ dx

            gout = similar(DEVICE(x))
            ∇conv_data!(gout, DEVICE(dy), DEVICE(w), cdims)
            @test Array(gout) ≈ dx

            dw = ∇conv_filter(x, dy, cdims)
            gdw = ∇conv_filter(DEVICE(x), DEVICE(dy), cdims)
            @test Array(gdw) ≈ dw

            gout = similar(DEVICE(w))
            ∇conv_filter!(gout, DEVICE(x), DEVICE(dy), cdims)
            @test Array(gout) ≈ dw
        end
    end
end

@testset "depthwise conv" begin
    for T in (Float16, Float32)
        x = randn(T, 7, 6, 4, 3)
        w = randn(T, 3, 2, 2, 4)
        cdims = DepthwiseConvDims(size(x), size(w); stride=(2, 1),
                                  padding=(1, 1, 0, 1))

        y = depthwiseconv(x, w, cdims)
        gy = depthwiseconv(DEVICE(x), DEVICE(w), cdims)
        @test Array(gy) ≈ y

        gout = similar(DEVICE(x), size(y))
        depthwiseconv!(gout, DEVICE(x), DEVICE(w), cdims)
        @test Array(gout) ≈ y

        dy = randn(T, size(y))
        dx = ∇depthwiseconv_data(dy, w, cdims)
        gdx = ∇depthwiseconv_data(DEVICE(dy), DEVICE(w), cdims)
        @test Array(gdx) ≈ dx

        gout = similar(DEVICE(x))
        ∇depthwiseconv_data!(gout, DEVICE(dy), DEVICE(w), cdims)
        @test Array(gout) ≈ dx

        dw = ∇depthwiseconv_filter(x, dy, cdims)
        gdw = ∇depthwiseconv_filter(DEVICE(x), DEVICE(dy), cdims)
        @test Array(gdw) ≈ dw

        gout = similar(DEVICE(w))
        ∇depthwiseconv_filter!(gout, DEVICE(x), DEVICE(dy), cdims)
        @test Array(gout) ≈ dw
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

@testset "pool backward" begin
    for T in (Float16, Float32)
        x = T[1 + mod(w - 1, 3) + 3 * mod(h - 1, 2)
              for w in 1:7, h in 1:6, c in 1:2, n in 1:3]

        pdims = PoolDims(size(x), (3, 2); stride=(2, 1))
        y = maxpool(x, pdims)
        dy = randn(T, size(y))
        dx = ∇maxpool(dy, y, x, pdims)
        gdx = ∇maxpool(DEVICE(dy), DEVICE(y), DEVICE(x), pdims)
        @test Array(gdx) ≈ dx

        gout = similar(DEVICE(x))
        ∇maxpool!(gout, DEVICE(dy), DEVICE(y), DEVICE(x), pdims)
        @test Array(gout) ≈ dx

        pdims = PoolDims(size(x), (3, 2); stride=(2, 1), padding=(1, 1, 0, 1))
        y = meanpool(x, pdims; count_include_pad=true)
        dy = randn(T, size(y))
        dx = ∇meanpool(dy, y, x, pdims; count_include_pad=true)
        gdx = ∇meanpool(DEVICE(dy), DEVICE(y), DEVICE(x), pdims; count_include_pad=true)
        @test Array(gdx) ≈ dx

        gout = similar(DEVICE(x))
        ∇meanpool!(gout, DEVICE(dy), DEVICE(y), DEVICE(x), pdims; count_include_pad=true)
        @test Array(gout) ≈ dx

        y = meanpool(x, pdims; count_include_pad=false)
        dy = randn(T, size(y))
        dx = ∇meanpool(dy, y, x, pdims; count_include_pad=false)
        gdx = ∇meanpool(DEVICE(dy), DEVICE(y), DEVICE(x), pdims; count_include_pad=false)
        @test Array(gdx) ≈ dx
    end
end
