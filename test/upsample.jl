@testset "upsample_bilinear 2d" begin
    x = Float32[1 2; 3 4][:,:,:,:]
    x = cat(x,x; dims=3)
    x = cat(x,x; dims=4)

    # this output matches the one of pytorch v1.5.0
    # nn.UpsamplingBilinear2d(scale_factor=(3,2))
    # for above x
    y_true = Float32[ 1//1  4//3   5//3   2//1;
                      7//5 26//15 31//15 12//5;
                      9//5 32//15 37//15 14//5;
                     11//5 38//15 43//15 16//5;
                     13//5 44//15 49//15 18//5;
                      3//1 10//3  11//3   4//1][:,:,:,:]
    y_true = cat(y_true,y_true; dims=3)
    y_true = cat(y_true,y_true; dims=4)

    y = upsample_bilinear(x, (3, 2))
    @test size(y) == size(y_true)
    @test eltype(y) == Float32
    @test y ≈ y_true

    gradtest(x->upsample_bilinear(x, (3, 2)), x, atol=1e-3) # works to higher precision for Float64

    # additional grad check, also compliant with pytorch
    o = ones(Float32,6,4,1,1)
    grad_true = Float32[6 6; 6 6][:,:,:,:]
    @test ∇upsample_bilinear(o, (3,2)) ≈ grad_true

    # this test can be performed again, as soon as the corresponding CUDA functionality is merged

    # if CUDA.has_cuda()
    #     y = upsample_bilinear(x |> cu, (3, 2))
    #     @test y isa CuArray
    #     @test Array(y) ≈ y_true
    #     g_gpu = Zygote.gradient(x -> sum(sin.(upsample_bilinear(x, (3, 2))))
    #                             , x |> cu)[1]
    #     @test g_gpu isa CuArray
    #     g_cpu = Zygote.gradient(x -> sum(sin.(upsample_bilinear(x, (3, 2))))
    #                             , x)[1]
    #     @test Array(g_cpu) ≈ g_cpu  atol=1e-4
    # end
end

@testset "pixel_shuffle" begin
    x = reshape(1:16, (2, 2, 4, 1))
    # [:, :, 1, 1] =
    #     1  3
    #     2  4
    # [:, :, 2, 1] =
    #     5  7
    #     6  8
    # [:, :, 3, 1] =
    #     9  11
    #     10  12
    # [:, :, 4, 1] =
    #     13  15
    #     14  16

    y_true = [1  9 3 11
              5 13 7 15
              2 10 4 12
              6 14 8 16][:,:,:,:]

    y = pixel_shuffle(x, 2)
    @test size(y) == size(y_true)
    @test y_true == y

    x = reshape(1:32, (2, 2, 8, 1))
    y_true = zeros(Int, 4, 4, 2, 1)
    y_true[:,:,1,1] .= [ 1   9  3  11
                         5  13  7  15
                         2  10  4  12
                         6  14  8  16 ]

    y_true[:,:,2,1] .= [ 17  25  19  27
                         21  29  23  31
                         18  26  20  28
                         22  30  24  32]

    y = pixel_shuffle(x, 2)
    @test size(y) == size(y_true)
    @test y_true == y

    x = reshape(1:4*3*27*2, (4,3,27,2))
    y = pixel_shuffle(x, 3)
    @test size(y) == (12, 9, 3, 2)
    # batch dimension is preserved
    x1 = x[:,:,:,[1]]
    x2 = x[:,:,:,[2]]
    y1 = pixel_shuffle(x1, 3)
    y2 = pixel_shuffle(x2, 3)
    @test cat(y1, y2, dims=4) == y

    for d in [1, 2, 3]
        r = rand(1:5)
        n = rand(1:5)
        c = rand(1:5)
        insize = rand(1:5, d)
        x = rand(insize..., r^d*c, n)

        y = pixel_shuffle(x, r)
        @test size(y) == ((r .* insize)..., c, n)

        gradtest(x -> pixel_shuffle(x, r), x)
    end
end
