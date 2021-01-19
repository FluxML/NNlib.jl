@testset "upsample_nearest, integer scale via reshape" begin
    x = reshape(Float32[1. 2.; 3. 4.], (2,2,1,1))
    @test upsample_nearest(x, (3,3))[1,:] == [1,1,1, 2,2,2]

    y = upsample_nearest(x, (2,3))
    @test size(y) == (4,6,1,1)
    ∇upsample_nearest(y, (2,3)) == [6 12; 18 24]

    gradtest(x -> upsample_nearest(x, (2,3)), rand(2,2,1,1), check_rrule=false)

    @test_throws ArgumentError ∇upsample_nearest(y, (2,4))
    @test_throws ArgumentError upsample_nearest(x, (1,2,3,4,5))
end

@testset "upsample_bilinear 2d" begin
    x = reshape(Float32[1. 2.; 3. 4.], (2,2,1,1))
    y_true =   [1//1  5//4    7//4    2//1;
                1//1  5//4    7//4    2//1;
                5//3  23//12  29//12  8//3;
                7//3  31//12  37//12  10//3;
                3//1  13//4   15//4   4//1;
                3//1  13//4   15//4   4//1][:,:,:,:]
   
    y = upsample_bilinear(x, (3, 2))
    @test size(y) == size(y_true)
    @test eltype(y) == Float32
    @test y ≈ y_true

    gradtest(x->upsample_bilinear(x, (3, 2)), x, atol=1e-4)

    if CUDA.has_cuda()
        y = upsample_bilinear(x |> cu, (3, 2))
        @test y isa CuArray 
        @test Array(y) ≈ y_true
        g_gpu = Zygote.gradient(x -> sum(sin.(upsample_bilinear(x, (3, 2))))
                                , x |> cu)[1]
        @test g_gpu isa CuArray
        g_cpu = Zygote.gradient(x -> sum(sin.(upsample_bilinear(x, (3, 2))))
                                , x)[1]
        @test Array(g_cpu) ≈ g_cpu  atol=1e-4
    end
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
