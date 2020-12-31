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
