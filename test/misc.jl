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
    @test size(y) == (4, 4, 1, 1)
    @test y_true == y

    x = reshape(1:4*3*27*2, (4,3,27,2))
    y = pixel_shuffle(x, 3)
    @test size(y) == (12, 9, 3, 2)
    x1 = x[:,:,:,[1]]
    x2 = x[:,:,:,[2]]
    y1 = pixel_shuffle(x1, 3)
    y2 = pixel_shuffle(x2, 3)
    @test cat(y1, y2, dims=4) == y
    gradtest(x -> pixel_shuffle(x, 2), rand(3,3,8,2))
end
