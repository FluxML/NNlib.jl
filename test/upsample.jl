@testset "upsample_nearest, integer scale via reshape" begin
    x = reshape(Float32[1. 2.; 3. 4.], (2,2,1,1))
    @test upsample_nearest(x, (3,3))[1,:] == [1,1,1, 2,2,2]

    y = upsample_nearest(x, (2,3))
    @test size(y) == (4,6,1,1)
    ∇upsample_nearest(y, (2,3)) == [6 12; 18 24]

    gradtest(x -> upsample_nearest(x, (2,3)), rand(2,2,1,1))

    y2 = upsample_nearest(x, size=(4,6))
    @test y ≈ y2
    gradtest(x -> upsample_nearest(x, size=(4,6)), rand(2,2,1,1))

    @test_throws ArgumentError ∇upsample_nearest(y, (2,4))
    @test_throws ArgumentError upsample_nearest(x, (1,2,3,4,5))
    @test_throws ArgumentError upsample_nearest(x, size=(3,4))
end

@testset "Linear upsampling (1D)" begin
    x = Float64[1,2,3,4]
    x = hcat(x,x,x)[:,:,:]

    y = collect(1:1//3:4)
    y = hcat(y,y,y)[:,:,:]
    yF64 = Float64.(y)

    @test y ≈ upsample_linear(x, 2.5)
    @test y ≈ upsample_linear(x; size=10)
    gradtest(x->upsample_linear(x, 2.5), x)
end

@testset "Bilinear upsampling (2D)" begin
    x = Float32[1 2; 3 4][:,:,:,:]
    x = cat(x,x; dims=3)
    x = cat(x,x; dims=4)

    # this output matches the one of pytorch v1.5.0
    # nn.UpsamplingBilinear2d(scale_factor=(3,2), align_corners=True)
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
    o = ones(Float32,6,4,2,1)
    grad_true = 6*ones(Float32,2,2,2,1)
    @test ∇upsample_bilinear(o; size = (2,2)) ≈ grad_true

    y_true_2 = Rational{Int}[1//1  5//4  6//4  7//4 2//1;
                             3//2  7//4  8//4  9//4 5//2;
                             4//2  9//4 10//4 11//4 6//2;
                             5//2 11//4 12//4 13//4 7//2;
                             3//1 13//4 14//4 15//4 4//1][:,:,:,:]

     # check for real-valued single-number argument and type stability for rationals
    upsample_bilinear(x, 2.5) == y_true_2

    # check Integer support for forward pass
    # grads are always assumed to be floats, so no extension there
    x = UInt8[1 3; 3 5][:,:,:,:]
    y_true_int = UInt8[1 2 3; 2 3 4; 3 4 5][:,:,:,:]
    y = upsample_bilinear(x, 1.5)

    @test eltype(y) == UInt8
    @test y == y_true_int
end

@testset "Trilinear upsampling (3D)" begin
    # Layout: WHDCN, where D is depth
    # we generate data which is constant along W & H and differs in D
    # then we upsample along all dimensions
    x = ones(Float32, 3,3,3,1,1)
    x[:,:,1,:,:] .= 1.
    x[:,:,2,:,:] .= 2.
    x[:,:,3,:,:] .= 3.

    y_true = ones(Float32, 5,5,5,1,1)
    y_true[:,:,1,:,:] .= 1.
    y_true[:,:,2,:,:] .= 1.5
    y_true[:,:,3,:,:] .= 2.
    y_true[:,:,4,:,:] .= 2.5
    y_true[:,:,5,:,:] .= 3.

    y = upsample_trilinear(x; size=(5,5,5))

    @test size(y) == size(y_true)
    @test eltype(y) == Float32
    @test collect(y) ≈ collect(y_true)

    # this test only works when align_corners=false (not present for CPU yet)
    # o = ones(Float32,8,8,8,1,1)
    # grad_true = 8*ones(Float32,4,4,4,1,1)
    # @test ∇upsample_trilinear(o; size=(4,4,4)) ≈ grad_true

    x = Float64.(x)
    gradtest(x -> upsample_trilinear(x, (2,2,2)), x)
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

@testset "Complex-valued upsample" begin
    for (d, method) in zip([1, 2, 3], [upsample_linear, upsample_bilinear, upsample_trilinear])
        for (k, interp) in zip((2, ntuple(_ -> 2,  d)), [method, upsample_nearest])
            x = randn(Complex{Float32}, (4,8,12)[1:d]..., 1, 1)

            upsize = (8, 16, 24)[1:d]
            xup = interp(x, k)
            @test size(xup)[1:d] == upsize
            @test real(xup) == interp(real(x), k)
            @test imag(xup) == interp(imag(x), k)

            upsize = (8,24,48)[1:d]
            xup = interp(x; size=upsize)
            @test size(xup)[1:d] == upsize
            @test real(xup) == interp(real(x), size=upsize)
            @test imag(xup) == interp(imag(x), size=upsize)
        end
    end
end
