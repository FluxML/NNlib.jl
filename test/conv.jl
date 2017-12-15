using NNlib: conv2d_grad_w, conv2d_grad_x, pool_grad, pool

@testset "conv" begin

    x = reshape(Float64[1:20;], 5, 4, 1, 1)
    w = reshape(Float64[1:4;], 2, 2, 1, 1)

    @test squeeze(conv2d(x, w),(3,4)) == [
        29 79 129;
        39 89 139;
        49 99 149;
        59 109 159.]

    @test squeeze(conv2d(x, w; stride=2),(3,4)) == [
        29 129;
        49 149.]

    @test squeeze(conv2d(x, w; padding=1),(3,4)) == [
        1.0   9.0   29.0   49.0   48.0;
        4.0  29.0   79.0  129.0  115.0;
        7.0  39.0   89.0  139.0  122.0;
        10.0  49.0   99.0  149.0  129.0;
        13.0  59.0  109.0  159.0  136.0;
        10.0  40.0   70.0  100.0   80.0
    ]


    # for gradients, check only size
    # correctness of gradients is cross-checked with CUDNN.jl
    # (it's assumed convolution code won't change often)

    @test size(conv2d_grad_w(x, w, reshape(rand(4,3), 4, 3, 1, 1))) == size(w)
    @test size(conv2d_grad_x(x, w, reshape(rand(4,3), 4, 3, 1, 1))) == size(x)

end


@testset "pool" begin

    x = reshape(Float64[1:20;], 5, 4, 1, 1)

    @test squeeze(pool(x), (3,4)) == [7 17; 9 19]
    @test squeeze(pool(x; stride=2), (3,4)) == [7 17; 9 19]
    @test squeeze(pool(x; padding=1), (3,4)) == [
        1.0  11.0  16.0;
        3.0  13.0  18.0;
        5.0  15.0  20.0;
    ]

    # for gradients, check only size
    # correctness of gradients is cross-checked with CUDNN.jl
    # (it's assumed pooling code won't change often)

    y = pool(x)
    dy = reshape(rand(2,2), 2, 2, 1, 1)
    @test size(pool_grad(x, y, dy)) == size(x)

end
