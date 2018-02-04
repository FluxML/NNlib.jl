using NNlib: conv2d_grad_w, conv2d_grad_x, pool2d_grad, pool2d,
             conv3d_grad_w, conv3d_grad_x, pool3d_grad, pool3d

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


@testset "pool2d" begin

    x = reshape(Float64[1:20;], 5, 4, 1, 1)

    @test squeeze(pool2d(x), (3,4)) == [7 17; 9 19]
    @test squeeze(pool2d(x; stride=2), (3,4)) == [7 17; 9 19]
    @test squeeze(pool2d(x; padding=1), (3,4)) == [
        1.0  11.0  16.0;
        3.0  13.0  18.0;
        5.0  15.0  20.0;
    ]

    # for gradients, check only size
    # correctness of gradients is cross-checked with CUDNN.jl
    # (it's assumed pooling code won't change often)

    y = pool2d(x)
    dy = reshape(rand(2,2), 2, 2, 1, 1)
    @test size(pool2d_grad(x, y, dy)) == size(x)

end


@testset "conv3d" begin

    x = reshape(Float64[1:60;], 5, 4, 3, 1, 1)
    w = reshape(Float64[1:8;], 2, 2, 2, 1, 1)
    res = zeros(4,3,2)
    res[:, :, 1] = [
        322.0  502.0  682.0;
        358.0  538.0  718.0;
        394.0  574.0  754.0;
        430.0  610.0  790.0]
    res[:, :, 2] = [
        1042.0  1222.0  1402.0;
        1078.0  1258.0  1438.0;
        1114.0  1294.0  1474.0;
        1150.0  1330.0  1510.0]
    @test squeeze(conv3d(x, w),(4,5)) == res

    @test squeeze(conv3d(x, w; stride=2),(3,4,5)) == [
        322.0 682.0;
        394.0 754.0]

    res = zeros(6,5,4)
    res[:, :, 1] = [
        1.0   9.0   29.0   49.0   48.0;
        4.0  29.0   79.0  129.0  115.0;
        7.0  39.0   89.0  139.0  122.0;
        10.0  49.0   99.0  149.0  129.0;
        13.0  59.0  109.0  159.0  136.0;
        10.0  40.0   70.0  100.0   80.0]
    res[:, :, 2] = [
        26.0  126.0  206.0  286.0  220.0;
        80.0  322.0  502.0  682.0  502.0;
        94.0  358.0  538.0  718.0  524.0;
        108.0  394.0  574.0  754.0  546.0;
        122.0  430.0  610.0  790.0  568.0;
        80.0  260.0  360.0  460.0  320.0]
    res[:, :, 3] = [
        146.0   446.0   526.0   606.0   420.0;
        360.0  1042.0  1222.0  1402.0   942.0;
        374.0  1078.0  1258.0  1438.0   964.0;
        388.0  1114.0  1294.0  1474.0   986.0;
        402.0  1150.0  1330.0  1510.0  1008.0;
        240.0   660.0   760.0   860.0   560.0]
    res[:, :, 4] = [
        205.0   517.0   577.0   637.0  392.0;
        456.0  1133.0  1263.0  1393.0  847.0;
        467.0  1159.0  1289.0  1419.0  862.0;
        478.0  1185.0  1315.0  1445.0  877.0;
        489.0  1211.0  1341.0  1471.0  892.0;
        270.0   660.0   730.0   800.0  480.0]
    @test squeeze(conv3d(x, w; padding=1),(4,5)) == res


    # for gradients, check only size
    # correctness of gradients is cross-checked with CUDNN.jl
    # (it's assumed convolution code won't change often)

    @test size(conv3d_grad_w(x, w, reshape(rand(4,3,2), 4, 3, 2, 1, 1))) == size(w)
    @test size(conv3d_grad_x(x, w, reshape(rand(4,3,2), 4, 3, 2, 1, 1))) == size(x)

end


@testset "pool3d" begin

    x = reshape(Float64[1:60;], 5, 4, 3, 1, 1)

    @test squeeze(pool3d(x), (3,4,5)) == [27 37; 29 39.]
    @test squeeze(pool3d(x; stride=2), (3,4,5)) == [27 37; 29 39.]
    res = zeros(3,3,2)
    res[:, :, 1] = [
        1.0  11.0  16.0;
        3.0  13.0  18.0;
        5.0  15.0  20.0]
    res[:, :, 2] = [
        41.0  51.0  56.0;
        43.0  53.0  58.0;
        45.0  55.0  60.0]
    @test squeeze(pool3d(x; padding=1), (4,5)) == res

    # for gradients, check only size
    # correctness of gradients is cross-checked with CUDNN.jl
    # (it's assumed pooling code won't change often)

    y = pool3d(x)
    dy = reshape(rand(2,2), 2, 2, 1, 1, 1)
    @test size(pool3d_grad(x, y, dy)) == size(x)

end
