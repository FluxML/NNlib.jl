using NNlib: pad_constant, pad_repeat, pad_zeros, pad_reflect, pad_symmetric, pad_circular

@testset "padding constant" begin
    x = rand(2, 2, 2)

    p = NNlib.gen_pad((1,2,3,4,5,6), (1,2,3), 4)
    @test p == ((1, 2), (3, 4), (5, 6), (0, 0))

    @test_throws ArgumentError NNlib.gen_pad((1,2,3,4,5,), (1,2,3), 4)

    p = NNlib.gen_pad((1,3), (1,3), 4)
    @test p == ((1, 1), (0, 0), (3, 3), (0, 0))

    p = NNlib.gen_pad(1, (1,2,3), 4)
    @test p == ((1, 1), (1, 1), (1, 1), (0, 0))

    p = NNlib.gen_pad(3, :, 2)
    @test p == ((3, 3), (3, 3))

    p = NNlib.gen_pad((1,0), 1, 2)
    @test p == ((1,0), (0,0))

    y = pad_constant(x, (3, 2, 4))
    @test size(y) == (8, 6, 10)
    @test y[4:5, 3:4, 5:6] ≈ x
    y[4:5, 3:4, 5:6] .= 0
    @test all(y .== 0)

    @test pad_constant(x, (3, 2, 4)) ≈ pad_zeros(x, (3, 2, 4))
    @test pad_zeros(x, 2) ≈ pad_zeros(x, (2,2,2))

    y = pad_constant(x, (3, 2, 4, 5), 1.2, dims = (1,3))
    @test size(y) == (7, 2, 11)
    @test y[4:5, 1:2, 5:6] ≈ x
    y[4:5, 1:2, 5:6] .= 1.2
    @test all(y .== 1.2)

    @test pad_constant(x, (2,2,2,2), 1.2, dims = (1,3)) ≈
        pad_constant(x, 2, 1.2, dims = (1,3))

    @test pad_constant(x, 1, dims = 1:2) ==
        pad_constant(x, 1, dims = (1,2))

    @test size(pad_constant(x, 1, dims = 1)) == (4,2,2)

    @test all(pad_zeros(randn(2), (1, 2))[[1, 4, 5]] .== 0)

    gradtest(x -> pad_constant(x, 2), rand(2,2,2))
    gradtest(x -> pad_constant(x, (2, 1, 1, 2)), rand(2,2))
    gradtest(x -> pad_constant(x, (2, 1,)), rand(2))
end

@testset "padding repeat" begin
    x = rand(2, 2, 2)

    # y = @inferred pad_repeat(x, (3, 2, 4, 5))
    y = pad_repeat(x, (3, 2, 4, 5))
    @test size(y) == (7, 11, 2)
    @test y[4:5, 5:6, :] ≈ x

    # y = @inferred pad_repeat(x, (3, 2, 4, 5), dims=(1,3))
    y = pad_repeat(x, (3, 2, 4, 5), dims=(1,3))
    @test size(y) == (7, 2, 11)
    @test y[4:5, :, 5:6] ≈ x

    @test pad_repeat(reshape(1:9, 3, 3), (1,2)) ==
        [1  4  7
         1  4  7
         2  5  8
         3  6  9
         3  6  9
         3  6  9]

    @test pad_repeat(reshape(1:9, 3, 3), (2,2), dims=2) ==
        [1  1  1  4  7  7  7
         2  2  2  5  8  8  8
         3  3  3  6  9  9  9]

    @test pad_repeat(x, (2, 2, 2, 2), dims=(1,3)) ≈
        pad_repeat(x, 2, dims=(1,3))

    gradtest(x -> pad_repeat(x, (2,2,2,2)), rand(2,2,2))
end

@testset "padding reflect" begin
    y = pad_reflect(reshape(1:9, 3, 3), (2,2), dims=2)
    @test y == [7  4  1  4  7  4  1
                8  5  2  5  8  5  2
                9  6  3  6  9  6  3]

    y = pad_reflect(reshape(1:9, 3, 3), (2,2,2,2))
    @test y == [9  6  3  6  9  6  3
                8  5  2  5  8  5  2
                7  4  1  4  7  4  1
                8  5  2  5  8  5  2
                9  6  3  6  9  6  3
                8  5  2  5  8  5  2
                7  4  1  4  7  4  1]

    x = rand(4, 4, 4)
    @test pad_reflect(x, (2, 2, 2, 2), dims=(1,3)) ≈
        pad_reflect(x, 2, dims=(1,3))

    # pad_reflect needs larger test input as padding must
    # be strictly less than array size in that dimension
    gradtest(x -> pad_reflect(x, (2,2,2,2)), rand(3,3,3))

    x = reshape(1:9, 3, 3, 1, 1)
    @test NNlib.pad_reflect(x, (1, 0, 1, 0); dims=1:2) == [
        5 2 5 8;
        4 1 4 7;
        5 2 5 8;
        6 3 6 9;;;;]
    @test NNlib.pad_reflect(x, (0, 1, 0, 1); dims=1:2) == [
        1 4 7 4;
        2 5 8 5;
        3 6 9 6;
        2 5 8 5;;;;]
end

@testset "padding symmetric" begin
    y = pad_symmetric(reshape(1:9, 3, 3), (2,2), dims=2)
    @test y == [4  1  1  4  7  7  4
                5  2  2  5  8  8  5
                6  3  3  6  9  9  6]

    y = pad_symmetric(reshape(1:9, 3, 3), (2,2,2,2))
    @test y == [5  2  2  5  8  8  5
                4  1  1  4  7  7  4
                4  1  1  4  7  7  4
                5  2  2  5  8  8  5
                6  3  3  6  9  9  6
                6  3  3  6  9  9  6
                5  2  2  5  8  8  5]

    x = rand(4, 4, 4)
    @test pad_symmetric(x, (2, 2, 2, 2), dims=(1,3)) ≈
        pad_symmetric(x, 2, dims=(1,3))

    gradtest(x -> pad_symmetric(x, (2,2,2,2)), rand(2,2,2))

    x = reshape(1:9, 3, 3, 1, 1)
    @test NNlib.pad_symmetric(x, (1, 0, 1, 0); dims=1:2) == [
        1 1 4 7;
        1 1 4 7;
        2 2 5 8;
        3 3 6 9;;;;]
    @test NNlib.pad_symmetric(x, (0, 1, 0, 1); dims=1:2) == [
        1 4 7 7;
        2 5 8 8;
        3 6 9 9;
        3 6 9 9;;;;]
end

@testset "padding circular" begin
    y = pad_circular(reshape(1:9, 3, 3), (2,2), dims=2)
    @test y == [4  7  1  4  7  1  4
                5  8  2  5  8  2  5
                6  9  3  6  9  3  6]

    y = pad_circular(reshape(1:9, 3, 3), (2,2,2,2))
    @test y == [5  8  2  5  8  2  5
                6  9  3  6  9  3  6
                4  7  1  4  7  1  4
                5  8  2  5  8  2  5
                6  9  3  6  9  3  6
                4  7  1  4  7  1  4
                5  8  2  5  8  2  5]

    x = rand(4, 4, 4)
    @test pad_circular(x, (2, 2, 2, 2), dims=(1,3)) ≈
        pad_circular(x, 2, dims=(1,3))

    gradtest(x -> pad_circular(x, (2,2,2,2)), rand(2,2,2))
end
