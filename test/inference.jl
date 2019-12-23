import NNlib: conv_direct, conv_im2col

@testset "Conv Inference" begin
    x = rand(10, 10, 3, 2)
    w = rand(3, 3, 3, 1)

    impl = [conv, conv_direct, conv_im2col]
    NNlib.is_nnpack_available() && push!(impl, NNlib.conv_nnpack)

    for T in impl
        @test T(x, w, DenseConvDims(x, w)) isa AbstractArray{K,4} where K
    end
end
