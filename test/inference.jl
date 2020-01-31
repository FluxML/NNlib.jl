@testset "Conv Inference" begin
    x = rand(Float32, 10, 10, 3, 2)
    w = rand(Float32, 3, 3, 3, 1)

    impl = [conv, NNlib.conv_direct, NNlib.conv_im2col]
    NNlib.is_nnpack_available() && push!(impl, NNlib.conv_nnpack)

    for T in impl
        @test T(x, w, DenseConvDims(x, w)) isa AbstractArray{eltype(x),4}
    end
end
