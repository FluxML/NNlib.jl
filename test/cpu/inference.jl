import NNlib: conv_direct, conv_im2col, channels_in, channels_out

@testset "Conv Inference" begin
    for T in (Float32, Float64)
        impl = [conv, conv_direct, conv_im2col]

        x = rand(T, 10, 10, 3, 2)
        w = rand(T, 3, 3, 3, 1)
        cdims = DenseConvDims(x, w)
        dy = conv(x, w, cdims)

        for f in impl
            @test @inferred(f(x, w, cdims)) isa Array{T,4}
        end

        @test @inferred(conv(x, w)) isa Array{T,4}
        @test @inferred(∇conv_filter(x, dy, cdims)) isa Array{T,4}
        @test @inferred(∇conv_data(dy, w, cdims)) isa Array{T,4}
    end
end

@testset "DepthwiseConv Inference" begin
    for T in (Float32, Float64)
        x = rand(T, 10, 10, 3, 2)
        w = rand(T, 3, 3, 3, 3)
        cdims = DepthwiseConvDims(x, w)
        dy = depthwiseconv(x, w)

        @test @inferred(depthwiseconv(x, w)) isa Array{T,4}
        @test @inferred(∇depthwiseconv_filter(x, dy, cdims)) isa Array{T,4}
        @test @inferred(∇depthwiseconv_data(dy, w, cdims)) isa Array{T,4}
    end
end

@testset "DenseConvDims Inference" begin
    # this needs to be in a function to trigger inference problems
    function channels_in_test(w::AbstractArray)
        cdims = DenseConvDims((1,1,1,1), size(w))
        channels_in(cdims)
    end

    # this needs to be in a function to trigger inference problems
    function channels_out_test(w::AbstractArray)
        cdims = DenseConvDims((1,1,1,1), size(w))
        channels_out(cdims)
    end

    w = rand(Float32, 1, 1, 1, 1)
    @test @inferred(channels_in_test(w)) isa Int
    @test @inferred(channels_out_test(w)) isa Int
end

@testset "Pooling inference" begin
    for T in (Float32, Float64)
        x = rand(T, 10, 10, 3, 2)
        pdims = PoolDims(x, 3)

        y_maxpool = NNlib.maxpool(x, pdims)
        y_meanpool = NNlib.meanpool(x, pdims)
        dy = ones(T, size(y_maxpool)...)

        @test @inferred(NNlib.maxpool(x, pdims)) isa Array{T, 4}
        @test @inferred(NNlib.meanpool(x, pdims)) isa Array{T, 4}
        @test @inferred(NNlib.∇maxpool(dy, y_maxpool, x, pdims)) isa Array{T, 4}
        @test @inferred(NNlib.∇maxpool(dy, y_meanpool, x, pdims)) isa Array{T, 4}
    end
end
