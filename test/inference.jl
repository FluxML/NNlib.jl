import NNlib: conv_direct, conv_im2col, channels_in, channels_out

@testset "Conv Inference" begin
    for T in (Float32, Float64)
        impl = [conv, conv_direct, conv_im2col]
        if NNlib.is_nnpack_available() && T == Float32 
            push!(impl, NNlib.conv_nnpack)
        end

        x = rand(T, 10, 10, 3, 2)
        w = rand(T, 3, 3, 3, 1)
        
        for f in impl
            @test @inferred(f(x, w, DenseConvDims(x, w))) isa Array{T,4} 
        end
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
