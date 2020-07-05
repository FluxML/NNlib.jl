import NNlib: conv_direct, conv_im2col

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
