@testset "bilinear_upsample 2d" begin
    x = reshape(Float32[1. 2.; 3. 4.], (2,2,1,1))
    y_true =   [1//1  5//4    7//4    2//1;
                1//1  5//4    7//4    2//1;
                5//3  23//12  29//12  8//3;
                7//3  31//12  37//12  10//3;
                3//1  13//4   15//4   4//1;
                3//1  13//4   15//4   4//1][:,:,:,:]
   
    y = bilinear_upsample(x, (3, 2))
    @test size(y) == size(y_true)
    @test eltype(y) == Float32
    @test y ≈ y_true

    gradtest(x->bilinear_upsample(x, (3, 2)), x, atol=1e-4)

    if CUDA.has_cuda()
        y = bilinear_upsample(x |> cu, (3, 2))
        @test y isa CuArray 
        @test Array(y) ≈ y_true
        g_gpu = Zygote.gradient(x -> sum(sin.(bilinear_upsample(x, (3, 2))))
                                , x |> cu)[1]
        @test g_gpu isa CuArray
        g_cpu = Zygote.gradient(x -> sum(sin.(bilinear_upsample(x, (3, 2))))
                                , x)[1]
        @test Array(g_cpu) ≈ g_cpu  atol=1e-4
    end
end
