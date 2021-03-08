@testset "Bilinear upsampling" begin
    x = Float32[1 2; 3 4][:,:,:,:]
    x = cat(x,x; dims=3)
    x = cat(x,x; dims=4)
    xgpu = cu(x)
  
    y_true = Float32[ 1//1  4//3   5//3   2//1;
            7//5 26//15 31//15 12//5;
            9//5 32//15 37//15 14//5;
           11//5 38//15 43//15 16//5;
           13//5 44//15 49//15 18//5;
            3//1 10//3  11//3   4//1]
    y_true = cat(y_true,y_true; dims=3)
    y_true = cat(y_true,y_true; dims=4)
    y_true_gpu = cu(y_true)
  
    y = upsample_bilinear(xgpu, (3,2))
    @test size(y) == size(y_true_gpu)
    @test eltype(y) == Float32
    @test collect(y) ≈ collect(y_true_gpu)

    o = CUDA.ones(Float32,6,4,2,1)
    grad_true = 6*CUDA.ones(Float32,2,2,2,1)
    @test ∇upsample_bilinear(o; size=(2,2)) ≈ grad_true

    gputest(x -> upsample_bilinear(x, (3, 2)), x, atol=1e-5)
end
