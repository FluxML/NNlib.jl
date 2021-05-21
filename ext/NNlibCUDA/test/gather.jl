@testset "gather" begin 
    T = Float32
    CT = CuArray{Float32}
    
    ## 1d src, 2d index of ints -> 2d output
    src = CT([3, 4, 5, 6, 7])
    index = cu([1 2 3 4;
                4 2 1 3;
                3 5 5 3])
    output = CT([3 4 5 6;
                6 4 3 5;
                5 7 7 5])
    
    y = NNlib.gather(src, index)
    @test y isa CuArray{Float32,2}
    @test size(y) == size(index)
    gputest(src -> NNlib.gather(src, index), src, checkgrad=false)
    @test NNlib.gather!(CUDA.zeros(T, size(index)...), src, index) == output
    @test_throws ArgumentError NNlib.gather!(zeros(T, 3, 5), src, index)

    ## 1d src, 3d index of ints -> 3d output
    src = CT([3, 4, 5, 6, 7])
    index = cu([1 2 3 4;
                4 2 1 3;
                3 5 5 3][:,:,1:1])
    output = CT([3 4 5 6;
                6 4 3 5;
                5 7 7 5][:,:,1:1])

    y = NNlib.gather(src, index)
    @test y isa CuArray{Float32,3}
    @test size(y) == size(index)
    gputest(src -> NNlib.gather(src, index), src, checkgrad=false)


    ## 2d src, 2d index of ints -> 3d output
    src = CT([3 5 7 
             4 6 8])
    index = cu([1 2 3;
                2 2 1;
                3 1 3])

    output = zeros(T, 2, 3, 3)

    output[:,:,1] = [3 5 7
                    4 6 8]

    output[:,:,2] = [5 5 3
                    6 6 4]
        
    output[:,:,3] = [7 3 7
                    8 4 8]
              
    y = NNlib.gather(src, index)
    M = NNlib.typelength(eltype(index))
    Nsrc = ndims(src)
    @test y isa CuArray{Float32,3}
    @test size(y) == (size(src)[1:Nsrc-M]..., size(index)...) 
    gputest(src -> NNlib.gather(src, index), src, checkgrad=false)
end
