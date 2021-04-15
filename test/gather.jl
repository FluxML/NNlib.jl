using NNlib: gather, gather!

@testset "gather scalar index" begin 
    T = Float32
    
    ## 1d src, 2d index of ints -> 2d output
    src = T[3, 4, 5, 6, 7]
    index = [1 2 3 4;
            4 2 1 3;
            3 5 5 3]
    output = T[3 4 5 6;
              6 4 3 5;
              5 7 7 5]
    
    y = gather(src, index)
    @test y isa Array{T,2}
    @test size(y) == size(index)
    @test y == output
    @test gather!(T.(zero(index)), src, index) == output
    @test_throws ArgumentError gather!(zeros(T, 3, 5), src, index)
    
    index2 = [1 2 3 4;
              4 2 1 3;
              3 6 5 3]
    @test_throws BoundsError gather!(T.(zero(index)), src, index2)

    ## 1d src, 3d index of ints -> 3d output
    src = T[3, 4, 5, 6, 7]
    index = [1 2 3 4;
            4 2 1 3;
            3 5 5 3][:,:,1:1]
    output = T[3 4 5 6;
              6 4 3 5;
              5 7 7 5][:,:,1:1]

    y = gather(src, index)
    @test y isa Array{T,3}
    @test size(y) == size(index)
    @test y == output


    ## 2d src, 2d index of ints -> 3d output
    src = T[3 5 7 
            4 6 8]
    index = [1 2 3;
            2 2 1;
            3 1 3]

    output = zeros(T, 2, 3, 3)

    output[:,:,1] = [3 5 7
                    4 6 8]

    output[:,:,2] = [5 5 3
                    6 6 4]
        
    output[:,:,3] = [7 3 7
                    8 4 8]
              
    y = gather(src, index)
    M = NNlib.typelength(eltype(index))
    Nsrc = ndims(src)
    @test y isa Array{T,3}
    @test size(y) == (size(src)[1:Nsrc-M]..., size(index)...) 
    @test y == output
end

@testset "gather tuple index" begin 
    T = Float32
    
    ## 2d src, 1d index of 2-tuples -> 1d output
    src = T[3 5 7 
            4 6 8]

    index = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)]

    output = T[3, 5, 7, 4, 6, 8]

    y = gather(src, index)
    M = NNlib.typelength(eltype(index))
    Nsrc = ndims(src)
    @test y isa Array{T,1}
    @test size(y) == (size(src)[1:Nsrc-M]..., size(index)...) 
    @test y == output

    ## 3d src, 2d index of 2-tuples -> 3d output
    n1, nsrc, nidx = 2, 3, 6
    src = rand(Float32, n1, nsrc, nsrc)
    index = [(rand(1:nsrc), rand(1:nsrc)) for i=1:nidx, j=1:nidx]
    
    y = gather(src, index)
    M = NNlib.typelength(eltype(index))
    Nsrc = ndims(src)
    @test y isa Array{T,3}
    @test size(y) == (size(src)[1:Nsrc-M]..., size(index)...) 
end

@testset "gather cartesian index" begin 
    T = Float32
    
    ## 2d src, 1d index of 2-tuples -> 1d output
    src = T[3 5 7 
            4 6 8]

    index = CartesianIndex.([(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)])

    output = T[3, 5, 7, 4, 6, 8]

    y = gather(src, index)
    M = NNlib.typelength(eltype(index))
    Nsrc = ndims(src)
    @test y isa Array{T,1}
    @test size(y) == (size(src)[1:Nsrc-M]..., size(index)...) 
    @test y == output

    ## 3d src, 2d index of 2-tuples -> 3d output
    n1, nsrc, nidx = 2, 3, 6
    src = rand(Float32, n1, nsrc, nsrc)
    index = [CartesianIndex((rand(1:nsrc), rand(1:nsrc))) for i=1:nidx, j=1:nidx]
    
    y = gather(src, index)
    M = NNlib.typelength(eltype(index))
    Nsrc = ndims(src)
    @test y isa Array{T,3}
    @test size(y) == (size(src)[1:Nsrc-M]..., size(index)...) 
end
