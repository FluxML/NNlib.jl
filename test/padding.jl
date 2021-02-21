@testset "padding constant" begin
  x = rand(2, 2, 2)  
  
  y = @inferred pad_constant(x, (3, 2, 4, 5))
  @test size(y) == (7, 11, 2)
  @test y[4:5, 5:6, :] ≈ x
  y[4:5, 5:6, :] .= 0
  @test all(y .== 0)

  @test pad_constant(x, (3, 2, 4, 5)) ≈ pad_zeros(x, (3, 2, 4, 5))
  @test pad_zeros(x, 2) ≈ pad_zeros(x, (2,2)) 
  
  y = @inferred pad_constant(x, (3, 2, 4, 5), 1.2, dims=(1,3))
  @test size(y) == (7, 2, 11)
  @test y[4:5, :, 5:6] ≈ x
  y[4:5, :, 5:6] .= 1.2
  @test all(y .== 1.2)
  
  @test pad_constant(x, (2, 2, 2, 2), 1.2, dims=(1,3)) ≈
          pad_constant(x, 2, 1.2, dims=(1,3))
  
  gradtest(x -> pad_constant(x, (2,2,2,2)), rand(2,2,2))
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
  @test y ==  [ 7  4  1  4  7  4  1
                8  5  2  5  8  5  2
                9  6  3  6  9  6  3]

  y = pad_reflect(reshape(1:9, 3, 3), (2,2,2,2))
  @test y ==   [9  6  3  6  9  6  3
                8  5  2  5  8  5  2
                7  4  1  4  7  4  1
                8  5  2  5  8  5  2
                9  6  3  6  9  6  3
                8  5  2  5  8  5  2
                7  4  1  4  7  4  1]

  x = rand(4, 4, 4)  
  
  @test pad_reflect(x, (2, 2, 2, 2), dims=(1,3)) ≈
          pad_reflect(x, 2, dims=(1,3))
      
  gradtest(x -> pad_repeat(x, (2,2,2,2)), rand(2,2,2))
end
