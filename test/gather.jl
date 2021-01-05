T = Float32
input = T[3, 4, 5, 6, 7]
index = [1 2 3 4;
         4 2 1 3;
         3 5 5 3]
output = T[3 4 5 6;
          6 4 3 5;
          5 7 7 5]
@test gather(input, index, dims=0) == output
@test gather!(T.(zero(index)), input, index, dims=0) == output
@test_throws ArgumentError gather!(T.(zeros(3, 5)), input, index, dims=0)

index2 = [1 2 3 4;
          4 2 1 3;
          3 6 5 3]
@test_throws BoundsError gather!(T.(zero(index)), input, index2, dims=0)
