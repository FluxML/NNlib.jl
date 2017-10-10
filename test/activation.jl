@testset "Activation Functions" begin

xs = rand(5,5)

@test all(sum(softmax(xs), 1) .≈ 1)

@test sum(softmax(vec(xs))) ≈ 1
  
@test leakyrelu(0.3)( 0.4) ≈ 0.4
@test leakyrelu(0.3)(-0.4) ≈ -0.12

end
