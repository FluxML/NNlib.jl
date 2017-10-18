@testset "Activation Functions" begin

xs = rand(5,5)

@test all(sum(softmax(xs), 1) .≈ 1)

@test sum(softmax(vec(xs))) ≈ 1

@test relu( 0.4,0.3) ≈  0.4
@test relu(-0.4,0.3) ≈ -0.12

end
