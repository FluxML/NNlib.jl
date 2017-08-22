@testset "Activation Functions" begin

xs = rand(5,5)

@test all(sum(softmax(xs), 1) .≈ 1)

@test sum(softmax(vec(xs))) ≈ 1

end
