@testset "Activation Functions" begin

xs = rand(5,5)

@test all(sum(softmax(xs), 1) .≈ 1)

@test sum(softmax(vec(xs))) ≈ 1

@testset "elu" begin
    @test elu(42) == 42
    @test elu(42.) == 42.

    @test elu(-4) ≈ (exp(-4) - 1)
end

@test leakyrelu( 0.4,0.3) ≈  0.4
@test leakyrelu(-0.4,0.3) ≈ -0.12

end
