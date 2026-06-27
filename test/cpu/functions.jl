@testset "glu" begin
    x = [1 2 3; 4 5 6; 7 8 9; 10 11 12]	 
    @test ceil.(NNlib.glu(x, 1)) == [1 2 3; 4 5 6]
    @test_throws AssertionError NNlib.glu(x, 2)
end

@testset "AutoDiff" begin
    local rng = StableRNG(17)
    @test test_gradients(NNlib.glu, rand(rng, 4, 3))
end

