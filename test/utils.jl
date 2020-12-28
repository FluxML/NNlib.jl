@testset "least_dims" begin
    ind1 = [1,2,3,4,5,6]
    @test NNlib.least_dims(ind1) == (6,)
    ind2 = [(3,4,5), (1,2,3), (2,3,9)]
    @test NNlib.least_dims(ind2) == (3,4,9)
    ind3 = [(3,4,5) (1,2,3) (2,3,9);
            (4,6,2) (5,3,2) (4,4,4)]
    @test NNlib.least_dims(ind3) == (5,6,9)
end
