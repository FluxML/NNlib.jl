@testset "within_gradient" begin
    @test NNlib.within_gradient([1.0]) === false
    @test gradient(x -> NNlib.within_gradient(x) * x, 2.0) == (1.0,)
    @test NNlib.within_gradient([ForwardDiff.Dual(1.0, 2)]) === true
end

@testset "maximum_dims" begin
    ind1 = [1,2,3,4,5,6]
    @test NNlib.maximum_dims(ind1) == (6,)
    ind2 = [(3,4,5), (1,2,3), (2,3,9)]
    @test NNlib.maximum_dims(ind2) == (3,4,9)
    ind3 = [(3,4,5) (1,2,3) (2,3,9);
            (4,6,2) (5,3,2) (4,4,4)]
    @test NNlib.maximum_dims(ind3) == (5,6,9)
    ind4 = CartesianIndex.(
           [(3,4,5) (1,2,3) (2,3,9);
            (4,6,2) (5,3,2) (4,4,4)])
    @test NNlib.maximum_dims(ind4) == (5,6,9)
end

@testset "reverse_indices" begin
    res = [
        CartesianIndex.([(1,1), (2,3)]),
        CartesianIndex.([(1,2), (2,2)]),
        CartesianIndex.([(3,1), (1,3), (2,4), (3,4)]),
        CartesianIndex.([(2,1), (1,4)]),
        CartesianIndex.([(3,2), (3,3)])
    ]
    idx = [1 2 3 4;
           4 2 1 3;
           3 5 5 3]
    @test NNlib.reverse_indices(idx) == res
    @test NNlib.reverse_indices(idx) isa typeof(res)
    idx = [(1,) (2,) (3,) (4,);
           (4,) (2,) (1,) (3,);
           (3,) (5,) (5,) (3,)]
    @test NNlib.reverse_indices(idx) == res
    @test NNlib.reverse_indices(idx) isa typeof(res)
    idx = CartesianIndex.(
        [(1,) (2,) (3,) (4,);
        (4,) (2,) (1,) (3,);
        (3,) (5,) (5,) (3,)])
    @test NNlib.reverse_indices(idx) == res
    @test NNlib.reverse_indices(idx) isa typeof(res)
end
