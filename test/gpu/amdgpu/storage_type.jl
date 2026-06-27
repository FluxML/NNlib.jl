@testset "NNlib storage type" begin
    x = ROCArray(ones(Float32, 10, 10))
    @test NNlib.storage_type(x) <: ROCArray{Float32, 2}
    @test NNlib.storage_type(reshape(view(x, 1:2:10,:), 10, :)) <: ROCArray{Float32, 2}

    @test NNlib.is_strided(x)
    @test NNlib.is_strided(view(x, 1:2:5,:))
    @test NNlib.is_strided(PermutedDimsArray(x, (2, 1)))

    @test !NNlib.is_strided(reshape(view(x, 1:2:10, :), 10, :))
    @test !NNlib.is_strided((x .+ im)')
    @test !NNlib.is_strided(LinearAlgebra.Diagonal(ROCArray(ones(3))))
end
