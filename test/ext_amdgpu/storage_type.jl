@testset "NNlib storage type" begin
    x = ROCArray(ones(Float32, 10, 10))
    @test storage_type(x) <: ROCArray{Float32, 2}
    @test storage_type(reshape(view(x, 1:2:10,:), 10, :)) <: ROCArray{Float32, 2}

    @test is_strided(x)
    @test is_strided(view(x, 1:2:5,:))
    @test is_strided(PermutedDimsArray(x, (2, 1)))

    @test !is_strided(reshape(view(x, 1:2:10, :), 10, :))
    @test !is_strided((x .+ im)')
    @test !is_strided(Diagonal(ROCArray(ones(3))))
end
