using Statistics: mean

@testset "sparsemax integer input" begin
    @test sparsemax(Int[0 0; 0 0]) == [0.5 0.5; 0.5 0.5]
    @test sparsemax(Int[1 0; 0 1]) == [1.0 0.0; 0.0 1.0]
    @test sparsemax(Int[1 2; 1 1]) == [0.5 1.0; 0.5 0.0]
end

@testset "AutoDiff" begin
    gradtest(x -> sparsemax(x) .* (1:3), 3)
    gradtest(x -> sparsemax(x) .* (1:3), (3,5), atol = 1e-4)
    gradtest(x -> sparsemax(x, dims = 2) .* (1:3), (3,5), atol = 1e-4)

    for d in (1, 2)
        gradtest(sparsemax, (3,4), fkwargs = (dims = d,))
    end
end