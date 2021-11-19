using Statistics: mean

@testset "sparsemax integer input" begin
    @test sparsemax(Int[0 0; 0 0]) == [0.5 0.5; 0.5 0.5]
end