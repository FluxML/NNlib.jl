using Statistics: mean

@testset "sparsemax integer, float input" begin
    @test sparsemax(Int[0 0; 0 0]) == [0.5 0.5; 0.5 0.5]
    @test sparsemax(Int[1 0; 0 1]) == [1.0 0.0; 0.0 1.0]
    @test sparsemax(Int[1 2; 1 1]) == [0.5 1.0; 0.5 0.0]

    @test sparsemax(Float32[0 0; 0 0]) == Float32[0.5 0.5; 0.5 0.5]
    @test sparsemax(Float64[0 0; 0 0]) == Float64[0.5 0.5; 0.5 0.5]
end


@testset "sparsemax vector input" begin
    @test sparsemax(Float32[1, 2, 2]; dims=1) == Float32[0.0, 0.5, 0.5]
    @test sparsemax(Float32[1, 2, 2]; dims=2) == Float32[1.0, 1.0, 1.0]
end

@testset "sparsemax inferred return type" begin
    for T in [Float16, Float32, Float64]
        val = @inferred sparsemax(T[1,2,3])
        @test typeof(val) == Vector{T}
    end
end

@testset "AutoDiff" begin
    gradtest(x -> sparsemax(x) .* (1:3), 3)
    gradtest(x -> sparsemax(x) .* (1:3), (3,5), atol = 1e-4)
    gradtest(x -> sparsemax(x, dims = 2) .* (1:3), (3,5), atol = 1e-4)

    for d in (1, 2)
        gradtest(sparsemax, (3,4), fkwargs = (dims = d,))
    end
end