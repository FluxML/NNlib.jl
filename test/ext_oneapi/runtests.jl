oneAPI.allowscalar(false)

@testset "Batched multiplication" begin
    include("batched_mul.jl")
end
