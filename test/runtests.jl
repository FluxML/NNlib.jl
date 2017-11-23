using NNlib
using Base.Test

@testset "NNlib" begin

include("activation.jl")
include("conv.jl")

xs = rand(5)
@test softmax(xs) ≈ exp.(xs) ./ sum(exp.(xs))

xs = rand(5,10)
@test softmax(xs) ≈ exp.(xs) ./ sum(exp.(xs),1)

end
