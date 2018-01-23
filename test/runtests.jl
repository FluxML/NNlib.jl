using NNlib
using Base.Test

@testset "NNlib" begin

include("activation.jl")
include("conv.jl")

xs = rand(5)
@test softmax(xs) ≈ exp.(xs) ./ sum(exp.(xs))
@test logsoftmax(xs) ≈ log.(softmax(xs))

xs = rand(5,10)
@test softmax(xs) ≈ exp.(xs) ./ sum(exp.(xs),1)
@test logsoftmax(xs) ≈ log.(softmax(xs))

## compare the outputs with the PyTorch nn.LogSoftmax returns
xs = Float32[1, 2, 3000.]
@test logsoftmax(xs) ≈ [-2999, -2998, 0]

xs = Float32[1 2 3; 1000 2000 3000]
@test logsoftmax(xs) ≈ [-999 -1998 -2997; 0 0 0.]
end
