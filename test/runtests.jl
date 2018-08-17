using NNlib, Test

@testset "NNlib" begin

include("activation.jl")
include("conv.jl")

xs = [-100_000, -100_000.]
@test softmax(xs) ≈ [0.5, 0.5]
@test logsoftmax(xs) ≈ log.([0.5, 0.5])

xs = rand(5)
@test softmax(xs) ≈ exp.(xs) ./ sum(exp.(xs))
@test logsoftmax(xs) ≈ log.(softmax(xs))
@test logsigmoid.(xs) ≈ log.(sigmoid.(xs))

xs = rand(5,10)
@test softmax(xs) ≈ exp.(xs) ./ sum(exp.(xs), dims = 1)
@test logsoftmax(xs) ≈ log.(softmax(xs))
@test logsigmoid.(xs) ≈ log.(sigmoid.(xs))

for T in [:Float32, :Float64]
  @eval @test logsigmoid.($T[-100_000, 100_000.]) ≈ $T[-100_000, 0.]
end

## compare the outputs with the PyTorch nn.LogSoftmax returns
xs = Float32[1, 2, 3000.]
@test logsoftmax(xs) ≈ [-2999, -2998, 0]

xs = Float32[1 2 3; 1000 2000 3000]
@test logsoftmax(xs) ≈ [-999 -1998 -2997; 0 0 0.]
end
