using NNlib, Test

@testset "NNlib" begin

include("activation.jl")
include("conv.jl")

xs = [-100_000, -100_000.]
@test softmax(xs) ≈ [0.5, 0.5]
@test logsoftmax(xs) ≈ log.([0.5, 0.5])

xs = rand(5)
@test softmax(xs) ≈ Float32.(exp.(xs) ./ sum(exp.(xs)))
@test logsoftmax(xs) ≈ Float32.(log.(softmax(xs)))
@test logsigmoid.(xs) ≈ Float32.(log.(sigmoid.(xs)))

xs = rand(5,10)
@test softmax(xs) ≈ Float32.(exp.(xs) ./ sum(exp.(xs), dims = 1))
@test logsoftmax(xs) ≈ Float32.(log.(softmax(xs)))
@test logsigmoid.(xs) ≈ Float32.(log.(sigmoid.(xs)))

for T in [:Float32, :Float64]
  @eval @test logsigmoid.($T[-100_000, 100_000.]) ≈ $T[-100_000, 0.]
end

## compare the outputs with the PyTorch nn.LogSoftmax returns
xs = Float32[1, 2, 3000.]
@test logsoftmax(xs) ≈ [-2999, -2998, 0]

xs = Float32[1 2 3; 1000 2000 3000]
@test logsoftmax(xs) ≈ [-999 -1998 -2997; 0 0 0.]

@test NNlib.∇logsoftmax(ones(size(xs)), xs) ≈ zeros(Float32, size(xs))
@test NNlib.∇softmax(ones(size(xs)), xs) ≈ zeros(Float32, size(xs))

xs = [-0.238639 0.748142 -0.283194 -0.525461 -1.5348 -0.797842; 0.690384 0.211427 0.254794 -0.213572 -0.314174 -0.372663; -1.14637 -0.577988 0.718952 0.91972 -0.620773 0.929977]
@test isapprox(NNlib.∇logsoftmax(ones(size(xs)), xs), [0.237703 -0.621474 0.448193 0.546047 0.564185 0.632273; -0.930163 0.0519798 0.0549979 0.3799 -0.477112 0.437428; 0.69246 0.569494 -0.503191 -0.925947 -0.0870738 -1.0697]; rtol = 1e-6)
@test isapprox(NNlib.∇softmax(ones(size(xs)), xs), zeros(size(xs)); atol = 1e-6)
end
