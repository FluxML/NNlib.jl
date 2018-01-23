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

<<<<<<< HEAD
xs = rand(5)
@test logsoftmax(xs) ≈ log.(softmax(xs))

xs = rand(5,10)
@test logsoftmax(xs) ≈ log.(softmax(xs))

#Comparing results from tests with TensorFlow results
xs = Float32[-1000., -500., -20., 0., 20., 500., 1000.]
@test logsoftmax(xs) ≈ Float32[-2000., -1500., -1020., -1000., -980., -500., 0.]

xs = Float32[-1000. -500. 2.; -2. 500. 1000.]
@test logsoftmax(xs) ≈ Float32[-998. -1000. -998.; 0. 0. 0.]


=======
## compare the outputs with the PyTorch nn.LogSoftmax returns
xs = Float32[1, 2, 3000.]
@test logsoftmax(xs) ≈ [-2999, -2998, 0]

xs = Float32[1 2 3; 1000 2000 3000]
@test logsoftmax(xs) ≈ [-999 -1998 -2997; 0 0 0.]
>>>>>>> upstream/master
end
