module NNlib

using Requires

export σ, sigmoid, hardσ, hard_sigmoid, hardσ_keras, hard_sigmoid_keras,
  relu, leakyrelu, elu, swish, selu, softplus, softsign, logσ, logsigmoid,
  softmax, logsoftmax, maxpool, meanpool, hard_tanh

include("numeric.jl")
include("activation.jl")
include("softmax.jl")
include("logsoftmax.jl")
include("linalg.jl")
include("conv.jl")
include("cubroadcast.jl")

end # module
