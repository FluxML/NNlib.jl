module NNlib

using Requires

export σ, sigmoid, relu, leakyrelu, elu, swish, selu, softplus, softsign, logσ, logsigmoid,
  softmax, logsoftmax, conv2d, conv3d, maxpool2d, maxpool3d, avgpool2d, avgpool3d

include("numeric.jl")
include("activation.jl")
include("softmax.jl")
include("logsoftmax.jl")
include("linalg.jl")
include("conv.jl")

end # module
