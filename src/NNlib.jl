module NNlib

using Requires, Libdl

export σ, sigmoid, relu, leakyrelu, elu, gelu, swish, selu, softplus, softsign, logσ, logsigmoid,
  softmax, logsoftmax, maxpool, meanpool, batchedmul

include("numeric.jl")
include("activation.jl")
include("softmax.jl")
include("logsoftmax.jl")
include("linalg.jl")
include("batchedmul.jl")
include("conv.jl")
include("cubroadcast.jl")

end # module
