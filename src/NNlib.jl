module NNlib

using Requires

export σ, sigmoid, relu, leakyrelu, elu, swish, selu, softplus, softsign, logσ, logsigmoid,
  softmax, logsoftmax, conv2d, maxpool2d, avgpool2d

const libnnlib = Libdl.find_library("nnlib.$(Libdl.dlext)", [joinpath(@__DIR__, "..", "deps")])

include("numeric.jl")
include("activation.jl")
include("logsigmoid.jl")
include("softmax.jl")
include("logsoftmax.jl")
include("linalg.jl")
include("conv.jl")

end # module
