module NNlib

using Requires

export σ, sigmoid, relu, leakyrelu, elu, swish, selu, softplus, softsign,
  softmax, conv2d, maxpool2d, avgpool2d

const libnnlib = Libdl.find_library("nnlib.$(Libdl.dlext)", [joinpath(@__DIR__, "..", "deps")])

include("numeric.jl")
include("activation.jl")
include("softmax.jl")
include("adapt.jl")
include("linalg.jl")
include("conv.jl")

end # module
