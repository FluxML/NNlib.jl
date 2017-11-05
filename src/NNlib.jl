module NNlib

export σ, relu, leakyrelu, elu, swish, softmax, selu, softplus, softsign

include("activation.jl")
include("softmax.jl")
include("adapt.jl")

end # module
