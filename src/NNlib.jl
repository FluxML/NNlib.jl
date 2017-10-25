module NNlib

export σ, relu, leakyrelu, elu, swish, softmax

include("activation.jl")
include("softmax.jl")
include("convolution.jl")
include("adapt.jl")

end # module
