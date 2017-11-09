module NNlib

export σ, relu, leakyrelu, elu, swish, softmax

include("numeric.jl")
include("activation.jl")
include("softmax.jl")
include("adapt.jl")

end # module
