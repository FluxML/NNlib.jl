module NNlib

using Requires

export σ, sigmoid, relu, leakyrelu, elu, swish, selu, softplus, softsign, logσ, logsigmoid,
  softmax, logsoftmax, maxpool, meanpool, nnpack_available

nnpack_available()= false ||  (is_linux() || is_apple()) && isfile("libnnpack.so")

include("numeric.jl")
include("activation.jl")
include("softmax.jl")
include("logsoftmax.jl")
include("linalg.jl")
include("conv.jl")
include("cubroadcast.jl")

if nnpack_available()
	include("NNPACK.jl")
end

end # module
