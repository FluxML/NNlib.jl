module NNlib

using Requires, Libdl

export σ, sigmoid, relu, leakyrelu, elu, gelu, swish, selu, softplus, softsign, logσ, logsigmoid,
  softmax, logsoftmax, maxpool, meanpool

include("numeric.jl")
include("activation.jl")
include("softmax.jl")
include("logsoftmax.jl")
include("linalg.jl")
include("conv.jl")
include("cubroadcast.jl")

try
    global ENABLE_NNPACK = parse(UInt64, ENV["ENABLE_NNPACK"])
catch
    global ENABLE_NNPACK = 1
end

if Sys.islinux() && ENABLE_NNPACK == 1
    include("nnpack/NNPACK.jl")
    include("backends.jl")
end

end # module
