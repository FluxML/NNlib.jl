module NNlib

using Requires

export
    σ,
    relu,
    leakyrelu,
    elu,
    swish,
    softmax,
    selu,
    softplus,
    softsign,
    conv2d,
    conv2d_grad_w,
    conv2d_grad_x,
    pool,
    pool_grad

include("init.jl")
include("numeric.jl")
include("activation.jl")
include("softmax.jl")
include("adapt.jl")
include("linalg.jl")
include("conv.jl")

end # module
