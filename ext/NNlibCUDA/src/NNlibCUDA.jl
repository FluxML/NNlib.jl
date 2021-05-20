module NNlibCUDA

using NNlib
using CUDA
using Random, Statistics

const IntOrIntTuple = Union{Integer, NTuple{N,<:Integer} where N}

include("upsample.jl")
include("activations.jl")
include("batchedmul.jl")
include("scatter.jl")
include("cudnn/cudnn.jl")
include("cudnn/conv.jl")
include("cudnn/pooling.jl")
include("cudnn/softmax.jl")
include("cudnn/activations.jl")
include("cudnn/batchnorm.jl")

function main()
    dx = randn(Float32, 10, 10, 8, 1)
    dw = randn(Float32, 3, 3, 1, 8)
    dwcdims = DenseConvDims(dx, dw; groups=8)
    @info dwcdims

    dxd = dx |> cu
    dwd = dw |> cu
    o = NNlib.conv(dx, dw, dwcdims)
    od = NNlib.conv(dxd, dwd, dwcdims)
    @info "conv: ", o ≈ collect(od)
    @info "conv filter: ", ∇conv_filter(dx, o, dwcdims) ≈ collect(∇conv_filter(dxd, od, dwcdims))
    @info "conv data: ", ∇conv_data(o, dw, dwcdims) ≈ collect(∇conv_data(od, dwd, dwcdims))
end
# main()

end
