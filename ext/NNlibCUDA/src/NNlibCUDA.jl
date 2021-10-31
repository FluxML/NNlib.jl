module NNlibCUDA

using NNlib
using CUDA
using Random, Statistics

const IntOrIntTuple = Union{Integer, NTuple{N,<:Integer} where N}

include("upsample.jl")
include("sampling.jl")
include("activations.jl")
include("batchedmul.jl")
include("scatter.jl")
include("gather.jl")
include("utils.jl")
include("cudnn/cudnn.jl")
include("cudnn/conv.jl")
include("cudnn/pooling.jl")
include("cudnn/softmax.jl")
include("cudnn/activations.jl")
include("cudnn/batchnorm.jl")

function main()
    input = reshape(collect(1.0:4.0), (2, 2, 1, 1))
    grid = zeros(Float64, 2, 2, 2, 1)
    grid[1, 1, :, 1] .= (-1, -1)
    grid[2, 1, :, 1] .= (1, -1)

    grid[1, 2, :, 1] .= (-1, 1)
    grid[2, 2, :, 1] .= (1, 1)

    padding_mode = 0
    dx = CuArray(input)
    dg = CuArray(grid)
    dy = grid_sampler(dx, dg, padding_mode)

    out_grad = CUDA.ones(Float64, size(dy))
    ∇input, ∇grid = ∇grid_sampler(out_grad, dx, dg, padding_mode)

    display(dy |> collect); println()
    display(∇input |> collect); println()
    display(∇grid |> collect); println()
end
main()

end # module
