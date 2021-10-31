module NNlib

using Pkg
using Requires
using ChainRulesCore
import ChainRulesCore: rrule
using Base.Broadcast: broadcasted
using Statistics: mean

using BenchmarkTools

const IntOrIntTuple = Union{Integer, NTuple{N,<:Integer} where N}
const Numeric = Union{AbstractArray{<:T}, T} where {T<:Number}

# Include APIs
include("dim_helpers.jl")

is_nnpack_available() = false

@init @require NNPACK_jll="a6bfbf70-4841-5cb9-aa18-3a8ad3c413ee"  begin
  if isdefined(NNPACK_jll, :libnnpack)
    include("nnpack/NNPACK.jl")
  else
    @warn "NNPACK not available for your platform: " *
          "$( Pkg.BinaryPlatforms.platform_name(Pkg.BinaryPlatforms.platform_key_abi()))" *
          "($( Pkg.BinaryPlatforms.triplet(Pkg.BinaryPlatforms.platform_key_abi())))
          You will be able to use only the default Julia NNlib backend"
  end
end

include("activations.jl")
include("softmax.jl")
include("batched/batchedmul.jl")
include("gemm.jl")
include("conv.jl")
include("conv_bias_act.jl")
include("pooling.jl")
include("padding.jl")
include("upsample.jl")
include("gather.jl")
include("scatter.jl")
include("utils.jl")
include("sampling.jl")

## Include implementations
include("impl/padding_edges.jl")

# Direct implementations of convolutional and depthwise-convolutional algorithms
include("impl/conv_direct.jl")
include("impl/depthwiseconv_direct.jl")
# im2col implementations of convolutional and depthwise-convolutional algorithms
include("impl/conv_im2col.jl")
include("impl/depthwiseconv_im2col.jl")

# Direct implementations of pooling
include("impl/pooling_direct.jl")
include("deprecations.jl")

function main()
    T = Float64
    padding_mode = Val(:border)
    w, h, c, n = 128, 128, 8, 32
    input = rand(T, w, h, c, n)
    grid = zeros(T, 2, w, h, n)
    delta = 0.01

    for xi in 1:w, yi in 1:h, ni in 1:n
        grid[1, xi, yi, ni] = (xi / w) * 2 - 1 + delta
        grid[2, xi, yi, ni] = (yi / h) * 2 - 1
    end

    output = similar(input, T, (w, h, c, n))
    dx = similar(input)
    dgrid = similar(grid)

    sampled = grid_sampler(input, grid, padding_mode)
    external_grad = ones(size(sampled))
    ∇input, ∇grid = ∇grid_sampler(external_grad, input, grid, padding_mode)

    @btime grid_sampler!($output, $input, $grid, $padding_mode)
    @btime ∇grid_sampler!($dx, $dgrid, $external_grad, $input, $grid, $padding_mode)
end
main()

end # module NNlib
