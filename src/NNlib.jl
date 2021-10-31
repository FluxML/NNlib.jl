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
    w, h, c = 1024, 1024, 2
    input = rand(Float64, w, h, c, 1)
    grid = zeros(Float64, w, h, 2, 1)
    delta = 0.01

    for xi in 1:1024, yi in 1:1024
        grid[xi, yi, 1, 1] = (xi / 1024) * 2 - 1 + delta
        grid[xi, yi, 2, 1] = (yi / 1024) * 2 - 1
    end

    padding_mode = 0
    sampled = grid_sampler(input, grid, padding_mode)
    out_grad = ones(size(sampled))
    ∇input, ∇grid = ∇grid_sampler(out_grad, input, grid, padding_mode)

    @btime grid_sampler($input, $grid, $padding_mode)
    @btime ∇grid_sampler($out_grad, $input, $grid, $padding_mode)

    # display(sampled); println()
    # @info "dx"
    # display(∇input); println()
    # @info "dgrid"
    # display(∇grid); println()
end
main()

end # module NNlib
