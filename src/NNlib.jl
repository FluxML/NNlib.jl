module NNlib
using Pkg
using Requires
using NNPACK_jll

# Include APIs
include("dim_helpers.jl")


if isdefined(NNPACK_jll, :libnnpack)
  include("nnpack/NNPACK.jl")
else
  @warn "NNPACK not available for your platform: " *
        "$( Pkg.BinaryPlatforms.platform_name(Pkg.BinaryPlatforms.platform_key_abi()))" *
        "($( Pkg.BinaryPlatforms.triplet(Pkg.BinaryPlatforms.platform_key_abi())))
        You will be able to use only the default Julia NNlib backend"
  is_nnpack_available() = false
end

include("activation.jl")
include("softmax.jl")
include("batched/batchedmul.jl")
include("gemm.jl")
include("conv.jl")
include("pooling.jl")

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

end # module NNlib
