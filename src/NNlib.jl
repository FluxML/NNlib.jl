module NNlib
using Requires

# Include APIs
include("dim_helpers.jl")

# NNPACK support
include(joinpath(@__DIR__, "..", "deps", "deps.jl"))
if check_deps() == nothing
  include("nnpack/NNPACK.jl")
else
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
