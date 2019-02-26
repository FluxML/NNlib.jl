module NNlib
using Requires

# Include APIs
include("dim_helpers.jl")
include("activation.jl")
include("softmax.jl")
include("gemm.jl")
include("conv.jl")
include("pooling.jl")
include("upsample.jl")

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

# Upsampling implementation
include("impl/upsampling.jl")

end # module NNlib
