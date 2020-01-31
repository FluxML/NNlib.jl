module NNlib

# Start with the simplest stuff in here; activation functions
include("activation/activation.jl")
include("activation/softmax.jl")

# Load dimensionality helpers for convolution dispatching
include("dim_helpers/dim_helpers.jl")

# Define our convolution/pooling interface backend holders
include("interface.jl")

# Begin with straightforward direct implementations
include("direct/direct.jl")
# Next, im2col implementations
include("im2col/im2col.jl")

# Next, NNPACK implementations
using NNPACK_jll

# Check to see if NNPACK_jll is loadable
if isdefined(NNPACK_jll, :libnnpack)
    include("nnpack/NNPACK.jl")
else
    # Otherwise, signal to the rest of the world that this is unavailable
    """
        is_nnpack_available()

    Checks if the current platform/hardware is supported by NNPACK.
    Your platform sadly, is not supported by NNPACK.
    """
    is_nnpack_available() = false
end

# Finally, generate all the goodies for conv() and maxpool() and friends!
include("interface_impl.jl")

end # module NNlib
