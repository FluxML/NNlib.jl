"""
Direct implementations of convolution, pooling, etc...
"""
module Direct
using ..NNlib
using ..NNlib: output_size, input_size, kernel_size, channels_in, channels_out, check_dims,
               spatial_dims, stride, padding, dilation, flipkernel, calc_padding_regions,
               transpose_swapbatch, predilate, transpose_pad, channel_multiplier

include("conv_direct.jl")
include("depthwiseconv_direct.jl")
include("pooling_direct.jl")

# Here we register our convolution and pooling methods with the parent NNlib module.
# We have direct implementations of just about everything, so push them all!
import ..conv_backends, ..pooling_backends
push!(conv_backends[:conv], :direct)
push!(conv_backends[:∇conv_data], :direct)
push!(conv_backends[:∇conv_filter], :direct)
push!(conv_backends[:depthwiseconv], :direct)
push!(conv_backends[:∇depthwiseconv_data], :direct)
push!(conv_backends[:∇depthwiseconv_filter], :direct)

push!(pooling_backends[:maxpool], :direct)
push!(pooling_backends[:meanpool], :direct)

end # module Direct

# Self-using?  Yes.
using .Direct
import .Direct: conv_direct!, ∇conv_data_direct!, ∇conv_filter_direct!,
                depthwiseconv_direct!, ∇depthwiseconv_data_direct!, ∇depthwiseconv_filter_direct!,
                meanpool_direct!, maxpool_direct!, ∇meanpool_direct!, ∇maxpool_direct!