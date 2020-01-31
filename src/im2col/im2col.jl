module Im2col
using ..NNlib
using ..NNlib: im2col_dims, output_size, input_size, kernel_size, channels_in, channels_out, check_dims,
               spatial_dims, stride, padding, dilation, flipkernel, calc_padding_regions, channel_multiplier
using Base.Threads

include("gemm.jl")
include("conv_im2col.jl")
include("depthwiseconv_im2col.jl")


# Here we register our convolution methods with the parent NNlib module.
# We only do convolution, no pooling.
import ..conv_backends
push!(conv_backends[:conv], :im2col)
push!(conv_backends[:∇conv_data], :im2col)
push!(conv_backends[:∇conv_filter], :im2col)
push!(conv_backends[:depthwiseconv], :im2col)
push!(conv_backends[:∇depthwiseconv_data], :im2col)
push!(conv_backends[:∇depthwiseconv_filter], :im2col)

end # module Im2col

using .Im2col
import .Im2col: conv_im2col!, ∇conv_data_im2col!, ∇conv_filter_im2col!,
                depthwiseconv_im2col!, ∇depthwiseconv_data_im2col!,
                ∇depthwiseconv_filter_im2col!