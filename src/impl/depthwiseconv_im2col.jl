# ## This file contains adapter code for doing depthwise convolutions with im2col.
#
#
# """
#     depthwiseconv_im2col!(y, x, w, cdims, col=similar(x); alpha=1, beta=0)
#
# Perform a depthwise convolution using im2col and GEMM, store the result in `y`.
#
# See `conv_im2col!()` for an explanation of optional parameters.
# """
# depthwiseconv_im2col!
#
# function depthwiseconv_im2col!(
#                 y::AbstractArray{T,5}, x::AbstractArray{T,5},
#                 w::AbstractArray{T,5}, cdims::DepthwiseConvDims;
#                 col::AbstractArray{T,2} = similar(x, im2col_dims(cdims)),
#                 alpha=T(1), beta=T(0)) where T
#     check_dims(size(x), size(w), size(y), cdims)
#
#     # This functions exactly the same as conv_im2col!(), except that we shard the
#     # incoming data into slices of single channels.  This means that we need to walk
#     # each pointer forward individually, as done below, taking a single input channel
#     # and combining it with each kernel individually, before walking forward and doing
#     # the next input channel.
#     M = prod(output_size(cdims))
#     N = channel_multiplier(cdims)
#     K = prod(kernel_size(cdims))
#
#     dcdims = DenseConvDims(cdims)
#     @inbounds for batch_idx in 1:size(x)[end]
#         im2col!(col, view(x, :, :, :, :, batch_idx), dcdims)
#
#         # We do a separate convolution for each channel in x, as we must
#         for c_in in 1:channels_in(cdims)
#             # Walk each pointer forward as we process each input channel
#             GC.@preserve col, w, y, begin
#                 col_ptr = pointer(col, (c_in-1)*M*K+1)
#                 w_ptr = pointer(w, (c_in-1)*K*N+1)
#                 y_ptr = pointer(y, ((batch_idx - 1)*channels_in(cdims) + c_in - 1)*M*N + 1)
#                 gemm!(Val(false), Val(false), M, N, K, alpha, col_ptr, w_ptr, beta, y_ptr)
#             end
#         end
#     end
#     return y
# end
#
# """
#     ∇depthwiseconv_filter_im2col!(dw, w, dy, cdims, col=similar(dw); alpha=1, beta)
#
# Depthwise conv2d backward pass onto the weights using im2col and GEMM.
# See the documentation for `conv_im2col!()` for explanation of optional parameters.
# """
# ∇depthwiseconv_filter_im2col!
#
# function ∇depthwiseconv_filter_im2col!(
#                 dw::AbstractArray{T,5}, x::AbstractArray{T,5},
#                 dy::AbstractArray{T,5}, cdims::DepthwiseConvDims;
#                 col::AbstractArray{T,2} = similar(dw, im2col_dims(cdims)),
#                 alpha=T(1), beta=T(0)) where T
#     check_dims(size(x), size(dw), size(dy), cdims)
#
#     M = prod(kernel_size(cdims))
#     N = channel_multiplier(cdims)
#     K = prod(output_size(cdims))
#
#     @inbounds for batch_idx in 1:size(x)[end]
#         im2col!(col, view(x, :, :, :, :, batch_idx), cdims)
#
#         # We do a separate convolution for each channel in x, as we must
#         for c_in in 1:channels_in(cdims)
#             # Walk each pointer forward as we process each input channel
#             GC.@preserve col, dw, dy, begin
#                 col_ptr = pointer(col, (c_in - 1)*M*K + 1)
#                 dy_ptr = pointer(dy, (batch_idx - 1)*N*K*channels_in(cdims) + (c_in - 1)*K*N + 1)
#                 dw_ptr = pointer(dw, (c_in - 1)*M*N + 1)
#                 gemm!(Val(true), Val(false), M, N, K, alpha, col_ptr, dy_ptr, beta, dw_ptr)
#             end
#         end
#
#         # Because we accumulate over batches in this loop, we must set `beta` equal
#         # to `1.0` from this point on.
#         beta = T(1)
#     end
#     return dw
# end
#
# """
#     depthwiseconv2d_Δx_im2col!(dx, w, dy, cdims, col=similar(dx); alpha=1, beta=0)
#
# Depwthwise conv2d backward pass onto the input using im2col and GEMM.
# See the documentation for `conv_im2col!()` for explanation of optional parameters.
# """
# ∇depthwiseconv_data_im2col!
#
# function ∇depthwiseconv_data_im2col!(
#                 dx::AbstractArray{T,5}, dy::AbstractArray{T,5},
#                 w::AbstractArray{T,5}, cdims::DepthwiseConvDims;
#                 col::AbstractArray{T,2} = similar(dx, im2col_dims(cdims)),
#                 alpha=T(1), beta=T(0)) where T
#     check_dims(size(dx), size(w), size(dy), cdims)
#
#     M = prod(output_size(cdims))
#     N = prod(kernel_size(cdims))
#     K = channel_multiplier(cdims)
#
#     @inbounds for batch_idx in 1:size(dx)[end]
#         # We do a separate convolution for each channel in x, as we must
#         for cidx in 1:channels_in(cdims)
#             GC.@preserve col, w, dy, begin
#                 # Walk each pointer forward as we process each input channel
#                 dy_ptr = pointer(dy, (batch_idx - 1)*M*K*channels_in(cdims)+(cidx - 1)*K*M + 1)
#                 w_ptr = pointer(w, (cidx - 1)*K*N + 1)
#                 col_ptr = pointer(col, (cidx - 1)*M*N + 1)
#                 gemm!(Val(false), Val(true), M, N, K, alpha, dy_ptr, w_ptr, T(0), col_ptr)
#             end
#         end
#         col2im!(view(dx, :, :, :, :, batch_idx), col, cdims)
#     end
#     return dx
# end
