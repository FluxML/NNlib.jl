# ## This file contains direct Julia implementations of depwthwise convolutions
#
# """
#     depthwiseconv_direct!(y, x, w, cdims; alpha=1, beta=0)
#
# Direct depthwise convolution implementation; used for debugging, tests, and mixing/
# matching of strange datatypes within a single convolution.  Uses naive nested for loop
# implementation and does not attempt to optimize performance.  Rather, this implementation
# is intended to be maximally understandable and debuggable, to aid in testing other, more
# performant implementations.  We also explicitly support mixing and matching of strange
# datatypes, so that if the user really wants to convolve an image of `UInt8`'s with a
# `Float16` kernel, storing the result in a `Float32` output, there is at least a function
# call for that madness.
#
# One subtlety about depthwise convolutions; the shape of a depthwise convolutional kernel
# is `(spatial_dims..., C_mult, C_in)`, so the axis that must match with the number of
# channels in `x` is the last, not the second-to-last, as in a normal dense convolution.
#
# See the docstring for `conv_direct!()` for more on the optional parameters.
# """
# function depthwiseconv_direct!(
#                 y::AbstractArray{yT,5}, x::AbstractArray{xT,5},
#                 w::AbstractArray{wT,5}, cdims::DepthwiseConvDims;
#                 alpha::yT = yT(1), beta::yT = yT(0)) where {yT, xT, wT}
#     check_dims(size(x), size(w), size(y), cdims)
#
#     width, height, depth = input_size(cdims)
#     kernel_w, kernel_h, kernel_d = kernel_size(cdims)
#     out_c = channels_out(cdims)
#     pad_w_lo, pad_w_hi, pad_h_lo, pad_h_hi, pad_d_lo, pad_d_hi = padding(cdims)
#     dil_w, dil_h, dil_d = dilation(cdims)
#     stride_w, stride_h, stride_d = stride(cdims)
#     out_width, out_height, out_depth = output_size(cdims)
#
#     # If we're doing crosscorr instead of conv, then don't bother to flip `w`
#     if !flipkernel(cdims)
#         w = w[end:-1:1, end:-1:1, end:-1:1, :, :]
#     end
#
#     # A helper function to project from output (w, h) to input (input_w, input_h)
#     @inline project(idx, stride, pad) = (idx - 1)*stride - pad + 1
#
#     # explicit formulation of convolution.  Oh hoisting gods, hear my plea.
#     @inbounds for batch in 1:size(x)[end],
#         c_mult in 1:channel_multiplier(cdims),
#         c_in in 1:channels_in(cdims),
#         h_idx in 1:out_height,
#         w_idx in 1:out_width,
#         d_idx in 1:out_depth
#
#         # Starting points of the window of x we're going to grab
#         x_w = project(w_idx, stride_w, pad_w_lo)
#         x_h = project(h_idx, stride_h, pad_h_lo)
#         x_d = project(d_idx, stride_d, pad_d_lo)
#
#         # Grow that starting point into ranges
#         x_widxs = x_w .+ (0:dil_w:(dil_w*kernel_w-1))
#         x_hidxs = x_h .+ (0:dil_h:(dil_h*kernel_h-1))
#         x_didxs = x_d .+ (0:dil_d:(dil_d*kernel_d-1))
#         w_widxs = 1:kernel_w
#         w_hidxs = 1:kernel_h
#         w_didxs = 1:kernel_d
#
#         # Clamp the ranges to simulate padding
#         x_widxs, w_widxs = clamp_lo(x_widxs, w_widxs)
#         x_widxs, w_widxs = clamp_hi(x_widxs, w_widxs, width)
#         x_hidxs, w_hidxs = clamp_lo(x_hidxs, w_hidxs)
#         x_hidxs, w_hidxs = clamp_hi(x_hidxs, w_hidxs, height)
#         x_didxs, w_didxs = clamp_lo(x_didxs, w_didxs)
#         x_didxs, w_didxs = clamp_hi(x_didxs, w_didxs, depth)
#
#         # Grab our slices (for a single channel pairing, as this is depthwise)
#         c_out = (c_in - 1)*channel_multiplier(cdims) + c_mult
#         x_slice = view(x, x_widxs, x_hidxs, x_didxs, c_in, batch)
#         w_slice = view(w, w_widxs, w_hidxs, w_didxs, c_mult, c_in)
#
#         # Do the dotproduct dance, then weight by alpha/beta and git 'er done
#         dotprod = sum(x_slice .* w_slice)
#         prev_yval::yT = beta*y[w_idx, h_idx, d_idx, c_out, batch]
#         y[w_idx, h_idx, d_idx, c_out, batch] = alpha*convert(yT, dotprod) + prev_yval
#     end
#
#     return y
# end
#
# """
#     ∇depthwiseconv_data_direct!(dx, dy, w, cdims; alpha=1, beta=0)
#
# Calculate the gradient imposed upon `x` in the depthwise convolution `y = x * w`.
# We make use of the fact that a depthwise convolution is equivalent to `C_in` separate
# normal convolutions between that channel of `x` and the `C_mult` different kernels that
# get applied to it.  The output of such a convolution is the gradient imposed upon that
# particular channel of `x`, and so we simply walk through `x`, calculating the gradient
# for each batch and channel independently.
# """
# ∇depthwiseconv_data_direct!
#
# function ∇depthwiseconv_data_direct!(
#                 dx::AbstractArray{xT,5}, dy::AbstractArray{yT,5},
#                 w::AbstractArray{wT,5}, cdims::DepthwiseConvDims;
#                 alpha::xT=xT(1), beta::xT=xT(0)) where {xT, yT, wT}
#     # We do a separate convolution for each channel in x
#     @inbounds for cidx in 1:channels_in(cdims)
#         # For this batch and in-channel, we have a normal transposed convolution
#         # between this slice of `x` and the corresponding slices of `w` and `dy`:
#         dx_slice = view(dx, :, :, :, cidx:cidx, :)
#         C_mult = channel_multiplier(cdims)
#         dy_slice = view(dy, :, :, :, ((cidx-1)*C_mult + 1):cidx*C_mult, :)
#         w_slice = permutedims(view(w, :, :, :, :, cidx:cidx), (1, 2, 3, 5, 4))
#
#         # Adapt a DenseConvDims out of this DepthwiseConvDims, setting the in/out
#         # channels appropriately for this one convolution.
#         cdims_slice = DenseConvDims(cdims;
#             C_in=1,
#             C_out=channel_multiplier(cdims),
#         )
#
#         ∇conv_data_direct!(dx_slice, dy_slice, w_slice, cdims_slice;
#                                                alpha=alpha, beta=beta)
#     end
#     return dx
# end
#
# """
#     ∇depthwiseconv_filter_direct!(dw, x, dy, cdims; alpha=1, beta=0)
#
# Calculate the gradient imposed upon `w` in the depthwise convolution `y = x * w`.
# """
# ∇depthwiseconv_filter_direct!
#
# function ∇depthwiseconv_filter_direct!(
#                 dw::AbstractArray{wT,5}, x::AbstractArray{xT,5},
#                 dy::AbstractArray{yT,5}, cdims::DepthwiseConvDims;
#                 alpha::wT=wT(1),beta::wT=wT(0)) where {xT, yT, wT}
#     # We do a separate convolution for each channel in x
#     @inbounds for cidx in 1:channels_in(cdims)
#         # For this batch and in-channel, we have a normal transposed convolution
#         # between this slice of `x` and the corresponding slices of `w` and `dy`:
#         x_slice = view(x, :, :, :, cidx:cidx, :)
#         C_mult = channel_multiplier(cdims)
#         dy_slice = view(dy, :, :, :, ((cidx-1)*C_mult + 1):cidx*C_mult, :)
#         dw_slice = permutedims(view(dw, :, :, :, :, cidx:cidx), (1, 2, 3, 5, 4))
#
#         # Adapt a DenseConvDims out of this DepthwiseConvDims, setting the in/out
#         # channels appropriately for this one convolution.
#         cdims_slice = DenseConvDims(cdims;
#             C_in=1,
#             C_out=channel_multiplier(cdims),
#         )
#
#         ∇conv_filter_direct!(dw_slice, x_slice, dy_slice, cdims_slice;
#                                                 alpha=alpha, beta=beta)
#         dw[:, :, :, :, cidx:cidx] .= permutedims(dw_slice, (1, 2, 3, 5, 4))
#     end
#     return dw
# end
#
#
