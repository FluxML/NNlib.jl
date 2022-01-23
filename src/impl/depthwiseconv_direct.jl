## This file contains direct Julia implementations of depwthwise convolutions

"""
    depthwiseconv_direct!(y, x, w, cdims; alpha=1, beta=0)

Direct depthwise convolution implementation; used for debugging, tests, and mixing/
matching of strange datatypes within a single convolution.  Uses naive nested for loop
implementation and does not attempt to optimize performance.  Rather, this implementation
is intended to be maximally understandable and debuggable, to aid in testing other, more
performant implementations.  We also explicitly support mixing and matching of strange
datatypes, so that if the user really wants to convolve an image of `UInt8`'s with a
`Float16` kernel, storing the result in a `Float32` output, there is at least a function
call for that madness.

One subtlety about depthwise convolutions; the shape of a depthwise convolutional kernel
is `(spatial_dims..., C_mult, C_in)`, so the axis that must match with the number of
channels in `x` is the last, not the second-to-last, as in a normal dense convolution.

See the docstring for `conv_direct!()` for more on the optional parameters.
"""
function depthwiseconv_direct!(y::AbstractArray{yT,5}, x::AbstractArray{xT,5},
                      w::AbstractArray{wT,5}, cdims::DepthwiseConvDims;
                      alpha::yT=yT(1), beta=false) where {yT, xT, wT}
    check_dims(size(x), size(w), size(y), cdims)

    width, height, depth = input_size(cdims)
    kernel_w, kernel_h, kernel_d = kernel_size(cdims)
    pad_w_lo, _, pad_h_lo, _, pad_d_lo, _ = padding(cdims)
    dil_w, dil_h, dil_d = dilation(cdims)
    stride_w, stride_h, stride_d = stride(cdims)

    # Create a method that determines how we're going to index into `w`
    kproj(k, M, cdims::DepthwiseConvDims) = flipkernel(cdims) ? k : (M - k + 1)

    # A helper function to project from output (w, h) to input (input_w, input_h)
    project(idx, stride, pad) = (idx - 1)*stride - pad + 1

    # Use `calc_padding_regions` to determine where we do or don't need to worry about padding
    padded_regions, central_region = calc_padding_regions(cdims)

    # Start with the central region
    w_region, h_region, d_region = central_region
    @inbounds for batch in 1:size(x)[end],
        c_mult in 1:channel_multiplier(cdims),
        c_in in 1:channels_in(cdims),
        d_idx in d_region,
        h_idx in h_region,
        w_idx in w_region

        # Since we're in the central region, we don't need to worry about clamping
        dotprod = yT(0)
        c_out = (c_in - 1)*channel_multiplier(cdims) + c_mult
        for kd in 1:kernel_d,
            kh in 1:kernel_h,
            kw in 1:kernel_w

            # Hoist me, you coward.
            x_d = project(d_idx, stride_d, pad_d_lo) + (kd - 1)*dil_d
            x_h = project(h_idx, stride_h, pad_h_lo) + (kh - 1)*dil_h
            x_w = project(w_idx, stride_w, pad_w_lo) + (kw - 1)*dil_w

            x_val = x[x_w, x_h, x_d, c_in, batch]
            w_val = w[kproj(kw, kernel_w, cdims),
                      kproj(kh, kernel_h, cdims),
                      kproj(kd, kernel_d, cdims),
                      c_mult, c_in]
            dotprod = muladd(x_val, w_val, dotprod)
        end
        y[w_idx, h_idx, d_idx, c_out, batch] = alpha*dotprod + beta*y[w_idx, h_idx, d_idx, c_out, batch]
    end

    # Next, do potentially-padded regions:
    @inbounds for (w_region, h_region, d_region) in padded_regions,
        batch in 1:size(x)[end],
        c_mult in 1:channel_multiplier(cdims),
        c_in in 1:channels_in(cdims),
        d_idx in d_region,
        h_idx in h_region,
        w_idx in w_region

        dotprod = yT(0)
        c_out = (c_in - 1)*channel_multiplier(cdims) + c_mult
        for kd in 1:kernel_d
            # Probe for out-of-bounds accesses on `x` and `continue` if we hit one
            x_d = project(d_idx, stride_d, pad_d_lo) + (kd - 1)*dil_d
            if x_d <= 0 || x_d > depth
                continue
            end

            for kh in 1:kernel_h
                x_h = project(h_idx, stride_h, pad_h_lo) + (kh - 1)*dil_h
                if x_h <= 0 || x_h > height
                    continue
                end

                for kw in 1:kernel_w
                    x_w = project(w_idx, stride_w, pad_w_lo) + (kw - 1)*dil_w
                    if x_w <= 0 || x_w > width
                        continue
                    end

                    x_val = x[x_w, x_h, x_d, c_in, batch]
                    w_val = w[kproj(kw, kernel_w, cdims),
                              kproj(kh, kernel_h, cdims),
                              kproj(kd, kernel_d, cdims),
                              c_mult, c_in]
                    dotprod = muladd(x_val, w_val, dotprod)
                end
            end
        end

        y[w_idx, h_idx, d_idx, c_out, batch] = alpha*dotprod + beta*y[w_idx, h_idx, d_idx, c_out, batch]
    end

    return y
end

"""
    ∇depthwiseconv_data_direct!(dx, dy, w, cdims; alpha=1, beta=0)

Calculate the gradient imposed upon `x` in the depthwise convolution `y = x * w`.
We make use of the fact that a depthwise convolution is equivalent to `C_in` separate
normal convolutions between that channel of `x` and the `C_mult` different kernels that
get applied to it.  The output of such a convolution is the gradient imposed upon that
particular channel of `x`, and so we simply walk through `x`, calculating the gradient
for each batch and channel independently.
"""
∇depthwiseconv_data_direct!

function ∇depthwiseconv_data_direct!(
                dx::AbstractArray{xT,5}, dy::AbstractArray{yT,5},
                w::AbstractArray{wT,5}, cdims::DepthwiseConvDims;
                alpha::xT=xT(1), beta=false) where {xT, yT, wT}
    # We do a separate convolution for each channel in x
    @inbounds for cidx in 1:channels_in(cdims)
        # For this batch and in-channel, we have a normal transposed convolution
        # between this slice of `x` and the corresponding slices of `w` and `dy`:
        dx_slice = view(dx, :, :, :, cidx:cidx, :)
        C_mult = channel_multiplier(cdims)
        dy_slice = view(dy, :, :, :, ((cidx-1)*C_mult + 1):cidx*C_mult, :)
        w_slice = permutedims(view(w, :, :, :, :, cidx:cidx), (1, 2, 3, 5, 4))

        # Adapt a DenseConvDims out of this DepthwiseConvDims, setting the in/out
        # channels appropriately for this one convolution.
        cdims_slice = DenseConvDims(cdims;
            C_in=1,
            C_out=channel_multiplier(cdims),
        )

        ∇conv_data_direct!(dx_slice, dy_slice, w_slice, cdims_slice;
                                               alpha=alpha, beta=beta)
    end
    return dx
end

"""
    ∇depthwiseconv_filter_direct!(dw, x, dy, cdims; alpha=1, beta=0)

Calculate the gradient imposed upon `w` in the depthwise convolution `y = x * w`.
"""
∇depthwiseconv_filter_direct!

function ∇depthwiseconv_filter_direct!(
                dw::AbstractArray{wT,5}, x::AbstractArray{xT,5},
                dy::AbstractArray{yT,5}, cdims::DepthwiseConvDims;
                alpha::wT=wT(1),beta=false) where {xT, yT, wT}
    # We do a separate convolution for each channel in x
    @inbounds for cidx in 1:channels_in(cdims)
        # For this batch and in-channel, we have a normal transposed convolution
        # between this slice of `x` and the corresponding slices of `w` and `dy`:
        x_slice = view(x, :, :, :, cidx:cidx, :)
        C_mult = channel_multiplier(cdims)
        dy_slice = view(dy, :, :, :, ((cidx-1)*C_mult + 1):cidx*C_mult, :)
        dw_slice = permutedims(view(dw, :, :, :, :, cidx:cidx), (1, 2, 3, 5, 4))

        # Adapt a DenseConvDims out of this DepthwiseConvDims, setting the in/out
        # channels appropriately for this one convolution.
        cdims_slice = DenseConvDims(cdims;
            C_in=1,
            C_out=channel_multiplier(cdims),
        )

        ∇conv_filter_direct!(dw_slice, x_slice, dy_slice, cdims_slice;
                                                alpha=alpha, beta=beta)
        dw[:, :, :, :, cidx:cidx] .= permutedims(dw_slice, (1, 2, 3, 5, 4))
    end
    return dw
end


