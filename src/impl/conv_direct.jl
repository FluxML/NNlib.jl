## This file contains direct Julia implementations of 2d and 3d convolutions

# Helper functions for restricting x/w overreach
function clamp_lo(x, w)
    idx = 1
    while idx <= length(x) && x[idx] <= 0
        idx += 1
    end
    return (x[idx:end], w[idx:end])
end
function clamp_hi(x, w, L)
    idx = length(x)
    while idx >= 1 && x[idx] > L
        idx -= 1
    end
    return (x[1:idx], w[1:idx])
end

"""
    conv_direct!(y, x, w, cdims; alpha=1, beta=0)

Direct convolution implementation; used for debugging, tests, and mixing/matching of
strange datatypes within a single convolution.  Uses naive nested for loop implementation
and does not attempt to optimize performance.  Rather, this implementation is intended to
be maximally understandable and debuggable, to aid in testing other, more performant
implementations.  We also explicitly support mixing and matching of strange datatypes,
so that if the user really wants to convolve an image of `UInt8`'s with a `Float16`
kernel, storing the result in a `Float32` output, there is at least a function call
for that madness.

The keyword arguments `alpha` and `beta` control accumulation behavior; this function
calculates `y = alpha * x * w + beta * y`, therefore by setting `beta` to a nonzero
value, the user is able to accumulate values into a preallocated `y` buffer, or by
setting `alpha` to a nonunitary value, an arbitrary gain factor can be applied.

By defaulting `beta` to `false`, we make use of the Bradbury promotion trick to override
`NaN`'s that may pre-exist within our output buffer, as `false*NaN == 0.0`, whereas
`0.0*NaN == NaN`.  Only set `beta` if you are certain that none of the elements within
`y` are `NaN`.

The basic implementation performs 3-dimensional convolution; 1-dimensional and 2-
dimensional casesa are supported by simply reshaping `y`, `x` and `w`, for which
wrapper methods are available.
"""
conv_direct!

@timeit_debug to function conv_direct!(y::AbstractArray{yT,5}, x::AbstractArray{xT,5},
                      w::AbstractArray{wT,5}, cdims::DenseConvDims;
                      alpha::yT = yT(1), beta = false) where {yT, xT, wT}
    check_dims(size(x), size(w), size(y), cdims)

    width, height, depth = input_size(cdims)
    kernel_w, kernel_h, kernel_d = kernel_size(cdims)
    out_c = channels_out(cdims)
    pad_w_lo, pad_w_hi, pad_h_lo, pad_h_hi, pad_d_lo, pad_d_hi = padding(cdims)
    dil_w, dil_h, dil_d = dilation(cdims)
    stride_w, stride_h, stride_d = stride(cdims)
    out_width, out_height, out_depth = output_size(cdims)
    
    # If we're doing crosscorr instead of conv, then don't bother to flip `w`
    if !flipkernel(cdims)
        w = w[end:-1:1, end:-1:1, end:-1:1, :, :]
    end

    # A helper function to project from output (w, h) to input (input_w, input_h)
    @inline project(idx, stride, pad) = (idx - 1)*stride - pad + 1
    
    # explicit formulation of convolution.  Oh hoisting gods, hear my plea.
    @inbounds for batch in 1:size(x)[end],
        c_out in 1:out_c,
        d_idx in 1:out_depth,
        h_idx in 1:out_height,
        w_idx in 1:out_width

        # Starting points of the window of x we're going to grab
        x_w = project(w_idx, stride_w, pad_w_lo)
        x_h = project(h_idx, stride_h, pad_h_lo)
        x_d = project(d_idx, stride_d, pad_d_lo)
        
        # Grow that starting point into ranges
        x_widxs = x_w .+ (0:dil_w:(dil_w*kernel_w-1))
        x_hidxs = x_h .+ (0:dil_h:(dil_h*kernel_h-1))
        x_didxs = x_d .+ (0:dil_d:(dil_d*kernel_d-1))
        w_widxs = 1:kernel_w
        w_hidxs = 1:kernel_h
        w_didxs = 1:kernel_d
        
        # Clamp the ranges to simulate padding
        x_widxs, w_widxs = clamp_lo(x_widxs, w_widxs)
        x_widxs, w_widxs = clamp_hi(x_widxs, w_widxs, width)
        x_hidxs, w_hidxs = clamp_lo(x_hidxs, w_hidxs)
        x_hidxs, w_hidxs = clamp_hi(x_hidxs, w_hidxs, height)
        x_didxs, w_didxs = clamp_lo(x_didxs, w_didxs)
        x_didxs, w_didxs = clamp_hi(x_didxs, w_didxs, depth)

        # Grab our slices
        x_slice = view(x, x_widxs, x_hidxs, x_didxs, :, batch)
        w_slice = view(w, w_widxs, w_hidxs, w_didxs, :, c_out)
        
        # Do the dotproduct dance, then weight by alpha/beta and git 'er done
        dotprod = sum(x_slice .* w_slice)
        y[w_idx, h_idx, d_idx, c_out, batch] = alpha*convert(yT, dotprod) +
                                               beta*y[w_idx, h_idx, d_idx, c_out, batch]
    end

    return y
end

## Gradient definitions
"""
    ∇conv_data_direct!(dx, dy, w, cdims; alpha=1, beta=0)

Calculate the gradient imposed upon `x` in the convolution `y = x * w`.
"""
∇conv_data_direct!

@timeit_debug to function ∇conv_data_direct!(dx::AbstractArray{xT,5}, dy::AbstractArray{yT,5},
                            w::AbstractArray{wT,5}, cdims::DenseConvDims;
                            alpha::xT=xT(1), beta=false) where {xT, yT, wT}
    w = transpose_swapbatch(w[end:-1:1, end:-1:1, end:-1:1, :, :])
    dy = predilate(dy, stride(cdims))
    ctdims = DenseConvDims(dy, w; padding=transpose_pad(cdims),
                                  dilation=dilation(cdims),
                                  flipkernel=flipkernel(cdims))
    dx = conv_direct!(dx, dy, w, ctdims; alpha=alpha, beta=beta)
    return transpose_swapbatch(dx)
end

"""
    ∇conv_filter_direct!(dw, x, dy, cdims; alpha=1, beta=0)

Calculate the gradient imposed upon `w` in the convolution `y = x * w`.
"""
∇conv_filter_direct!

@timeit_debug to function ∇conv_filter_direct!(dw::AbstractArray{wT,5}, x::AbstractArray{xT,5},
                              dy::AbstractArray{yT,5}, cdims::DenseConvDims;
                              alpha::wT=wT(1), beta=false) where {xT, yT, wT}
    x = transpose_swapbatch(x[end:-1:1, end:-1:1, end:-1:1, :, :])
    dy = transpose_swapbatch(predilate(dy, stride(cdims)))
    ctdims = DenseConvDims(dy, x; padding=transpose_pad(cdims),
                                    stride=dilation(cdims))
    conv_direct!(dw, dy, x, ctdims; alpha=alpha, beta=beta)
    if flipkernel(cdims)
        dw .= dw[end:-1:1, end:-1:1, end:-1:1, :, :]
    end
    return dw
end
