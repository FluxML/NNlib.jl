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
dimensional cases are supported by simply reshaping `y`, `x` and `w`, for which
wrapper methods are available.
"""
conv_direct!

function conv_direct!(y::AbstractArray{yT,5}, x::AbstractArray{xT,5},
                      w::AbstractArray{wT,5}, cdims::DenseConvDims;
                      alpha::yT = yT(1), beta = false) where {yT, xT, wT}
    conv_direct!(
        y, x, w, cdims,
        Val(kernel_size(cdims)), Val(channels_out(cdims)),
        Val(padding(cdims)), Val(dilation(cdims)), Val(stride(cdims)),
        Val(flipkernel(cdims)); alpha, beta)
    return y
end


function conv_direct!(
    y::AbstractArray{yT,5}, x::AbstractArray{xT,5},
    w::AbstractArray{wT,5}, cdims::DenseConvDims,
    # kernel size, output channels, padding, dilation, stride, flipped kernel
    ::Val{K}, ::Val{C}, ::Val{P}, ::Val{D}, ::Val{S}, fk::Val{F};
    alpha::yT = yT(1), beta = false,
) where {yT, xT, wT, K, C, P, D, S, F}
    check_dims(size(x), size(w), size(y), cdims)

    width, height, depth = input_size(cdims)
    kernel_w, kernel_h, kernel_d = K
    pad_w_lo, _, pad_h_lo, _, pad_d_lo, _ = P
    dil_w, dil_h, dil_d = D
    stride_w, stride_h, stride_d = S

    # Create a method that determines how we're going to index into `w`.
    kproj(k, _, ::Val{true}) = k
    kproj(k, M, ::Val{false}) = M - k + 1

    # A helper function to project from output (w, h) to input (input_w, input_h)
    project(idx, stride, pad) = (idx - 1)*stride - pad + 1

    # Use `calc_padding_regions` to determine where we do or don't need to worry about padding
    padded_regions, central_region = calc_padding_regions(cdims)

    # Set outputs to zero to support custom datatypes (https://github.com/FluxML/NNlib.jl/issues/490)
    if iszero(beta)
        y = fill!(y, zero(yT))
    end

    # Start with the central region
    w_region, h_region, d_region = central_region
    @inbounds for batch in 1:size(x, 5),
        c_out in 1:C,
        d_idx in d_region,
        h_idx in h_region,
        w_idx in w_region

        # Since we're in the central region, we don't need to worry about clamping
        dotprod = yT(0)
        for c_in in 1:channels_in(cdims),
            kd in 1:kernel_d,
            kh in 1:kernel_h,
            kw in 1:kernel_w

            # Hoist me, you coward.
            x_d = project(d_idx, stride_d, pad_d_lo) + (kd - 1)*dil_d
            x_h = project(h_idx, stride_h, pad_h_lo) + (kh - 1)*dil_h
            x_w = project(w_idx, stride_w, pad_w_lo) + (kw - 1)*dil_w

            x_val = x[x_w, x_h, x_d, c_in, batch]
            w_val = w[kproj(kw, kernel_w, fk),
                    kproj(kh, kernel_h, fk),
                    kproj(kd, kernel_d, fk),
                    c_in, c_out]
            dotprod = muladd(x_val, w_val, dotprod)
        end
        y[w_idx, h_idx, d_idx, c_out, batch] = alpha*dotprod + beta*y[w_idx, h_idx, d_idx, c_out, batch]
    end

    # Next, do potentially-padded regions:
    @inbounds for (w_region, h_region, d_region) in padded_regions,
        batch in 1:size(x, 5),
        c_out in 1:C,
        d_idx in d_region,
        h_idx in h_region,
        w_idx in w_region

        # Probe for out-of-bounds accesses on `x` and `continue` if we hit one
        dotprod = yT(0)
        for c_in in 1:channels_in(cdims),
            kd in 1:kernel_d

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
                    w_val = w[kproj(kw, kernel_w, fk),
                            kproj(kh, kernel_h, fk),
                            kproj(kd, kernel_d, fk),
                            c_in, c_out]
                    dotprod = muladd(x_val, w_val, dotprod)
                end
            end
        end

        y[w_idx, h_idx, d_idx, c_out, batch] = alpha*dotprod + beta*y[w_idx, h_idx, d_idx, c_out, batch]
    end

    return y
end

## Gradient definitions
"""
    ∇conv_data_direct!(dx, dy, w, cdims; alpha=1, beta=0)

Calculate the gradient imposed upon `x` in the convolution `y = x * w`.
"""
∇conv_data_direct!

function ∇conv_data_direct!(dx::AbstractArray{xT,5}, dy::AbstractArray{yT,5},
                            w::AbstractArray{wT,5}, cdims::DenseConvDims;
                            alpha::xT=xT(1), beta=false) where {xT, yT, wT}
    w = conj(transpose_swapbatch(w[end:-1:1, end:-1:1, end:-1:1, :, :]))
    dy = predilate(dy, stride(cdims))
    ctdims = DenseConvDims(dy, w; padding=transpose_pad(cdims),
                                  dilation=dilation(cdims),
                                  flipkernel=flipkernel(cdims))
    dx = conv_direct!(dx, dy, w, ctdims; alpha=alpha, beta=beta)
    return dx
end

"""
    ∇conv_filter_direct!(dw, x, dy, cdims; alpha=1, beta=0)

Calculate the gradient imposed upon `w` in the convolution `y = x * w`.
"""
∇conv_filter_direct!

function ∇conv_filter_direct!(dw::AbstractArray{wT,5}, x::AbstractArray{xT,5},
                              dy::AbstractArray{yT,5}, cdims::DenseConvDims;
                              alpha::wT=wT(1), beta=false) where {xT, yT, wT}
    x = conj(transpose_swapbatch(x[end:-1:1, end:-1:1, end:-1:1, :, :]))
    dy = transpose_swapbatch(predilate(dy, stride(cdims)))
    ctdims = DenseConvDims(dy, x; padding=transpose_pad(cdims),
                                    stride=dilation(cdims))
    dw_ = if flipkernel(cdims)
        view(dw, reverse(axes(dw, 1)), reverse(axes(dw, 2)), reverse(axes(dw, 3)), :, :)
    else
        dw
    end
    conv_direct!(dw_, dy, x, ctdims; alpha=alpha, beta=beta)
    return dw
end
