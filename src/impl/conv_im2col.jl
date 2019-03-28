## This file contains im2col-backed implementations of convolution for 2d and 3d
## convolutions.  Expect to see a lot of indexing.

# Helper functions for flipkernel-induced dyslexia
@inline function kernel_index(w, h, d, cdims::ConvDims{N, S, P, D, false}) where {N, S, P, D}
    kernel_w, kernel_h, kernel_d = kernel_size(cdims)
    return (kernel_w - w + 1, kernel_h - h + 1, kernel_d - d + 1)
end
@inline function kernel_index(w, h, d, cdim::ConvDims{N, S, P, D, true}) where {N, S, P, D}
    return (w, h, d)
end

"""
    conv_im2col!(y, x, w, cdims, col=similar(x); alpha=1, beta=0)

Perform a convolution using im2col and GEMM, store the result in `y`.  The  kwargs
`alpha` and `beta` control accumulation behavior; internally this operation is
implemented as a matrix multiply that boils down to `y = alpha * x * w + beta * y`, thus
by setting `beta` to a nonzero value, multiple results can be accumulated into `y`, or
by setting `alpha` to a nonunitary value, various gain factors can be applied.

Note for the particularly performance-minded, you can provide a pre-allocated `col`,
which should eliminate any need for large allocations within this method.
"""
@timeit_debug to function conv_im2col!(
                y::AbstractArray{T,5}, x::AbstractArray{T,5},
                w::AbstractArray{T,5}, cdims::DenseConvDims;
                col::AbstractArray{T,2}=similar(x, im2col_dims(cdims)),
                alpha::T=T(1), beta::T=T(0)) where {T}
    check_dims(size(x), size(w), size(y), cdims)

    #   COL   *    W    ->    Y
    # [M x K] * [K x N] -> [M x N]
    #
    #  M: output spatial resolution
    #  N: output channels
    #  K: size of input "patch" (kernel size and input channels combined)
    #
    # In english, we're grabbing each input patch and laying them out along
    # the M dimension in `col`, so that the GEMM call below multiplies each
    # kernel (which is kernel_h * kernel_w * channels_in elments long) is
    # dotproducted with that input patch, effectively computing a convolution
    # in a somewhat memory-wasteful but easily-computed way (since we already
    # have an extremely highly-optimized GEMM call available in BLAS).
    M = prod(output_size(cdims))
    N = channels_out(cdims)
    K = prod(kernel_size(cdims))*channels_in(cdims)
    
    @inbounds for batch_idx in 1:size(x,5)
        # We invoke `@timeit_debug` on the outside of `im2col!()` because inference
        # doesn't like us putting it on the inside.
        @timeit_debug to "im2col!" im2col!(col, view(x, :, :, :, :, batch_idx), cdims)
        col_ptr = pointer(col)
        w_ptr = pointer(w)
        y_ptr = pointer(y, (batch_idx - 1)*M*N + 1)
        @timeit_debug to "gemm!" gemm!(Val(false), Val(false), M, N, K, alpha, col_ptr, w_ptr, beta, y_ptr)
    end
    return y
end

"""
    ∇conv_filter_im2col!(dw, x, dy, cdims, col=similar(dw); alpha=1, beta=0)

Conv backward pass onto the weights using im2col and GEMM; stores the result in `dw`.
See the documentation for `conv_im2col!()` for explanation of optional parameters.
"""
@timeit_debug to function ∇conv_filter_im2col!(
                dw::AbstractArray{T,5}, x::AbstractArray{T,5},
                dy::AbstractArray{T,5}, cdims::DenseConvDims;
                col::AbstractArray{T,2} = similar(dw, im2col_dims(cdims)),
                alpha::T=T(1), beta::T=T(0)) where {T}
    check_dims(size(x), size(dw), size(dy), cdims)

    #   COL'   *   dY   ->    dW
    # [M x K] * [K x N] -> [M x N]
    #
    #  M: size of input "patch" (kernel size and input channels combined)
    #  N: output channels
    #  K: output spatial resolution
    #
    # In english, we're grabbing each input patch and laying them out along
    # the K dimension in `col`, then multiplying in `dY` to compute a dot
    # product between all pixels in the input that were multiplied by a single
    # position in the W kernel, and all output pixels of the same location,
    # across output channels.  This slice of `col` therefore constitutes every
    # input pixel that touched a particular element of the kernel.
    #
    # This is identical to a convolution between x and a dimension-permuted dY,
    # where we 
    
    M = prod(kernel_size(cdims))*channels_in(cdims)
    N = channels_out(cdims)
    K = prod(output_size(cdims))
    
    @inbounds for batch_idx in 1:size(x,5)
        # We invoke `@timeit_debug` on the outside of `im2col!()` because inference
        # doesn't like us putting it on the inside.
        @timeit_debug to "im2col!" im2col!(col, view(x, :, :, :, :, batch_idx), cdims)
        col_ptr = pointer(col)
        dy_ptr = pointer(dy,(batch_idx - 1)*K*N + 1)
        dw_ptr = pointer(dw)
        @timeit_debug to "gemm!" gemm!(Val(true), Val(false), M, N, K, alpha, col_ptr, dy_ptr, beta, dw_ptr)

        # Because we accumulate over batches in this loop, we must set `beta` equal
        # to `1.0` from this point on.
        beta = T(1)
    end
    return dw
end

"""
    ∇conv_data_im2col!(dx, w, dy, cdims, col=similar(dx); alpha=1, beta=0)

Conv2d backward pass onto the input using im2col and GEMM; stores the result in `dx`.
See the documentation for `conv_im2col!()` for explanation of other parameters.
"""
@timeit_debug to function ∇conv_data_im2col!(
                dx::AbstractArray{T,5}, dy::AbstractArray{T,5},
                w::AbstractArray{T,5}, cdims::DenseConvDims;
                col::AbstractArray{T,2} = similar(dx, im2col_dims(cdims)),
                alpha::T=T(1), beta::T=T(0)) where {T}
    check_dims(size(dx), size(w), size(dy), cdims)

    #    dY        W'   ->    dX
    # [M x K] * [K x N] -> [M x N]
    #
    #  M: output spatial resolution
    #  N: size of input "patch" (kernel size and input channels combined)
    #  K: output channels
    #
    # In english, we're taking the output image and laying it out by pixel,
    # with channels lying along the `K` dimension in `col`.  We then multiply
    # in `W'` to compute a dot product between each pixel location and the
    # entire kernel.  This dot product therefore constitutes every output pixel
    # that was a function of a particular input pixel.
    #
    # This is identical to a transposed convolution between dY and W

    M = prod(output_size(cdims))
    N = prod(kernel_size(cdims))*channels_in(cdims)
    K = channels_out(cdims)

    @inbounds for batch_idx in 1:size(dx, 5)
        dy_ptr = pointer(dy, (batch_idx - 1)*M*K + 1)
        w_ptr = pointer(w)
        col_ptr = pointer(col)
        @timeit_debug to "gemm!" gemm!(Val(false), Val(true), M, N, K, alpha, dy_ptr, w_ptr, T(0), col_ptr)
        @timeit_debug to "col2im!" col2im!(view(dx, :, :, :, :, batch_idx), col, cdims)
    end
    return dx
end





"""
    im2col!(col, x, cdims)

Converts a 3d image `x` into a matrix `col` for usage with GEMM-calculated convolution.
Patches of `x` of size (kernel_w, kernel_h, kernel_d, C_in) will be extracted and laid
out along the rows of `col`, one for each output pixel.  This routine is used by all
im2col-based convolutions, just with extra singleton dimensions added in the case of `2d`
or `1d` images.
"""
function im2col!(col::AbstractArray{T,2}, x::AbstractArray{T,4},
                                          cdims::ConvDims) where {T}
    if spatial_dims(cdims) != 3
        throw(DimensionMismatch("im2col!() only accepts 3d convoluitional inputs"))
    end

    # Extract those nice, compile-time constant type parameters from `cdims`.
    width, height, depth = input_size(cdims)
    kernel_w, kernel_h, kernel_d = kernel_size(cdims)
    C_in = channels_in(cdims)
    pad_w_lo, pad_w_hi, pad_h_lo, pad_h_hi, pad_d_lo, pad_d_hi = padding(cdims)
    dil_w, dil_h, dil_d = dilation(cdims)
    stride_w, stride_h, stride_d = stride(cdims)
    out_width, out_height, out_depth = output_size(cdims)

    # Reshape col for easy access.
    col_reshaped = reshape(col, (
        # Output resolution
        out_width,
        out_height,
        out_depth,
        
        # By input patch size
        kernel_w,
        kernel_h,
        kernel_d,
        C_in,
    ))

    padded_regions, central_region = calc_padding_regions(cdims)

    # A helper function to project from output (w, h) to input (input_w, input_h)
    @inline project(idx, stride, pad) = (idx - 1)*stride - pad + 1

    
    # We begin by copying the central region of the image which requires no padding at all.
    # Eliminating the branches of the fully generalized version below gives us a nice
    # speedup on the majority of the data.
    @timeit_debug to "im2col!() - central region" begin
        @inbounds for c in 1:C_in
            # Unpack "central region"
            w_region, h_region, d_region = central_region

            for kd in 1:kernel_d,
                kh in 1:kernel_h,
                kw in 1:kernel_w,
                d in d_region,
                h in h_region,
                w in w_region

                input_kd = project(d, stride_d, pad_d_lo) + (kd - 1)*dil_d
                input_kh = project(h, stride_h, pad_h_lo) + (kh - 1)*dil_h
                input_kw = project(w, stride_w, pad_w_lo) + (kw - 1)*dil_w
                kidxs = kernel_index(kw, kh, kd, cdims)

                xval::T = x[input_kw, input_kh, input_kd, c]
                col_reshaped[w, h, d, kidxs..., c] = xval
            end
        end
    end
    
    # For each "padded region", we run the fully general version
    @timeit_debug to "im2col!() - padded region" begin
        @inbounds for (w_region, h_region, d_region) in padded_regions
            for c in 1:C_in,
                d in d_region,
                h in h_region,
                w in w_region,
                kd in 1:kernel_d,
                kh in 1:kernel_h,
                kw in 1:kernel_w

                input_kd = project(d, stride_d, pad_d_lo) + (kd - 1)*dil_d
                input_kh = project(h, stride_h, pad_h_lo) + (kh - 1)*dil_h
                input_kw = project(w, stride_w, pad_w_lo) + (kw - 1)*dil_w

                kidxs = kernel_index(kw, kh, kd, cdims)

                # If this d is off the edge, then deal with the entire plane
                # in one fell swoop, like a ravenous flock of crows.  CAW CAW.
                if input_kd <= 0 || input_kd > depth
                    for kh in 1:kernel_h,
                        kw in 1:kernel_w
                        col_reshaped[w, h, d, kidxs..., c] = T(0)
                    end
                    continue
                end

                # Same for `h`, but in this case it's only a line, not a plane.
                # This results in slightly less caw'ing.
                if input_kh <= 0 || input_kh > height
                    for kw in 1:kernel_w
                        col_reshaped[w, h, d, kidxs..., c] = T(0)
                    end
                    continue
                end

                # If this `w` is off the edge it and only it gets cleared out
                if input_kw <= 0 || input_kw > width
                    col_reshaped[w, h, d, kidxs..., c] = T(0)
                    continue
                end

                # Copy the data over
                xval::T = x[input_kw, input_kh, input_kd, c]
                col_reshaped[w, h, d, kidxs..., c] = xval
            end
        end
    end
end


"""
    col2im!(x, col, cdims)

Does the inverse of `im2col!()`, converting `col` back into a 3d image, used for backward
passes, transposed convolutions, etc...

Note that this method has not been optimized in the same way as `im2col()` has, because
it is slightly more complicated due to the more chaotic data access patterns, and I'm not
desperate enough yet.
"""
col2im!

function col2im!(x::AbstractArray{T,4}, col::AbstractArray{T,2},
                                  cdims::ConvDims) where T
    if spatial_dims(cdims) != 3
        throw(DimensionMismatch("col2im!() only accepts 3d convoluitional inputs"))
    end

    # Extract those nice, compile-time constant type parameters from `cdims`.
    width, height, depth = input_size(cdims)
    kernel_w, kernel_h, kernel_d = kernel_size(cdims)
    C_in = channels_in(cdims)
    pad_w_lo, pad_w_hi, pad_h_lo, pad_h_hi, pad_d_lo, pad_d_hi = padding(cdims)
    dil_w, dil_h, dil_d = dilation(cdims)
    stride_w, stride_h, stride_d = stride(cdims)
    out_width, out_height, out_depth = output_size(cdims)
    
    # TODO: Rewrite this method so we don't have this fill!() at the beginning!
    # Calculate each output pixel once rather than accumulating into it?
    fill!(x, T(0))

    # Reshape col for easy access.
    col_reshaped = reshape(col, (
        # Output resolution
        out_width,
        out_height,
        out_depth,
        
        # By input patch size
        kernel_w,
        kernel_h,
        kernel_d,
        C_in,
    ))

    # A helper function to project from output (w, h) to input (input_w, input_h)
    @inline project(idx, stride, pad) = (idx - 1)*stride - pad + 1

    @inbounds for c in 1:C_in
        for kd in 1:kernel_d,
            kh in 1:kernel_h,
            kw in 1:kernel_w

            for d in 1:out_depth
                input_kd = project(d, stride_d, pad_d_lo) + (kd - 1)*dil_d
                
                # If this d is off the edge, then deal with the entire plane
                # in one fell swoop, like a ravenous flock of crows.  CAW CAW.
                if input_kd <= 0 || input_kd > depth
                    continue
                end

                for h in 1:out_height
                    input_kh = project(h, stride_h, pad_h_lo) + (kh - 1)*dil_h
                    
                    # Same for `h`, but in this case it's only a line, not a plane.
                    # This results in slightly less caw'ing.
                    if input_kh <= 0 || input_kh > height
                        continue
                    end
                
                    for w in 1:out_width
                        input_kw = project(w, stride_w, pad_w_lo) + (kw - 1)*dil_w
                
                        # If this `w` is off the edge, only it gets cleared out.
                        if input_kw <= 0 || input_kw > width
                            continue
                        end
                        
                        # Copy the data over
                        kidxs = kernel_index(kw, kh, kd, cdims)
                        cval::T = col_reshaped[w, h, d, kidxs..., c]
                        x[input_kw, input_kh, input_kd, c] += cval
                    end
                end
            end
        end
    end
end
