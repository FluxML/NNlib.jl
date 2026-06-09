## This file contains im2col-backed implementations of convolution for 2d and 3d
## convolutions.  Expect to see a lot of indexing.

# Helper function for flipkernel-induced dyslexia
function kernel_index(w, h, d, cdims::ConvDims)
    flipkernel(cdims) && return (w, h, d)
    kernel_w, kernel_h, kernel_d = kernel_size(cdims)
    return (kernel_w - w + 1, kernel_h - h + 1, kernel_d - d + 1)
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
function conv_im2col!(
                y::AbstractArray{T,5}, x::AbstractArray{T,5},
                w::AbstractArray{T,5}, cdims::DenseConvDims;
                col::AbstractArray{T,3}=similar(x, im2col_dims(cdims)),
                alpha::T=T(1), beta::T=T(0),
                ntasks::Int=nthreads()) where {T}
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

    nbatch = size(x, 5)
    ntasks = min(ntasks, size(col, 3))

    # Multiply a contiguous block of `m_len` output-spatial rows, starting at
    # 0-based row `m_off`, for image `n`, reading from workspace slice `cs`.
    # `cs` is the full `(M, K)` workspace; only rows `m_off .+ (1:m_len)` are
    # read, so several tasks can share `y`/`w` while writing disjoint rows.
    function gemm_block!(cs, n, m_off, m_len)
        GC.@preserve cs w y begin
            col_ptr = pointer(cs, m_off + 1)
            w_ptr   = pointer(w)
            y_ptr   = pointer(y, (n - 1)*M*N + m_off + 1)
            gemm!(Val(false), Val(false), m_len, N, K, alpha,
                  col_ptr, M, w_ptr, K, beta, y_ptr, M)
        end
    end

    # One task fully responsible for a chunk of whole images.
    function batch_task(task_n, part)
        cs = view(col, :, :, task_n)
        for n in part
            im2col!(cs, view(x, :, :, :, :, n), cdims)
            gemm_block!(cs, n, 0, M)
        end
    end

    # Serial path (single thread or spawning disabled), or enough images to keep
    # every task busy: process whole images, splitting the batch across tasks.
    # This is the original behavior and leaves the per-image GEMM to BLAS.
    if !should_use_spawn() || ntasks <= 1 || nbatch >= ntasks
        parts = Iterators.partition(1:nbatch, cld(nbatch, max(ntasks, 1)))
        if should_use_spawn() && ntasks > 1 && length(parts) > 1
            @sync for (task_n, part) in enumerate(parts)
                Threads.@spawn batch_task(task_n, part)
            end
        else
            for (task_n, part) in enumerate(parts)
                batch_task(task_n, part)
            end
        end
        return y
    end

    # Fewer images than tasks: split each image's output-spatial dimension so no
    # thread sits idle at small batch (issue #234). We drive this parallelism
    # ourselves and partition the GEMM by output rows, so pin BLAS to a single
    # thread to avoid oversubscription (as `batched_gemm!` does). We split the
    # outermost spatial axis with extent > 1, which keeps each task's output
    # rows contiguous in `y`.
    out_w, out_h, out_d = output_size(cdims)
    if out_d > 1
        naxis, stride_ax, axis = out_d, out_w*out_h, :d
    elseif out_h > 1
        naxis, stride_ax, axis = out_h, out_w, :h
    else
        naxis, stride_ax, axis = out_w, 1, :w
    end

    # Distribute ntasks spatial blocks across the nbatch images.
    base, extra = divrem(ntasks, nbatch)
    units = Tuple{Int,UnitRange{Int}}[]
    for n in 1:nbatch
        nblk = clamp(base + (n <= extra ? 1 : 0), 1, naxis)
        for cr in Iterators.partition(1:naxis, cld(naxis, nblk))
            push!(units, (n, cr))
        end
    end

    function tile_task(task_n, n, cr)
        cs = view(col, :, :, task_n)
        xn = view(x, :, :, :, :, n)
        if axis === :d
            im2col!(cs, xn, cdims; d_range=cr)
        elseif axis === :h
            im2col!(cs, xn, cdims; h_range=cr)
        else
            im2col!(cs, xn, cdims; w_range=cr)
        end
        gemm_block!(cs, n, (first(cr) - 1)*stride_ax, length(cr)*stride_ax)
    end

    old_blas = get_num_threads()
    set_num_threads(1)
    try
        @sync for (task_n, (n, cr)) in enumerate(units)
            Threads.@spawn tile_task(task_n, n, cr)
        end
    finally
        set_num_threads(old_blas)
    end
    return y
end

"""
    ∇conv_filter_im2col!(dw, x, dy, cdims, col=similar(dw, ∇filter_im2col_dims(cdims));
                         alpha=1, beta=0)

Conv backward pass onto the weights using im2col and GEMM; stores the result in `dw`.
See [`conv_im2col!`](@ref) for explanation of optional parameters.
"""
function ∇conv_filter_im2col!(
                dw::AbstractArray{T,5}, x::AbstractArray{T,5},
                dy::AbstractArray{T,5}, cdims::DenseConvDims;
                col::AbstractArray{T,3} = similar(dw, ∇filter_im2col_dims(cdims)),
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

    for batch_idx in 1:size(x,5)
        col_slice = view(col, :, :, 1)

        im2col!(col_slice, view(x, :, :, :, :, batch_idx), cdims)
        GC.@preserve col_slice dw dy begin
            col_ptr = pointer(col_slice)
            dy_ptr = pointer(dy,(batch_idx - 1)*K*N + 1)
            dw_ptr = pointer(dw)
            gemm!(Val(true), Val(false), M, N, K, alpha, col_ptr, dy_ptr, beta, dw_ptr)
        end

        # Because we accumulate over batches in this loop, we must set `beta` equal
        # to `1.0` from this point on.
        beta = T(1)
    end
    return dw
end

"""
    ∇conv_data_im2col!(dx, w, dy, cdims, col=similar(dx); alpha=1, beta=0)

Conv2d backward pass onto the input using im2col and GEMM; stores the result in `dx`.
See [`conv_im2col!`](@ref) for explanation of optional parameters.
"""
function ∇conv_data_im2col!(
                dx::AbstractArray{T,5}, dy::AbstractArray{T,5},
                w::AbstractArray{T,5}, cdims::DenseConvDims;
                col::AbstractArray{T,3} = similar(dx, im2col_dims(cdims)),
                alpha::T=T(1), beta::T=T(0),
                ntasks::Int=nthreads()) where {T}
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

    parts = Iterators.partition(axes(dx, 5), ceil(Int, size(dx, 5) / ntasks))

    function ∇conv_data_part(task_n, part)
        col_slice = col_slice = view(col, :, :, task_n) # col_slice is a task-local workspace
        for batch_idx in part
            GC.@preserve col_slice w dy begin
                dy_ptr = pointer(dy, (batch_idx - 1)*M*K + 1)
                w_ptr = pointer(w)
                col_ptr = pointer(col_slice)
                gemm!(Val(false), Val(true), M, N, K, alpha, dy_ptr, w_ptr, T(0), col_ptr)
            end
            col2im!(view(dx, :, :, :, :, batch_idx), col_slice, cdims, beta)
        end
    end
    if should_use_spawn() && length(parts) > 1
        @sync for (task_n, part) in enumerate(parts)
            Threads.@spawn ∇conv_data_part(task_n, part)
        end
    else
        for (task_n, part) in enumerate(parts)
            ∇conv_data_part(task_n, part)
        end
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
function im2col!(col::AbstractArray{T,2}, x::AbstractArray{T,4}, cdims::ConvDims;
                 w_range=nothing, h_range=nothing, d_range=nothing) where {T}
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

    # Optionally restrict the work to a sub-range of output positions, so that
    # several tasks can fill disjoint output-spatial slabs of `col` in parallel
    # (used to thread a single image; see `conv_im2col!`). Defaults span the
    # whole output, reproducing the unrestricted behavior.
    w_range = w_range === nothing ? (1:out_width)  : w_range
    h_range = h_range === nothing ? (1:out_height) : h_range
    d_range = d_range === nothing ? (1:out_depth)  : d_range
    @inline _isect(a, b) = max(first(a), first(b)):min(last(a), last(b))

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
    @inbounds for c in 1:C_in
        # Unpack "central region"
        w_region, h_region, d_region = central_region

        for kd in 1:kernel_d,
            kh in 1:kernel_h,
            kw in 1:kernel_w,
            d in _isect(d_region, d_range),
            h in _isect(h_region, h_range),
            w in _isect(w_region, w_range)

            input_kd = project(d, stride_d, pad_d_lo) + (kd - 1)*dil_d
            input_kh = project(h, stride_h, pad_h_lo) + (kh - 1)*dil_h
            input_kw = project(w, stride_w, pad_w_lo) + (kw - 1)*dil_w
            kidxs = kernel_index(kw, kh, kd, cdims)

            xval::T = x[input_kw, input_kh, input_kd, c]
            col_reshaped[w, h, d, kidxs..., c] = xval
        end
    end


    # For each "padded region", we run the fully general version
    @inbounds for (w_region, h_region, d_region) in padded_regions
        for c in 1:C_in,
            d in _isect(d_region, d_range),
            h in _isect(h_region, h_range),
            w in _isect(w_region, w_range),
            kd in 1:kernel_d,
            kh in 1:kernel_h,
            kw in 1:kernel_w

            input_kd = project(d, stride_d, pad_d_lo) + (kd - 1)*dil_d
            input_kh = project(h, stride_h, pad_h_lo) + (kh - 1)*dil_h
            input_kw = project(w, stride_w, pad_w_lo) + (kw - 1)*dil_w

            kidxs = kernel_index(kw, kh, kd, cdims)

            out_of_bounds = (
                input_kd <= 0 || input_kd > depth ||
                input_kh <= 0 || input_kh > height ||
                input_kw <= 0 || input_kw > width
            )
            if out_of_bounds
                col_reshaped[w, h, d, kidxs..., c] = T(0)
                continue
            end

            # Copy the data over
            xval::T = x[input_kw, input_kh, input_kd, c]
            col_reshaped[w, h, d, kidxs..., c] = xval
        end
    end
end


"""
    col2im!(x, col, cdims, beta=0)

Does the inverse of `im2col!()`, converting `col` back into a 3d image, used for backward
passes, transposed convolutions, etc...

Note that this method has not been optimized in the same way as `im2col()` has, because
it is slightly more complicated due to the more chaotic data access patterns, and I'm not
desperate enough yet.
"""
col2im!

function col2im!(x::AbstractArray{T,4}, col::AbstractArray{T,2}, cdims::ConvDims, beta::T=T(0)) where T
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
    if beta == T(0)
        fill!(x, T(0))
    elseif beta == T(1)
        # nothing
    else
        x .*= beta
    end

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
