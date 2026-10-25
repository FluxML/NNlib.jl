## This file contains im2col-backed implementations of convolution for 2d and 3d
## convolutions.  Expect to see a lot of indexing.

# Helper function for flipkernel-induced dyslexia
function kernel_index(w, h, d, cdims::ConvDims)
    flipkernel(cdims) && return (w, h, d)
    kernel_w, kernel_h, kernel_d = kernel_size(cdims)
    return (kernel_w - w + 1, kernel_h - h + 1, kernel_d - d + 1)
end

# The im2col matrix of an image has one row per output position, too many to hold once
# per task on large images (#864). It is filled and multiplied in tiles of consecutive
# rows instead, each task reusing one `(rows, K)` slice of the workspace `col`. A tile
# gets `IM2COL_TILE_BYTES` per thread working on it: tasks pin BLAS to one thread and
# keep their tiles in cache, while a lone task leaves its GEMMs to BLAS threads, which
# need bigger ones. The floor amortizes BLAS packing the other operand for every tile.
const IM2COL_TILE_BYTES = 2^19
const IM2COL_MIN_TILE_ROWS = 64

im2col_tile_rows(M, K, ::Type{T}, nthreads) where {T} =
    min(max(nthreads * IM2COL_TILE_BYTES ÷ (K * sizeof(T)), IM2COL_MIN_TILE_ROWS), M)

# Tasks for an im2col op on `cdims` over `nbatch` images: at most `ntasks`, each with at
# least `IM2COL_MIN_TASK_WORK` multiply-adds to pay for spawning it.
const IM2COL_MIN_TASK_WORK = 2^21

function im2col_ntasks(cdims::ConvDims, nbatch, ntasks)
    should_use_spawn() || return 1
    work = nbatch * prod(output_size(cdims)) * prod(kernel_size(cdims)) *
           channels_in(cdims) * channels_out(cdims)
    return clamp(cld(work, IM2COL_MIN_TASK_WORK), 1, ntasks)
end

# Workspace of `conv_im2col!`, whose tasks share the tiles of all images: these are cut
# finely enough to give every task work when there are fewer images than tasks.
function conv_im2col_dims(cdims::ConvDims, ::Type{T}, nbatch, ntasks) where {T}
    M = prod(output_size(cdims))
    K = prod(kernel_size(cdims)) * channels_in(cdims)
    ntasks = im2col_ntasks(cdims, nbatch, ntasks)
    rows = im2col_tile_rows(M, K, T, 1)
    if ntasks > nbatch > 0
        rows = min(rows, cld(M, cld(ntasks, nbatch)))
    end
    nt = clamp(nbatch * cld(M, rows), 1, ntasks)
    return nt > 1 ? (rows, K, nt) : (im2col_tile_rows(M, K, T, get_num_threads()), K, 1)
end

# `∇conv_data_im2col!` threads over slabs of `blk` indices along the outermost output
# axis `ax` with extent > 1. `col2im!` adds each slab into the inputs under it, so slabs
# are long enough that two slabs apart never add into the same input.
function ∇conv_data_im2col_slabs(cdims::ConvDims, nbatch, ntasks)
    out = output_size(cdims)
    ax = out[3] > 1 ? 3 : out[2] > 1 ? 2 : 1
    ntasks = im2col_ntasks(cdims, nbatch, ntasks)
    # The odd and the even slabs run in turn, so twice as many keep every task busy.
    nslab = ntasks > nbatch > 0 ? 2 * cld(ntasks, nbatch) : 1
    halo = fld((kernel_size(cdims)[ax] - 1) * dilation(cdims)[ax], stride(cdims)[ax])
    blk = max(halo, cld(out[ax], nslab), 1)
    return ax, blk, cld(out[ax], blk)
end

axis_stride(out, ax) = ax == 1 ? 1 : ax == 2 ? out[1] : out[1]*out[2]

function ∇conv_data_im2col_dims(cdims::ConvDims, ::Type{T}, nbatch, ntasks) where {T}
    out = output_size(cdims)
    K = prod(kernel_size(cdims)) * channels_in(cdims)
    ax, blk, nslab = ∇conv_data_im2col_slabs(cdims, nbatch, ntasks)
    nt = clamp(nbatch * cld(nslab, 2), 1, im2col_ntasks(cdims, nbatch, ntasks))
    rows = im2col_tile_rows(blk * axis_stride(out, ax), K, T, nt > 1 ? 1 : get_num_threads())
    return rows, K, nt
end

# `∇conv_filter_im2col!` threads over the tiles of all images, each task adding into its
# own copy of `dw`. Each task gets at least as many output positions as `dw` has columns
# (and `IM2COL_MIN_TASK_ROWS`), so the copies take less memory than the im2col matrix of
# the batch and summing them costs little next to the GEMMs.
const IM2COL_MIN_TASK_ROWS = 256

function ∇conv_filter_im2col_dims(cdims::ConvDims, ::Type{T}, nbatch, ntasks) where {T}
    M = prod(output_size(cdims))
    K = prod(kernel_size(cdims)) * channels_in(cdims)
    nt = clamp(nbatch * M ÷ max(IM2COL_MIN_TASK_ROWS, channels_out(cdims)), 1,
               im2col_ntasks(cdims, nbatch, ntasks))
    nt == 1 && return im2col_tile_rows(M, K, T, get_num_threads()), K, 1
    rows = im2col_tile_rows(M, K, T, 1)
    if nt > nbatch
        rows = min(rows, cld(M, cld(nt, nbatch)))
    end
    return rows, K, clamp(nbatch * cld(M, rows), 1, nt)
end

# Splits `1:nunits` into contiguous chunks for up to `nt` tasks, calling `f(i, chunk)` for
# the `i`-th, and returns the number of chunks.
function foreach_im2col_chunk(f, nunits, nt)
    if !should_use_spawn() || nt <= 1 || nunits <= 1
        f(1, 1:nunits)
        return 1
    end
    chunks = Iterators.partition(1:nunits, cld(nunits, nt))
    with_single_threaded_blas() do
        @sync for (i, us) in enumerate(chunks)
            Threads.@spawn f(i, us)
        end
    end
    return length(chunks)
end

# Tasks that each run their own GEMMs pin BLAS to one thread to avoid oversubscription.
function with_single_threaded_blas(f)
    old = get_num_threads()
    set_num_threads(1)
    try
        return f()
    finally
        set_num_threads(old)
    end
end

"""
    conv_im2col!(y, x, w, cdims; col=nothing, alpha=1, beta=0, ntasks=nthreads())

Perform a convolution using im2col and GEMM, store the result in `y`.  The  kwargs
`alpha` and `beta` control accumulation behavior; internally this operation is
implemented as a matrix multiply that boils down to `y = alpha * x * w + beta * y`, thus
by setting `beta` to a nonzero value, multiple results can be accumulated into `y`, or
by setting `alpha` to a nonunitary value, various gain factors can be applied.

Note for the particularly performance-minded, you can provide a pre-allocated workspace
`col` of size `(rows, K, ntasks)`, see `conv_im2col_dims`; each of up to `size(col, 3)`
tasks processes the output positions `rows` at a time.
"""
function conv_im2col!(
                y::AbstractArray{T,5}, x::AbstractArray{T,5},
                w::AbstractArray{T,5}, cdims::DenseConvDims;
                col::Union{Nothing,AbstractArray{T,3}}=nothing,
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
    # Zero-sized batch: `y` already has the correct (…, 0) shape and no elements
    # to write. See issue #760.
    nbatch == 0 && return y

    ws_rows, _, nt = conv_im2col_dims(cdims, T, nbatch, ntasks)
    col = col === nothing ? similar(x, ws_rows, K, nt) : col
    rows = min(ws_rows, size(col, 1))
    ntiles = cld(M, rows)

    # Unit `u` fills and multiplies the output positions `m` of image `n`; units write
    # disjoint rows of `y`.
    foreach_im2col_chunk(nbatch*ntiles, min(nt, size(col, 3))) do i, us
        ws = view(col, :, :, i)
        for u in us
            n, t = fldmod1(u, ntiles)
            m = (t - 1)*rows + 1 : min(t*rows, M)
            im2col!(ws, view(x, :, :, :, :, n), cdims, m)
            GC.@preserve ws w y begin
                gemm!(Val(false), Val(false), length(m), N, K, alpha,
                      pointer(ws), Base.stride(ws, 2), pointer(w), K, beta,
                      pointer(y, (n - 1)*M*N + first(m)), M)
            end
        end
    end
    return y
end

"""
    ∇conv_filter_im2col!(dw, x, dy, cdims; col=nothing, alpha=1, beta=0, ntasks=nthreads())

Conv backward pass onto the weights using im2col and GEMM; stores the result in `dw`.
See [`conv_im2col!`](@ref) for explanation of optional parameters.
"""
function ∇conv_filter_im2col!(
                dw::AbstractArray{T,5}, x::AbstractArray{T,5},
                dy::AbstractArray{T,5}, cdims::DenseConvDims;
                col::Union{Nothing,AbstractArray{T,3}}=nothing,
                alpha::T=T(1), beta::T=T(0),
                ntasks::Int=nthreads()) where {T}
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

    # Zero-sized batch: nothing to accumulate, so the result is just `beta*dw`.
    # Unlike the forward/data paths, `dw` is not empty here, and the loop below
    # would otherwise leave it untouched (uninitialized when beta==0). Mirror
    # GEMM's write-only semantics for beta==0 by zeroing it outright. See #760.
    if size(x, 5) == 0
        return iszero(beta) ? fill!(dw, zero(T)) : (dw .*= beta)
    end

    nbatch = size(x, 5)
    ws_rows, _, nt = ∇conv_filter_im2col_dims(cdims, T, nbatch, ntasks)
    col = col === nothing ? similar(dw, ws_rows, M, nt) : col
    rows = min(ws_rows, size(col, 1))
    ntiles = cld(K, rows)
    nunits = nbatch*ntiles

    # Adds the tiles `us` into the `M × N` matrix at `acc`, scaled by `b` first.
    function tiles!(acc, b, ws, us)
        for u in us
            n, t = fldmod1(u, ntiles)
            ks = (t - 1)*rows + 1 : min(t*rows, K)
            im2col!(ws, view(x, :, :, :, :, n), cdims, ks)
            GC.@preserve ws dy begin
                gemm!(Val(true), Val(false), M, N, length(ks), alpha,
                      pointer(ws), Base.stride(ws, 2), pointer(dy, (n - 1)*K*N + first(ks)), K,
                      b, acc, M)
            end
            b = T(1)
        end
    end
    nt = min(nt, size(col, 3))
    if nt == 1
        GC.@preserve dw tiles!(pointer(dw), beta, view(col, :, :, 1), 1:nunits)
        return dw
    end

    # Task `i > 1` adds into its own copy of `dw`, summed into `dw` at the end.
    partials = similar(dw, M, N, nt - 1)
    nchunks = foreach_im2col_chunk(nunits, nt) do i, us
        GC.@preserve dw partials begin
            acc = i == 1 ? pointer(dw) : pointer(partials, (i - 2)*M*N + 1)
            tiles!(acc, i == 1 ? beta : T(0), view(col, :, :, i), us)
        end
    end
    if nchunks > 1
        foreach_im2col_chunk(M*N, nt) do _, ls
            @inbounds for j in 1:nchunks-1, l in ls
                dw[l] += partials[l + (j - 1)*M*N]
            end
        end
    end
    return dw
end

"""
    ∇conv_data_im2col!(dx, w, dy, cdims; col=nothing, alpha=1, beta=0, ntasks=nthreads())

Conv2d backward pass onto the input using im2col and GEMM; stores the result in `dx`.
See [`conv_im2col!`](@ref) for explanation of optional parameters.
"""
function ∇conv_data_im2col!(
                dx::AbstractArray{T,5}, dy::AbstractArray{T,5},
                w::AbstractArray{T,5}, cdims::DenseConvDims;
                col::Union{Nothing,AbstractArray{T,3}}=nothing,
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

    nbatch = size(dx, 5)
    # Zero-sized batch: `dx` already has the correct (…, 0) shape and no elements
    # to write. See #760.
    nbatch == 0 && return dx

    ws_rows, _, nt = ∇conv_data_im2col_dims(cdims, T, nbatch, ntasks)
    col = col === nothing ? similar(dx, ws_rows, N, nt) : col
    rows = min(ws_rows, size(col, 1))
    ax, blk, nslab = ∇conv_data_im2col_slabs(cdims, nbatch, ntasks)
    out_len = output_size(cdims)[ax]
    in_len = input_size(cdims)[ax]
    stride_ax = axis_stride(output_size(cdims), ax)
    k_ax, s_ax, dil_ax = kernel_size(cdims)[ax], stride(cdims)[ax], dilation(cdims)[ax]
    pad_ax = padding(cdims)[2ax - 1]
    # Last input index along `ax` that slab `j` adds into.
    last_input(j) = clamp((min(j*blk, out_len) - 1)*s_ax - pad_ax + 1 + (k_ax - 1)*dil_ax,
                          0, in_len)

    function slab!(ws, n, j)
        dxn = view(dx, :, :, :, :, n)
        if isodd(j)
            # Scale by `beta` the inputs from past the previous odd slab's up to the end
            # of this one's (or of the image): all of `dx`, before the even slabs run.
            lo = j == 1 ? 1 : last_input(j - 2) + 1
            hi = j + 2 > nslab ? in_len : last_input(j)
            region = view(dxn, ntuple(i -> i == ax ? (lo:hi) : (1:size(dxn, i)), 4)...)
            if iszero(beta)
                fill!(region, zero(T))
            elseif !isone(beta)
                region .*= beta
            end
        end
        ms = (j - 1)*blk*stride_ax + 1 : min(j*blk, out_len)*stride_ax
        for m0 in first(ms):rows:last(ms)
            m = m0:min(m0 + rows - 1, last(ms))
            GC.@preserve ws w dy begin
                gemm!(Val(false), Val(true), length(m), N, K, alpha,
                      pointer(dy, (n - 1)*M*K + first(m)), M, pointer(w), N,
                      T(0), pointer(ws), Base.stride(ws, 2))
            end
            col2im!(dxn, ws, cdims, T(1), m)
        end
    end
    # Slabs two apart add into disjoint inputs: the odd slabs of all images run in
    # parallel, then the even ones.
    for parity in 1:2
        per = fld(nslab - parity, 2) + 1
        foreach_im2col_chunk(nbatch*per, min(nt, size(col, 3))) do i, us
            ws = view(col, :, :, i)
            for u in us
                n, j = fldmod1(u, per)
                slab!(ws, n, parity + 2(j - 1))
            end
        end
    end
    return dx
end


# Output positions `first:last` of an image, flattened over `(w, h, d)`, which a tile
# holds in its rows `1:last-first+1`. Its output lines `(:, h, d)` run from
# `(h_lo, d_lo)` to `(h_hi, d_hi)`.
struct Im2colTile
    first::Int
    last::Int
    out_w::Int
    out_h::Int
    h_lo::Int
    d_lo::Int
    h_hi::Int
    d_hi::Int
end

function Im2colTile(rows::UnitRange{Int}, out_w, out_h)
    l_lo, l_hi = fld1(first(rows), out_w), fld1(last(rows), out_w)
    return Im2colTile(first(rows), last(rows), out_w, out_h,
                      mod1(l_lo, out_h), fld1(l_lo, out_h), mod1(l_hi, out_h), fld1(l_hi, out_h))
end

_isect(a, b) = max(first(a), first(b)):min(last(a), last(b))

# The parts of the ranges of an output region in the tile, and the tile row of `(w, h, d)`.
@inline tile_ds(t::Im2colTile, ds) = _isect(ds, t.d_lo:t.d_hi)
@inline tile_hs(t::Im2colTile, hs, d) =
    _isect(hs, (d == t.d_lo ? t.h_lo : 1):(d == t.d_hi ? t.h_hi : t.out_h))
@inline tile_row(t::Im2colTile, w, h, d) = w + (h - 1 + (d - 1)*t.out_h)*t.out_w - t.first + 1
@inline function tile_ws(t::Im2colTile, ws, h, d)
    row0 = tile_row(t, 0, h, d)
    return _isect(ws, (1 - row0):(t.last - t.first + 1 - row0))
end

"""
    im2col!(col, x, cdims, rows=1:M)

Converts a 3d image `x` into a matrix `col` for usage with GEMM-calculated convolution.
Patches of `x` of size (kernel_w, kernel_h, kernel_d, C_in) will be extracted and laid
out along the rows of `col`, one for each output pixel.  Only the output pixels `rows`
(out of the `M` of the image, flattened) are extracted, into the first `length(rows)`
rows of `col`.  This routine is used by all im2col-based convolutions, just with extra
singleton dimensions added in the case of `2d` or `1d` images.
"""
function im2col!(col::AbstractArray{T,2}, x::AbstractArray{T,4}, cdims::ConvDims,
                 rows::UnitRange{Int}=1:prod(output_size(cdims))) where {T}
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
    tile = Im2colTile(rows, out_width, out_height)

    # Reshape col for easy access, by output pixel and input patch element.
    col_reshaped = reshape(col, (size(col, 1), kernel_w, kernel_h, kernel_d, C_in))

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
            d in tile_ds(tile, d_region),
            h in tile_hs(tile, h_region, d),
            w in tile_ws(tile, w_region, h, d)

            input_kd = project(d, stride_d, pad_d_lo) + (kd - 1)*dil_d
            input_kh = project(h, stride_h, pad_h_lo) + (kh - 1)*dil_h
            input_kw = project(w, stride_w, pad_w_lo) + (kw - 1)*dil_w
            kidxs = kernel_index(kw, kh, kd, cdims)

            xval::T = x[input_kw, input_kh, input_kd, c]
            col_reshaped[tile_row(tile, w, h, d), kidxs..., c] = xval
        end
    end


    # For each "padded region", we run the fully general version
    @inbounds for (w_region, h_region, d_region) in padded_regions
        for c in 1:C_in,
            d in tile_ds(tile, d_region),
            h in tile_hs(tile, h_region, d),
            w in tile_ws(tile, w_region, h, d),
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
                col_reshaped[tile_row(tile, w, h, d), kidxs..., c] = T(0)
                continue
            end

            # Copy the data over
            xval::T = x[input_kw, input_kh, input_kd, c]
            col_reshaped[tile_row(tile, w, h, d), kidxs..., c] = xval
        end
    end
end


"""
    col2im!(x, col, cdims, beta=0, rows=1:M)

Does the inverse of `im2col!()`, converting `col` back into a 3d image, used for backward
passes, transposed convolutions, etc...  `x` is first scaled by `beta`, then the output
pixels `rows`, held in the first `length(rows)` rows of `col`, are added into it.

Note that this method has not been optimized in the same way as `im2col()` has, because
it is slightly more complicated due to the more chaotic data access patterns, and I'm not
desperate enough yet.
"""
col2im!

function col2im!(x::AbstractArray{T,4}, col::AbstractArray{T,2}, cdims::ConvDims,
                 beta::T=T(0), rows::UnitRange{Int}=1:prod(output_size(cdims))) where T
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
    tile = Im2colTile(rows, out_width, out_height)

    # TODO: Rewrite this method so we don't have this fill!() at the beginning!
    # Calculate each output pixel once rather than accumulating into it?
    if beta == T(0)
        fill!(x, T(0))
    elseif beta == T(1)
        # nothing
    else
        x .*= beta
    end

    # Reshape col for easy access, by output pixel and input patch element.
    col_reshaped = reshape(col, (size(col, 1), kernel_w, kernel_h, kernel_d, C_in))

    # A helper function to project from output (w, h) to input (input_w, input_h)
    @inline project(idx, stride, pad) = (idx - 1)*stride - pad + 1

    @inbounds for c in 1:C_in
        for kd in 1:kernel_d,
            kh in 1:kernel_h,
            kw in 1:kernel_w

            for d in tile_ds(tile, 1:out_depth)
                input_kd = project(d, stride_d, pad_d_lo) + (kd - 1)*dil_d

                # If this d is off the edge, then deal with the entire plane
                # in one fell swoop, like a ravenous flock of crows.  CAW CAW.
                if input_kd <= 0 || input_kd > depth
                    continue
                end

                for h in tile_hs(tile, 1:out_height, d)
                    input_kh = project(h, stride_h, pad_h_lo) + (kh - 1)*dil_h

                    # Same for `h`, but in this case it's only a line, not a plane.
                    # This results in slightly less caw'ing.
                    if input_kh <= 0 || input_kh > height
                        continue
                    end

                    for w in tile_ws(tile, 1:out_width, h, d)
                        input_kw = project(w, stride_w, pad_w_lo) + (kw - 1)*dil_w

                        # If this `w` is off the edge, only it gets cleared out.
                        if input_kw <= 0 || input_kw > width
                            continue
                        end

                        # Copy the data over
                        kidxs = kernel_index(kw, kh, kd, cdims)
                        cval::T = col_reshaped[tile_row(tile, w, h, d), kidxs..., c]
                        x[input_kw, input_kh, input_kd, c] += cval
                    end
                end
            end
        end
    end
end
