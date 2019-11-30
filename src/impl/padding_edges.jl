"""
    calc_padding_regions(dims)

Padding is a jerk.  A HUGE jerk that tries to sneak a bunch of conditionals and edge
cases (quite literally) into our beautiful stencil operations such as convolution,
pooling, etc...  The way we deal with this is to, first, deal with everything in 3d,
and then define a single padding region helper function that returns the seven regions
that all 3d operations must deal with, including the central "unpadded" region where we
can run at full bore, not paying any attention to padding.
"""
function calc_padding_regions(dims)
    width, height, depth = input_size(dims)
    kernel_w, kernel_h, kernel_d = kernel_size(dims)
    C_in = channels_in(dims)
    pad_w_lo, pad_w_hi, pad_h_lo, pad_h_hi, pad_d_lo, pad_d_hi = padding(dims)
    dil_w, dil_h, dil_d = dilation(dims)
    stride_w, stride_h, stride_d = stride(dims)
    out_width, out_height, out_depth = output_size(dims)

    # Let us first calculate the number of rows/cols within which we must zero out some
    # portion of the image patches we're copying over.  The "spillage" here is the number
    # of indices along a particular dimension for which a kernel will have some portion
    # of its input domain overlapping the padding.  If padding is zero, these values are
    # all trivially zero.  The low spillage is trivially the low padding divided by the
    # stride; literally the number of shifts that overlap some padding.  The high
    # spillage is slightly more complicated; we first figure out how many elements of
    # high padding are wasted (e.g. through strides not fitting to the end perfectly)
    # subtract that from the high padding, then do the same:
    calc_lo_spill(O, S, P) = max(min(ceil(Int, P/S), O),0)
    @inline function calc_hi_spill(O, S, Pl, Ph, K, D, I)
        wasted_Ph = (I + Pl + Ph - (K - 1)*D - 1)%S
        return max(min(ceil(Int, (Ph - wasted_Ph)/S), O), 0)
    end

    spill_w_lo = calc_lo_spill(out_width, stride_w, pad_w_lo)
    spill_w_hi = calc_hi_spill(out_width, stride_w, pad_w_lo, pad_w_hi, kernel_w, dil_w, width)
    spill_h_lo = calc_lo_spill(out_height, stride_h, pad_h_lo)
    spill_h_hi = calc_hi_spill(out_height, stride_h, pad_h_lo, pad_h_hi, kernel_h, dil_h, height)
    spill_d_lo = calc_lo_spill(out_depth, stride_d, pad_d_lo)
    spill_d_hi = calc_hi_spill(out_depth, stride_d, pad_d_lo, pad_d_hi, kernel_d, dil_d, depth)

    spill_w_hi_abs = out_width  - spill_w_hi + 1
    spill_h_hi_abs = out_height - spill_h_hi + 1
    spill_d_hi_abs = out_depth  - spill_d_hi + 1

    # These are the regions we're going to have to run with cognizance of padding.
    # There are six of them; one for each face of the cube image.  We explicitly
    # design this so that we run over `width` most tightly, in the expectation that
    # this will generate better code for when `h` and `d` are singleton dimensions.
    # We visualize this as a cube, indexed by dimensions (w, h, d).
    padded_regions = (
        # First region is the lower-d WH face:
        (
            1:out_width,
            1:out_height,
            1:spill_d_lo,
        ),

        # The next largest chunk we choose will be the lower-h WD faces; we always
        # want to maximize going across full `w`, as its contiguous in memory.
        (
            1:out_width,
            1:spill_h_lo,
            (spill_d_lo+1):(spill_d_hi_abs-1),
        ),
        # Then the upper-h WD face
        (
            1:out_width,
            spill_h_hi_abs:out_height,
            (spill_d_lo+1):(spill_d_hi_abs-1),
        ),

        # Next, we fit the HD faces in, but without overlapping the `h` and `d`
        # regions we've done before:
        (
            1:spill_w_lo,
            (spill_h_lo+1):(spill_h_hi_abs-1),
            (spill_d_lo+1):(spill_d_hi_abs-1),
        ),
        (
            spill_w_hi_abs:out_width,
            (spill_h_lo+1):(spill_h_hi_abs-1),
            (spill_d_lo+1):(spill_d_hi_abs-1)
        ),
        
        # Last region is the higher-d WH face:
        (
            1:out_width,
            1:out_height,
            spill_d_hi_abs:out_depth,
        ),
    )

    # The central region that has no padding.
    central_region = (
        (spill_w_lo+1):(spill_w_hi_abs - 1),
        (spill_h_lo+1):(spill_h_hi_abs - 1),
        (spill_d_lo+1):(spill_d_hi_abs - 1),
    )
    return padded_regions, central_region
end