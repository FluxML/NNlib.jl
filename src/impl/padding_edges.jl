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
    calc_lo_spill(O, S, P) = min(ceil(Int, P/S), O)
    @inline function calc_hi_spill(O, S, Pl, Ph, K, D, I)
        wasted_Ph = (I + Pl + Ph - (K - 1)*D - 1)%S
        return min(ceil(Int, (Ph - wasted_Ph)/S), O)
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

    # Note; we need two special cases; for when we're actually dealing with a 1d or 2d
    # convolution.  This is detectable when `d` is a singleton dimension and there is no
    # padding along that dimension, or when the same holds true for `d` and `h`.  In
    # those cases, we can simplify the padded regions and central regions a bit.
    singleton(args...) = (args == (1, 1, 1, 1, 0, 0, 1))
    if singleton(depth, out_depth, kernel_d, stride_d, pad_d_lo, pad_d_hi, dil_d)
        # So we've got a singleton depth dimension.  Do the same check for height:
        if singleton(height, out_height, kernel_h, stride_h, pad_h_lo, pad_h_hi, dil_h)
            # This means it's a 1-d region.  So we simplify things quite a lot:
            padded_regions = (
                # Only two padded regions; low-w and hi-w.
                (1:spill_w_lo, 1:1, 1:1),
                (spill_w_hi_abs:out_width, 1:1, 1:1),
            )

            # The central region that has no padding.
            central_region = (
                (spill_w_lo+1):(spill_w_hi_abs - 1),
                1:1,
                1:1,
            )
        else
            # This means it's a 2-d region.  Still simplified, but not SO simplified:
            padded_regions = (
                # First region is the lower-H W edge:
                (
                    1:out_width,
                    1:spill_h_lo,
                    1:1,
                ),
                # Then the upper-H W edge
                (
                    1:out_width,
                    spill_h_hi_abs:out_height,
                    1:1,
                ),

                # Next, we fit the H edges in, but without overlapping the `h` regions
                # we've done before:
                (
                    1:spill_w_lo,
                    (spill_h_lo+1):(spill_h_hi_abs-1),
                    1:1,
                ),
                (
                    spill_w_hi_abs:out_width,
                    (spill_h_lo+1):(spill_h_hi_abs-1),
                    1:1
                ),
            )

            # The central region that has no padding.
            central_region = (
                (spill_w_lo+1):(spill_w_hi_abs - 1),
                (spill_h_lo+1):(spill_h_hi_abs - 1),
                1:1,
            )
        end
    else
        # Otherwise it's the full 3d dimensional calculation.
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
    end

    # Filter out regions that are empty in some dimension
    @inline function filter_empty(regions)
        return tuple((r for r in regions if all(!isempty(z) for z in r))...)
    end
    return filter_empty(padded_regions), central_region
end