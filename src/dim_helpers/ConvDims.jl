export ConvDims

"""
    ConvDims

Type system-level information about convolution dimensions. Critical for things like
`im2col!()` to generate efficient code, and helpful to reduce the number of kwargs
getting passed around.
"""
abstract type ConvDims{N} end

@inline spatial_dims(::ConvDims{N}) where N = N
@inline input_size(c::ConvDims) = c.input_size
@inline kernel_size(c::ConvDims) = c.kernel_size

@inline stride(c::ConvDims) = c.stride
@inline padding(c::ConvDims) = c.padding
@inline dilation(c::ConvDims) = c.dilation
@inline flipkernel(c::ConvDims) = c.flipkernel

@inline groupcount(c::ConvDims) = 1

function basetype(::Type{C}) where {C <: ConvDims}
    if C <: DenseConvDims
        return DenseConvDims
    elseif C <: DepthwiseConvDims
        return DepthwiseConvDims
    elseif C <: PoolDims
        return PoolDims
    end
    nothing
end

function output_size(c::ConvDims)
    I = input_size(c)
    K = kernel_size(c)
    S = stride(c)
    P = padding(c)
    D = dilation(c)

    return ntuple(spatial_dims(c)) do i
        return div(I[i] + P[(i-1)*2 + 1] + P[(i-1)*2 + 2] - (K[i] - 1) * D[i] - 1, S[i]) + 1
    end
end

function Base.show(io::IO, cdims::C) where {C <: ConvDims}
    I = (input_size(cdims)..., channels_in(cdims))
    O = (output_size(cdims)..., channels_out(cdims))
    K = kernel_size(cdims)
    S = stride(cdims)
    P = padding(cdims)
    D = dilation(cdims)
    F = flipkernel(cdims)
    G = groupcount(cdims)
    print(io, "$(basetype(C)): $I * $K -> $O, stride: $S, pad: $P, dil: $D, flip: $F, groups: $G")
end

"""
    im2col_dims(c::ConvDims)

im2col calculates, for each output pixel, the "convolution" of N kernels where N is the
number of output channels, by doing a matrix multiply.  The dimensions of that matrix
are given by this function.

Note that because im2col is multithreaded, we need to allocate a separate workspace of
memory per-thread; hence the dimensions returned by this will depend on the number of
threads Julia is currently running with.
"""
function im2col_dims(c)
    return (
        # Output size
        prod(output_size(c)),
        # Size of single dotproduct within convolution
        prod(kernel_size(c))*channels_in(c),
        # One workspace per thread
        Threads.nthreads(),
    )
end

# Protect your skin, kids.  Also do common validation of stride, padding, etc...
function check_spdf(x_size::NTuple{N}, w_size::NTuple{N}, stride, padding, dilation) where {N}
    # Number of spatial dimensions in `x` and `w`.
    nd = N - 2

    # Given a number, duplicate it out to have `nd` length.  If it's already a collection,
    # just splat it out into a tuple so it's always a tuple.  We'll lint length later.
    expand_size(p::Number) = ntuple(_ -> Int(p), nd)
    expand_size(p) = tuple(p...)

    # Convert stride, padding, dilation, etc.. to fully-specified tuples
    pstride = expand_size(stride)
    pdilation = expand_size(dilation)
    ppadding = expand_size(padding)

    if length(pstride) != nd
        throw(DimensionMismatch("Stride $(length(stride))d, should be $(nd)d!"))
    end
    if length(pdilation) != nd
        throw(DimensionMismatch("Dilation $(length(pdilation))d, should be $(nd)d!"))
    end

    # padding is kind of a special case; we allow it to be either 2-length or 4-length,
    # since we support asymmetrical padding
    if length(ppadding) != 2 * nd
        if length(ppadding) == nd
            # Do this repeat dance so that we get lo/hi symmetrical padding
            ppadding = NTuple{2 * nd, Int}(repeat(collect(ppadding), inner=2))
        else
            throw(DimensionMismatch("Padding $(length(ppadding))d, should be either $(nd)d or $(2*nd)d!"))
        end
    end

    # Assert that kernel size * dilation is <= padded input size
    for idx in 1:nd
        Is = x_size[idx]
        Pl = ppadding[(idx - 1) * 2 + 1]
        Ph = ppadding[(idx - 1) * 2 + 2]
        Ks = w_size[idx]
        Ds = pdilation[idx]
        if Is + Pl + Ph < (Ks - 1)*Ds + 1
            throw(DimensionMismatch("Kernel * dilation (($Ks - 1) * $Ds + 1) cannot be larger than input + padding ($Is + $Pl + $Ph)!"))
        end
    end

    return pstride, ppadding, pdilation
end
