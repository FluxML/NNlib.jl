export ConvDims

"""
    ConvDims

Type system-level information about convolution dimensions. Critical for things like
`im2col!()` to generate efficient code, and helpful to reduce the number of kwargs
getting passed around.

We don't want to specialize on things like image size/channel count, so we generally
store those as fields, just for convenience, and to allow for non-breaking changes when
we decide we _do_ want to specialize on those values.  We always want to specialize on
things like stride, padding, dilation, and kernel flipping though.
"""
abstract type ConvDims{N, S, P, D, F} end

# Hack to get rid of type parameters
function basetype(::Type{C}) where {C <: ConvDims}
    if C <: DepthwiseConvDims
        return DepthwiseConvDims
    elseif C <: DenseConvDims
        return DenseConvDims
    elseif C <: PoolDims
        return PoolDims
    else
        return nothing
    end
end

# Obvious getter definitions for the type system-level definitions
spatial_dims(c::ConvDims{N,S,P,D,F}) where {N, S, P, D, F} = N
stride(c::ConvDims{N,S,P,D,F}) where {N, S, P, D, F} = S
padding(c::ConvDims{N,S,P,D,F}) where {N, S, P, D, F} = P
dilation(c::ConvDims{N,S,P,D,F}) where {N, S, P, D, F} = D
flipkernel(c::ConvDims{N,S,P,D,F}) where {N, S, P, D, F} = F

"""
    im2col_dims(c::ConvDims)

im2col calculates, for each output pixel, the "convolution" of N kernels where N is the
number of output channels, by doing a matrix multiply.  The dimensions of that matrix
are given by this function.
"""
im2col_dims(c::ConvDims) = (prod(output_size(c)), prod(kernel_size(c))*channels_in(c))

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
    if length(ppadding) != 2*nd
        if length(ppadding) == nd
            # Do this repeat dance so that we get lo/hi symmetrical padding
            ppadding = tuple(repeat(collect(ppadding), inner=2)...)
        else
            throw(DimensionMismatch("Padding $(length(ppadding))d, should be either $(nd)d or $(2*nd)d!"))
        end
    end

    # Assert that kernel size * dilation is <= padded input size
    for idx in 1:nd
        Is = x_size[idx]
        Pl = ppadding[(idx - 1)*2 + 1]
        Ph = ppadding[(idx - 1)*2 + 2]
        Ks = w_size[idx]
        Ds = pdilation[idx]
        if Is + Pl + Ph < (Ks - 1)*Ds + 1
            throw(DimensionMismatch("Kernel * dilation (($Ks - 1) * $Ds + 1) cannot be larger than input + padding ($Is + $Pl + $Ph)!"))
        end
    end

    return pstride, ppadding, pdilation
end

"""
    output_size(c::ConvDims)

Calculate the output (spatial) dimensions of the convolution.  Get channel count via
`channels_out(c)`, and batch count is unknowable.
"""
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

# Override show() for these beauties
function Base.show(io::IO, cdims::C) where {C <: ConvDims}
    I = (input_size(cdims)..., channels_in(cdims))
    O = (output_size(cdims)..., channels_out(cdims))
    K = kernel_size(cdims)
    S = stride(cdims)
    P = padding(cdims)
    D = dilation(cdims)
    F = flipkernel(cdims)
    print(io, "$(basetype(C)): $I * $K -> $O, stride: $S pad: $P, dil: $D, flip: $F")
end
