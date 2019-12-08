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

struct ConvDims{N,K,C_in,C_out,S,P,D,F,G} <: AbstractDims{N,S,P,D,F}
    I::NTuple{N,Int}
end


# Getters for the fields
input_size(c::ConvDims) = c.I
kernel_size(c::ConvDims{N,K,C_in,C_out,S,P,D,F,G}) where {N,K,C_in,C_out,S,P,D,F,G} = K
channels_in(c::ConvDims{N,K,C_in,C_out,S,P,D,F,G}) where {N,K,C_in,C_out,S,P,D,F,G} = C_in
channels_out(c::ConvDims{N,K,C_in,C_out,S,P,D,F,G}) where {N,K,C_in,C_out,S,P,D,F,G} = C_out
group_count(c::ConvDims{N,K,C_in,C_out,S,P,D,F,G}) where {N,K,C_in,C_out,S,P,D,F,G} = G

# Convenience wrapper to create ConvDims objects
function ConvDims(x_size::NTuple{M}, w_size::NTuple{M};
                       stride=1, padding=0, dilation=1, flipkernel::Bool=false, groupcount=1) where M
    # Do common parameter validation
    stride, padding, dilation = check_spdf(x_size, w_size, stride, padding, dilation)

    # Ensure channels are equal
    if x_size[M-1] != w_size[M-1]*groupcount
        xs = x_size[M-1]
        ws = w_size[M-1]*groupcount
        throw(DimensionMismatch("Input channels must match! ($xs vs. $ws)"))
    end

    # The type parameters are what
    return ConvDims{
        M - 2,
        w_size[1:M-2],
        x_size[M-1],
        w_size[M],
        stride,
        padding,
        dilation,
        flipkernel,
        groupcount
    }(
        # Input spatial size
        x_size[1:M-2],
    )
end

# Auto-extract sizes and sub out to big brother above
function ConvDims(x::AbstractArray, w::AbstractArray; kwargs...)
    if ndims(x) != ndims(w)
        throw(DimensionMismatch("Rank of x and w must match! ($(ndims(x)) vs. $(ndims(w)))"))
    end
    return ConvDims(size(x), size(w); kwargs...)
end

# Useful for constructing a new ConvDims that has only a few elements different
# from the original progenitor object that it inherits shapes from.
function ConvDims(c::AbstractDims; N=spatial_dims(c), I=input_size(c), K=kernel_size(c),
                       C_in=channels_in(c), C_out=channels_out(c), S=stride(c),
                       P=padding(c), D=dilation(c), F=flipkernel(c), G=group_count(c))
    return ConvDims{N, K, C_in, C_out, S, P, D, F, G}(I)
end

function check_dims(x::NTuple{M}, w::NTuple{M}, y::NTuple{M}, cdims::ConvDims) where {M}
    # First, check that channel counts are all correct:
    @assert x[M-1] == channels_in(cdims) DimensionMismatch("Data input channel count ($(x[M-1]) vs. $(channels_in(cdims)))")
    @assert y[M-1] == channels_out(cdims) DimensionMismatch("Data output channel count ($(y[M-1]) vs. $(channels_out(cdims)))")
    @assert w[M-1] == channels_in(cdims)/group_count(cdims) DimensionMismatch("Kernel input channel count ($(w[M-1]) vs. $(channels_in(cdims)/group_count(cdims)))")
    @assert w[M] == channels_out(cdims) DimensionMismatch("Kernel output channel count ($(w[M]) vs. $(channels_out(cdims)))")

    # Next, check that the spatial dimensions match up
    @assert x[1:M-2] == input_size(cdims) DimensionMismatch("Data input spatial size ($(x[1:M-2]) vs. $(input_size(cdims)))")
    @assert y[1:M-2] == output_size(cdims) DimensionMismatch("Data output spatial size ($(y[1:M-2]) vs. $(output_size(cdims)))")
    @assert w[1:M-2] == kernel_size(cdims) DimensionMismatch("Kernel spatial size ($(w[1:M-2]) vs. $(kernel_size(cdims)))")

    # Finally, check that the batch size matches
    @assert x[M] == y[M] DimensionMismatch("Batch size ($(x[M]) vs. $(y[M]))")
end
