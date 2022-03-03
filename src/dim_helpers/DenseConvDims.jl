"""
    DenseConvDims

Concrete subclass of `ConvDims` for a normal, dense, conv2d/conv3d.
"""
struct DenseConvDims{N, K, S, P, D} <: ConvDims{N}
    input_size::NTuple{N, Int}

    kernel_size::NTuple{K, Int}
    channels_in::Int
    channels_out::Int
    groupcount::Int

    stride::NTuple{S, Int}
    padding::NTuple{P, Int}
    dilation::NTuple{D, Int}
    flipkernel::Bool
end

function DenseConvDims(
    x_size::NTuple{M}, w_size::NTuple{M};
    stride = 1, padding = 0, dilation = 1, groups = 1,
    flipkernel::Bool = false,
) where {M}
    sstride, ppadding, ddilation = check_spdf(
        x_size, w_size, stride, padding, dilation)

    # Ensure channels are equal
    if x_size[end - 1] != w_size[end - 1] * groups
        xs = x_size[end - 1]
        ws = w_size[end - 1]
        throw(DimensionMismatch("Input channels must match! ($xs vs. $ws)"))
    end

    # Ensure groups are valid
    if x_size[end - 1] % w_size[end - 1] != 0 || w_size[end] % groups != 0
        throw(DimensionMismatch(
            "Group count should be divisble by input and output channels ($groups vs. $(w_size[end-1:end]))"))
    end

    DenseConvDims(
        x_size[1:(end - 2)],
        w_size[1:(end - 2)], x_size[end - 1], w_size[end], groups,
        sstride, ppadding, ddilation, flipkernel)
end

function DenseConvDims(x::AbstractArray, w::AbstractArray; kwargs...)
    if ndims(x) != ndims(w)
        throw(DimensionMismatch(
            "Rank of x and w must match! ($(ndims(x)) vs. $(ndims(w)))"))
    end
    return DenseConvDims(size(x), size(w); kwargs...)
end

# Useful for constructing a new DenseConvDims that has only a few elements different
# from the original progenitor object that it inherits shapes from.
@inline DenseConvDims(
    c::C; I=input_size(c), K=kernel_size(c),
    C_in=channels_in(c), C_out=channels_out(c), S=stride(c),
    P=padding(c), D=dilation(c), F=flipkernel(c), G=groupcount(c),
) where C <: ConvDims = DenseConvDims(
    I,
    K, C_in, C_out, G,
    S, P, D, F)

@inline groupcount(c::DenseConvDims) = c.groupcount
@inline channels_in(c::DenseConvDims) = c.channels_in
@inline channels_out(c::DenseConvDims) = c.channels_out

@inline input_size(c::DenseConvDims) = c.input_size
@inline kernel_size(c::DenseConvDims) = c.kernel_size

@inline stride(c::DenseConvDims) = c.stride
@inline padding(c::DenseConvDims) = c.padding
@inline dilation(c::DenseConvDims) = c.dilation
@inline flipkernel(c::DenseConvDims) = c.flipkernel

function check_dims(x::NTuple{M}, w::NTuple{M}, y::NTuple{M}, cdims::DenseConvDims) where {M}
    # First, check that channel counts are all correct:
    @assert x[M-1] * groupcount(cdims) == channels_in(cdims) DimensionMismatch("Data input channel count ($(x[M-1]) vs. $(channels_in(cdims)))")
    @assert y[M-1] == channels_out(cdims) ÷ groupcount(cdims)  DimensionMismatch("Data output channel count ($(y[M-1]) vs. $(channels_out(cdims)))")
    @assert w[M-1] * groupcount(cdims) == channels_in(cdims) DimensionMismatch("Kernel input channel count ($(w[M-1]) vs. $(channels_in(cdims)))")
    @assert w[M] * groupcount(cdims) == channels_out(cdims) DimensionMismatch("Kernel output channel count ($(w[M]) vs. $(channels_out(cdims)))")

    # Next, check that the spatial dimensions match up
    @assert x[1:M-2] == input_size(cdims) DimensionMismatch("Data input spatial size ($(x[1:M-2]) vs. $(input_size(cdims)))")
    @assert y[1:M-2] == output_size(cdims) DimensionMismatch("Data output spatial size ($(y[1:M-2]) vs. $(output_size(cdims)))")
    @assert w[1:M-2] == kernel_size(cdims) DimensionMismatch("Kernel spatial size ($(w[1:M-2]) vs. $(kernel_size(cdims)))")

    # Check the groups match
    @assert channels_in(cdims) % groupcount(cdims) == 0 DimensionMismatch("Groups ($(groupcount(cdims))) should be divisble by input channels $(channels_in(cdims))")

    # Finally, check that the batch size matches
    @assert x[M] == y[M] DimensionMismatch("Batch size ($(x[M]) vs. $(y[M]))")
end
