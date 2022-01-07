export DenseConvDims

struct DenseConvDims{N, K, S, P, D}
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

    if x_size[end - 1] != w_size[end - 1] * groups
        xs = x_size[end - 1]
        ws = w_size[end - 1]
        throw(DimensionMismatch("Input channels must match! ($xs vs. $ws)"))
    end

    if x_size[end - 1] % w_size[end - 1] != 0 || w_size[end] % groups != 0
        throw(DimensionMismatch(
            "Group count should be divisble by input and output channels ($groups vs. $(w_size[end-1:end]))"))
    end

    DenseConvDims(
        x_size[1:(end - 2)], w_size[1:(end - 2)], x_size[end - 1], w_size[end],
        groups, sstride, ppadding, ddilation, flipkernel)
end

function DenseConvDims(x::AbstractArray, w::AbstractArray; kwargs...)
    if ndims(x) != ndims(w)
        throw(DimensionMismatch(
            "Rank of x and w must match! ($(ndims(x)) vs. $(ndims(w)))"))
    end
    return DenseConvDims(size(x), size(w); kwargs...)
end

# TODO: `N` parameter is for compatibility.
@inline DenseConvDims(
    c::C; N=spatial_dims(c), I=input_size(c), K=kernel_size(c),
    C_in=channels_in(c), C_out=channels_out(c), S=stride(c),
    P=padding(c), D=dilation(c), F=flipkernel(c), G=groupcount(c),
) where C<: Union{DenseConvDims, ConvDims} = DenseConvDims(
        I,
        K, C_in, C_out, G,
        S, P, D, F)

@inline spatial_dims(::DenseConvDims{N, K, S, P, D}) where {N, K, S, P, D} = N

@inline groupcount(c::DenseConvDims) = c.groupcount
@inline channels_in(c::DenseConvDims) = c.channels_in
@inline channels_out(c::DenseConvDims) = c.channels_out
@inline kernel_size(c::DenseConvDims) = c.kernel_size

@inline stride(c::DenseConvDims) = c.stride
@inline padding(c::DenseConvDims) = c.padding
@inline dilation(c::DenseConvDims) = c.dilation
@inline flipkernel(c::DenseConvDims) = c.flipkernel

@inline input_size(c::DenseConvDims) = c.input_size
@inline function output_size(c::DenseConvDims)
    ntuple(spatial_dims(c)) do i
        div(
            c.input_size[i] +
            c.padding[(i - 1) * 2 + 1] +
            c.padding[(i - 1) * 2 + 2] -
            (c.kernel_size[i] - 1) * c.dilation[i] - 1,
            c.stride[i]) + 1
    end
end

@inline basetype(::Type{D}) where D <: DenseConvDims = DenseConvDims

function check_dims(
    x::NTuple{M}, w::NTuple{M}, y::NTuple{M}, cdims::DenseConvDims,
) where {M}
    # First, check that channel counts are all correct:
    @assert x[M-1] * cdims.groupcount == cdims.channels_in DimensionMismatch("Data input channel count ($(x[M-1]) vs. $(channels_in(cdims)))")
    @assert y[M-1] == cdims.channels_out ÷ cdims.groupcount  DimensionMismatch("Data output channel count ($(y[M-1]) vs. $(channels_out(cdims)))")
    @assert w[M-1] * cdims.groupcount == cdims.channels_in DimensionMismatch("Kernel input channel count ($(w[M-1]) vs. $(channels_in(cdims)))")
    @assert w[M] * cdims.groupcount == cdims.channels_out DimensionMismatch("Kernel output channel count ($(w[M]) vs. $(channels_out(cdims)))")

    # Next, check that the spatial dimensions match up
    @assert x[1:M-2] == cdims.input_size DimensionMismatch("Data input spatial size ($(x[1:M-2]) vs. $(input_size(cdims)))")
    @assert y[1:M-2] == output_size(cdims) DimensionMismatch("Data output spatial size ($(y[1:M-2]) vs. $(output_size(cdims)))")
    @assert w[1:M-2] == cdims.kernel_size DimensionMismatch("Kernel spatial size ($(w[1:M-2]) vs. $(kernel_size(cdims)))")

    # Check the groups match
    @assert cdims.channels_in % cdims.groupcount == 0 DimensionMismatch("Groups ($(groupcount(cdims))) should be divisble by input channels $(channels_in(cdims))")

    # Finally, check that the batch size matches
    @assert x[M] == y[M] DimensionMismatch("Batch size ($(x[M]) vs. $(y[M]))")
end
