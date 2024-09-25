"""
    DepthwiseConvDims

Concrete subclass of `ConvDims` for a depthwise convolution.  Differs primarily due to
characterization by `C_in`, `C_mult`, rather than `C_in`, `C_out`.  Useful to be separate from
DenseConvDims primarily for channel calculation differences.
"""
struct DepthwiseConvDims{N, K, S, P, D} <: ConvDims{N}
    input_size::NTuple{N, Int}

    kernel_size::NTuple{K, Int}
    channels_in::Int
    channels_multiplier::Int

    stride::NTuple{S, Int}
    padding::NTuple{P, Int}
    dilation::NTuple{D, Int}
    flipkernel::Bool
end

function DepthwiseConvDims(
    x_size::NTuple{M}, w_size::NTuple{M};
    stride = 1, padding = 0, dilation = 1, flipkernel::Bool = false,
) where M
    sstride, ppadding, ddilation = check_spdf(
        x_size, w_size, stride, padding, dilation)

    # Ensure channels are equal
    if x_size[end-1] != w_size[end]
        xs = x_size[end-1]
        ws = w_size[end]
        throw(DimensionMismatch("Input channels must match! ($xs vs. $ws)"))
    end

    DepthwiseConvDims(
        x_size[1:(end - 2)],
        w_size[1:(end - 2)], x_size[end - 1], w_size[end - 1],
        sstride, ppadding, ddilation, flipkernel)
end

function DepthwiseConvDims(x::AbstractArray, w::AbstractArray; kwargs...)
    if ndims(x) != ndims(w)
        throw(DimensionMismatch("Rank of x and w must match! ($(ndims(x)) vs. $(ndims(w)))"))
    end
    return DepthwiseConvDims(size(x), size(w); kwargs...)
end

# Useful for constructing a new DepthwiseConvDims that has only a few elements different
# from the original progenitor object.
@inline DepthwiseConvDims(
    c::DepthwiseConvDims; I=input_size(c), K=kernel_size(c),
    C_in=channels_in(c), C_m=channel_multiplier(c), S=stride(c),
    P=padding(c), D=dilation(c), F=flipkernel(c),
) = DepthwiseConvDims(
    I,
    K, C_in, C_m,
    S, P, D, F)

@inline channels_in(c::DepthwiseConvDims) = c.channels_in
@inline channels_out(c::DepthwiseConvDims) = c.channels_in * c.channels_multiplier
@inline channel_multiplier(c::DepthwiseConvDims) = c.channels_multiplier

@inline input_size(c::DepthwiseConvDims) = c.input_size
@inline kernel_size(c::DepthwiseConvDims) = c.kernel_size

@inline stride(c::DepthwiseConvDims) = c.stride
@inline padding(c::DepthwiseConvDims) = c.padding
@inline dilation(c::DepthwiseConvDims) = c.dilation
@inline flipkernel(c::DepthwiseConvDims) = c.flipkernel

# This one is basically the same as for DenseConvDims, we only change a few lines for kernel channel count
function check_dims(x::NTuple{M}, w::NTuple{M}, y::NTuple{M}, cdims::DepthwiseConvDims) where {M}
    # First, check that channel counts are all correct:
    @assert x[M-1] == channels_in(cdims) DimensionMismatch("Data input channel count ($(x[M-1]) vs. $(channels_in(cdims)))")
    @assert y[M-1] == channels_out(cdims) DimensionMismatch("Data output channel count ($(y[M-1]) vs. $(channels_out(cdims)))")
    @assert w[M-1] == channel_multiplier(cdims) DimensionMismatch("Kernel multiplier channel count ($(w[M-1]) vs. $(channel_multiplier(cdims))")
    @assert w[M] == channels_in(cdims) DimensionMismatch("Kernel input channel count ($(w[M]) vs. $(channels_in(cdims)))")

    # Next, check that the spatial dimensions match up
    @assert x[1:M-2] == input_size(cdims) DimensionMismatch("Data input spatial size ($(x[1:M-2]) vs. $(input_size(cdims)))")
    @assert y[1:M-2] == output_size(cdims) DimensionMismatch("Data output spatial size ($(y[1:M-2]) vs. $(output_size(cdims)))")
    @assert w[1:M-2] == kernel_size(cdims) DimensionMismatch("Kernel spatial size ($(w[1:M-2]) vs. $(kernel_size(cdims)))")

    # Finally, check that the batch size matches
    @assert x[M] == y[M] DimensionMismatch("Batch size ($(x[M]) vs. $(y[M]))")
end
