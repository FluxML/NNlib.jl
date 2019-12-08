export PoolDims

"""
    PoolDims

Dimensions for a "pooling" operation that can have an arbitrary input size, kernel size,
stride, dilation, and channel count.  Used to dispatch onto efficient implementations at
compile-time.
"""
struct PoolDims{N,K,S,P,D} <: AbstractDims{N, S, P, D, false}
    I::NTuple{N,Int}
    C_in::Int
end

# Getters for both type parameters and fields
kernel_size(c::PoolDims{N,K,S,P,D}) where {N, K, S, P, D} = K
input_size(c::PoolDims) = c.I
channels_in(c::PoolDims) = c.C_in
channels_out(c::PoolDims) = c.C_in


# Convenience wrapper to create ConvDims objects
function PoolDims(x_size::NTuple{M}, k::Union{NTuple{L, Int}, Int};
                  stride=k, padding=0, dilation=1) where {M, L}
    # Expand `k` up to a tuple
    if typeof(k) <: Number
        k = ntuple(_ -> k, M - 2)
    end

    # Do common parameter validation
    stride, padding, dilation = check_spdf(x_size, (k..., 1, 1), stride, padding, dilation)

    # Build it
    return PoolDims{
        M - 2,
        k,
        stride,
        padding,
        dilation
    }(
        # Image spatial size
        x_size[1:end-2],

        # Input channels
        x_size[end-1],
    )
end

# Auto-take `size(x)` when `x` is an array.
function PoolDims(x::AbstractArray, k; kwargs...)
    return PoolDims(size(x), k; kwargs...)
end

# Useful for constructing a new PoolDims that has only a few elements different
# from the original progenitor object that it inherits shapes from.
function PoolDims(c::AbstractDims; N=spatial_dims(c), I=input_size(c), K=kernel_size(c),
                       C_in=channels_in(c), S=stride(c), P=padding(c), D=dilation(c))
    return PoolDims{N, K, S, P, D}(I, C_in)
end

function check_dims(x::NTuple{M}, y::NTuple{M}, pdims::PoolDims) where {M}
    # First, check that channel counts are all correct:
    @assert x[end-1] == channels_in(pdims) DimensionMismatch("Data input channel count ($(x[end-1]) vs. $(channels_in(pdims)))")
    @assert y[end-1] == channels_out(pdims) DimensionMismatch("Data output channel count ($(y[end-1]) vs. $(channels_out(pdims)))")

    # Next, check that the spatial dimensions match up
    @assert x[1:end-2] == input_size(pdims) DimensionMismatch("Data input spatial size ($(x[1:end-2]) vs. $(input_size(pdims)))")
    @assert y[1:end-2] == output_size(pdims) DimensionMismatch("Data output spatial size ($(y[1:end-2]) vs. $(output_size(pdims)))")

    # Finally, check that the batch size matches
    @assert x[end] == y[end] DimensionMismatch("Batch size ($(x[end]) vs. $(y[end]))")
end
