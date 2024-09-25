"""
    PoolDims(x_size::NTuple{M}, k::Union{NTuple{L, Int}, Int};
             stride=k, padding=0, dilation=1)  where {M, L}

Dimensions for a "pooling" operation that can have an arbitrary input size, kernel size,
stride, dilation, and channel count.  Used to dispatch onto efficient implementations at
compile-time.
"""
struct PoolDims{N, K, S, P, D} <: ConvDims{N}
    input_size::NTuple{N, Int}

    kernel_size::NTuple{K, Int}
    channels_in::Int

    stride::NTuple{S, Int}
    padding::NTuple{P, Int}
    dilation::NTuple{D, Int}
end

function PoolDims(
    x_size::NTuple{M}, k::Union{NTuple{L, Int}, Int};
    stride = k, padding = 0, dilation = 1,
) where {M, L}
    _check_kernel(k::Number, N::Int) = ntuple(_ -> Int(k), N)
    _check_kernel(k::NTuple, ::Int) = k

    kernel = _check_kernel(k, M - 2)
    length(x_size) == length(kernel) + 2 || error(
        "PoolDims expects ndim(x) == length(k)+2 or length(size(x)) == length(kernel)+2,
        dimension of x_size is $(length(x_size)),
        length of k need $(length(x_size) - 2),
        but now it's $(length(kernel))"
    )
    spdf_kernel = NTuple{M, Int}([kernel..., 1, 1])

    sstride, ppadding, ddilation = check_spdf(
        x_size, spdf_kernel, stride, padding, dilation)
    PoolDims(
        x_size[1:(end - 2)], kernel, x_size[end - 1],
        sstride, ppadding, ddilation)
end

PoolDims(x::AbstractArray, k; kwargs...) = PoolDims(size(x), k; kwargs...)

# Useful for constructing a new PoolDims that has only a few elements different
# from the original progenitor object that it inherits shapes from.
PoolDims(
    c::C; I=input_size(c), K=kernel_size(c),
    C_in=channels_in(c), S=stride(c), P=padding(c), D=dilation(c),
) where C <: ConvDims = PoolDims(I, K, C_in, S, P, D)

@inline channels_in(c::PoolDims) = c.channels_in
@inline channels_out(c::PoolDims) = c.channels_in

@inline input_size(c::PoolDims) = c.input_size
@inline kernel_size(c::PoolDims) = c.kernel_size

@inline stride(c::PoolDims) = c.stride
@inline padding(c::PoolDims) = c.padding
@inline dilation(c::PoolDims) = c.dilation
@inline flipkernel(c::PoolDims) = false

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
