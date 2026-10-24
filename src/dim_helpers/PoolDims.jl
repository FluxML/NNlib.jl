"""
    PoolDims(x_size::NTuple{M}, k::Union{NTuple{<:Any, Int}, Int};
             stride=k, padding=0, dilation=1, ceil_mode=false)  where M

Dimensions for a "pooling" operation that can have an arbitrary input size, kernel size,
stride, dilation, and channel count.  Used to dispatch onto efficient implementations at
compile-time.

With `ceil_mode=true` the output size is rounded up instead of down, as with PyTorch's
`ceil_mode=True`: a last window that runs past the padded input is kept, as long as it
starts inside the input or the low padding. Its positions past the padded input count
neither as input nor as padding: max pooling skips them, and mean pooling leaves them out
of the divisor even with `count_include_pad=true`. `ceil_mode` is dropped when it doesn't
change the output size.
"""
struct PoolDims{N, K, S, P, D} <: ConvDims{N}
    input_size::NTuple{N, Int}

    kernel_size::NTuple{K, Int}
    channels_in::Int

    stride::NTuple{S, Int}
    padding::NTuple{P, Int}
    dilation::NTuple{D, Int}
    ceil_mode::Bool
end

PoolDims(I, K, C_in, S, P, D) = PoolDims(I, K, C_in, S, P, D, false)
PoolDims{N, K, S, P, D}(I, Ks, C_in, Ss, Ps, Ds) where {N, K, S, P, D} =
    PoolDims{N, K, S, P, D}(I, Ks, C_in, Ss, Ps, Ds, false)

function PoolDims(
    x_size::NTuple{M}, k::Union{NTuple{<:Any, Int}, Int};
    stride = k, padding = 0, dilation = 1, ceil_mode::Bool = false,
) where M
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
    pdims = PoolDims(
        x_size[1:(end - 2)], kernel, x_size[end - 1],
        sstride, ppadding, ddilation)
    # Backends without `ceil_mode` support still run where it changes nothing
    ceil_pdims = PoolDims(pdims; ceil_mode=true)
    return ceil_mode && output_size(ceil_pdims) != output_size(pdims) ? ceil_pdims : pdims
end

PoolDims(x::AbstractArray, k; kwargs...) = PoolDims(size(x), k; kwargs...)

# Useful for constructing a new PoolDims that has only a few elements different
# from the original progenitor object that it inherits shapes from.
PoolDims(
    c::C; I=input_size(c), K=kernel_size(c),
    C_in=channels_in(c), S=stride(c), P=padding(c), D=dilation(c),
    ceil_mode=ceil_mode(c),
) where C <: ConvDims = PoolDims(I, K, C_in, S, P, D, ceil_mode)

@inline channels_in(c::PoolDims) = c.channels_in
@inline channels_out(c::PoolDims) = c.channels_in

@inline input_size(c::PoolDims) = c.input_size
@inline kernel_size(c::PoolDims) = c.kernel_size

@inline stride(c::PoolDims) = c.stride
@inline padding(c::PoolDims) = c.padding
@inline dilation(c::PoolDims) = c.dilation
@inline ceil_mode(c::PoolDims) = c.ceil_mode
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
