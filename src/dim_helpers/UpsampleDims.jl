export UpsampleDims

"""
    UpsampleDims

Dimensions for a "upsampling" operation that can have an arbitrary input size, stride,
and channel count.  Used to dispatch onto efficient implementations at compile-time.
"""
struct UpsampleDims{N,S,Sc} <: ConvDims{N, S, Sc, 1, false}
    I::NTuple{N,Int}
    C_in::Int
end

# Getters for both type parameters and fields
spatial_dims(c::UpsampleDims{N,S,Sc}) where {N, S, Sc} = N
stride(c::UpsampleDims{N,S,Sc}) where {N, S, Sc} = S
scale(c::UpsampleDims{N,S,Sc}) where {N, S, Sc} = Sc
input_size(c::UpsampleDims) = c.I
channels_in(c::UpsampleDims) = c.C_in
channels_out(c::UpsampleDims) = c.C_in
output_size(c::UpsampleDims) = map(* , stride(c), input_size(c))


# Convenience wrapper to create UpsampleDims objects
function UpsampleDims(x_size::NTuple{M}, stride::Union{NTuple{L, Int}, Int};
                      scale=1) where {M, L}
    # Expand `stride` up to a tuple
    if typeof(stride) <: Number
        stride = ntuple(_ -> stride, M - 2)
    end

    # Build it
    return UpsampleDims{
        M - 2,
        stride,
        scale
    }(
        # Image spatial size
        x_size[1:end-2],

        # Input channels
        x_size[end-1],
    )
end

# Auto-take `size(x)` when `x` is an array.
function UpsampleDims(x::AbstractArray, stride; kwargs...)
    return UpsampleDims(size(x), stride; kwargs...)
end

# Useful for constructing a new UpsampleDims that has only a few elements different
# from the original progenitor object that it inherits shapes from.
function UpsampleDims(c::UpsampleDims; N=spatial_dims(c), I=input_size(c),
                      C_in=channels_in(c), S=stride(c), Sc=scale(c))
    return UpsampleDims{N, S, Sc}(I, C_in)
end

function check_dims(x::NTuple{M}, y::NTuple{M}, udims::UpsampleDims) where {M}
    # First, check that channel counts are all correct:
    @assert x[end-1] == channels_in(udims) DimensionMismatch("Data input channel count ($(x[end-1]) vs. $(channels_in(udims)))")
    @assert y[end-1] == channels_out(udims) DimensionMismatch("Data output channel count ($(y[end-1]) vs. $(channels_out(udims)))")

    # Next, check that the spatial dimensions match up
    @assert x[1:end-2] == input_size(udims) DimensionMismatch("Data input spatial size ($(x[1:end-2]) vs. $(input_size(udims)))")
    @assert y[1:end-2] == output_size(udims) DimensionMismatch("Data output spatial size ($(y[1:end-2]) vs. $(output_size(udims)))")

    # Finally, check that the batch size matches
    @assert x[end] == y[end] DimensionMismatch("Batch size ($(x[end]) vs. $(y[end]))")
end

# Override show() for these beauties
function Base.show(io::IO, udims::UpsampleDims)
    I = (input_size(udims)..., channels_in(udims))
    O = (output_size(udims)..., channels_out(udims))
    S = stride(udims)
    Sc = scale(udims)
    print(io, "UpsampleDims: $I -> $O, stride: $S scale: $Sc")
end
