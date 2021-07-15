export DenseConvDims

"""
    DenseConvDims

Concrete subclass of `ConvDims` for a normal, dense, conv2d/conv3d.
"""
struct DenseConvDims{N,K,C_in,C_out,G,S,P,D,F} <: ConvDims{N,S,P,D,F}
    I::NTuple{N,Int}
end

# Getters for the fields
input_size(c::DenseConvDims) = c.I
kernel_size(::DenseConvDims{N,K,C_in,C_out}) where {N,K,C_in,C_out} = K
channels_in(::DenseConvDims{N,K,C_in,C_out}) where {N,K,C_in,C_out} = C_in::Int
channels_out(::DenseConvDims{N,K,C_in,C_out,G}) where {N,K,C_in,C_out,G} = (C_out*G)::Int
groupcount(::DenseConvDims{N,K,C_in,C_out,G}) where {N,K,C_in,C_out,G} = G::Int

# Convenience wrapper to create DenseConvDims objects
function DenseConvDims(x_size::NTuple{M}, w_size::NTuple{M};
                       stride=1, padding=0, dilation=1, 
                       flipkernel::Bool=false, groups=1) where M
    # Do common parameter validation
    stride, padding, dilation = check_spdf(x_size, w_size, stride, padding, dilation)

    # Ensure channels are equal
    if x_size[end-1] != w_size[end-1] * groups
        xs = x_size[end-1]
        ws = w_size[end-1] * groups
        throw(DimensionMismatch("Input channels must match! ($xs vs. $ws)"))
    end
    
    (w_size[end] % groups) != 0 && 
        throw(DimensionMismatch("Output channels must be divisible by groups' number"))
    
    
    # The type parameters are what 
    return DenseConvDims{
        M - 2,
        w_size[1:end-2],
        x_size[end-1],
        w_size[end],
        groups,
        stride,
        padding,
        dilation,
        flipkernel
    }(
        # Input spatial size
        x_size[1:end-2],
    )
end

# Auto-extract sizes and sub out to big brother above
function DenseConvDims(x::AbstractArray, w::AbstractArray; kwargs...)
    if ndims(x) != ndims(w)
        throw(DimensionMismatch("Rank of x and w must match! ($(ndims(x)) vs. $(ndims(w)))"))
    end
    return DenseConvDims(size(x), size(w); kwargs...)
end

# Useful for constructing a new DenseConvDims that has only a few elements different
# from the original progenitor object that it inherits shapes from.
function DenseConvDims(c::ConvDims; N=spatial_dims(c), I=input_size(c), K=kernel_size(c),
                       C_in=channels_in(c), C_out=channels_out(c), S=stride(c),
                       P=padding(c), D=dilation(c), F=flipkernel(c), G=groupcount(c))
    return DenseConvDims{N, K, C_in, C_out, G, S, P, D, F}(I)
end

function check_dims(x::NTuple{M}, w::NTuple{M}, y::NTuple{M}, cdims::DenseConvDims) where {M}
    g = groupcount(cdims)
    
    # First, check that channel counts are all correct:
    @assert x[M-1] == channels_in(cdims) DimensionMismatch("Data input channel count ($(x[M-1]) vs. $(channels_in(cdims)))")
    @assert y[M-1] == channels_out(cdims) DimensionMismatch("Data output channel count ($(y[M-1]) vs. $(channels_out(cdims)))")
    @assert x[M-1] % g == 0  DimensionMismatch("Data input channel count ($(x[M-1]) vs. $(groupcount(cdims)) groups)")
    @assert y[M-1] % g == 0  DimensionMismatch("Data output channel count ($(y[M-1]) vs. $(groupcount(cdims)) groups)")
    @assert w[M-1] * g == channels_in(cdims) DimensionMismatch("Kernel input channel count ($(w[M-1]) vs. $(channels_in(cdims)))")
    @assert w[M] * g == channels_out(cdims) DimensionMismatch("Kernel output channel count ($(w[M]) vs. $(channels_out(cdims)))")

    # Next, check that the spatial dimensions match up
    @assert x[1:M-2] == input_size(cdims) DimensionMismatch("Data input spatial size ($(x[1:M-2]) vs. $(input_size(cdims)))")
    @assert y[1:M-2] == output_size(cdims) DimensionMismatch("Data output spatial size ($(y[1:M-2]) vs. $(output_size(cdims)))")
    @assert w[1:M-2] == kernel_size(cdims) DimensionMismatch("Kernel spatial size ($(w[1:M-2]) vs. $(kernel_size(cdims)))")

    # Finally, check that the batch size matches
    @assert x[M] == y[M] DimensionMismatch("Batch size ($(x[M]) vs. $(y[M]))")
end
