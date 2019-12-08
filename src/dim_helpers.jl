# Various helper functions to calculate dimensions for operations
include("dim_helpers/AbstractDims.jl")
include("dim_helpers/ConvDims.jl")
include("dim_helpers/PoolDims.jl")


"""
    transpose_swapbatch(x::AbstractArray)

Given an AbstractArray, swap its batch and channel axes, as we must during transposed
convolution.  We do this to the operands during convolution, and then again to the
output once we're done.
"""
function transpose_swapbatch(x::AbstractArray)
    return permutedims(x, ((1:(ndims(x)-2))..., ndims(x), ndims(x)-1))
end
function transpose_swapbatch(x::Tuple)
    return (x[1:end-2]..., x[end], x[end-1])
end

"""
    transpose_pad(cdims::ConvDims)

Transposed convolution can be calculated in terms of typical convolution with some extra
padding.  This method computes the padding of the convolution that would result in the
transposed convolution of two operands, in essence taking care of that "extra padding".
Note that this method should almost always be accompanied by a call that predilates one
of the operands.
"""
function transpose_pad(cdims::ConvDims)
    I = input_size(cdims)
    K = kernel_size(cdims)
    D = dilation(cdims)
    P = padding(cdims)
    S = stride(cdims)
    return ntuple(length(P)) do i
        hi = ceil(Int, i/2)
        if mod(i, 2) == 1
            return (K[hi] - 1)*D[hi] - P[i]
        else
            return (K[hi] - 1)*D[hi] - P[i] + mod(I[hi] + P[i-1] + P[i] - (K[hi] - 1)*D[hi] - 1, S[hi])
        end
    end
end

"""
    insert_singleton_spatial_dimension(cdims::ConvDims)

When converting a 1d convolution to a 2d, or a 2d to a 3d, we need to insert a singleton
spatial dimension at the end of the spatial dimensions.  This does so for a ConvDims.
"""
@inline function insert_singleton_spatial_dimension(cdims::C) where {C <: ConvDims}
    return basetype(C)(cdims;
        N=spatial_dims(cdims) + 1,
        I=(input_size(cdims)..., 1),
        K=(kernel_size(cdims)..., 1),
        S=(stride(cdims)..., 1),
        # Padding is always the problem child....
        P=(padding(cdims)..., 0, 0),
        D=(dilation(cdims)..., 1),
    )
end

# We specialize common cases
@inline function insert_singleton_spatial_dimension(x::AbstractArray{T,3}) where {T}
    return reshape(x, size(x,1), 1, size(x,2), size(x,3))
end
@inline function insert_singleton_spatial_dimension(x::AbstractArray{T,4}) where {T}
    return reshape(x, size(x,1), size(x,2), 1, size(x,3), size(x,4))
end

# Helper to do this as many times as needed
@inline function insert_singleton_spatial_dimension(x, reps::Int)
    for r in 1:reps
        x = insert_singleton_spatial_dimension(x)
    end
    return x
end

"""
    predilated_size(x_size::Tuple, dilation::Tuple)

Calculate the size of a predilated `x` given a particular dilation factor.  This is used
within `predilate()` and `transpose_cdims()`.
"""
function predilated_size(x_size::NTuple{N}, dilation::NTuple{M}) where {N, M}
    @assert (M == N - 2) DimensionMismatch("len(dilation) != number of spatial dims")
    return ntuple(N) do idx
        if idx <= N - 2
            return (x_size[idx] - 1)*dilation[idx] + 1
        else
            x_size[idx]
        end
    end
end

"""
    predilate(x, dilation::Tuple)

Places elements of `x` within a lattice of zeros, used in expressing a transposed
convolution in terms of normal convolution.  Note that while we call this "predilation"
for aesthetic reasons, you are typically passing a "stride" value into here.  Yes,
transposed convolution is confusing.
"""
function predilate(x::AbstractArray{T,N}, dilation::NTuple{M}) where {T, N, M}
    @assert (M == N - 2) DimensionMismatch("len(dilation) != number of spatial dims")

    # If there is no dilation to be done, then ignore it.
    if all(dilation .== 1)
        return x
    end

    # Validate dilation factors
    for idx in 1:length(dilation)
        @assert dilation[idx] >= 1 ArgumentError("dilation cannot be less than 1")
    end

    # Create new x that is bigger and holier
    x_dil = zeros(eltype(x), predilated_size(size(x), dilation))

    # Fill in strategic locations within `x_dil`, such that there are `dilation[idx] - 1`
    # zeros between each element of `x` along each spatial dimension.
    x_dil[(1:dilation[idx]:size(x_dil,idx) for idx in 1:(N-2))..., :, :] .= x
    return x_dil
end

"""
    flipweight(w::AbstractArray)

Reorders the weight tensor for supporting both convolution and cross-correlation operations.
"""

# For any array with ndims <= 3 it makes no sense to flip the weights so simply return the
# original array
@inline flipweight(w::AbstractArray) = w

@inline flipweight(w::AbstractArray{T, 4}) where {T} = w[end:-1:1, end:-1:1, :, :]

@inline flipweight(w::AbstractArray{T, 5}) where {T} = w[end:-1:1, end:-1:1, end:-1:1, :, :]
