export gather, gather!

"""
    gather!(dst, src, idx, dims)

Reverse operation of scatter, which gather data in `src` to destination according to `idx`.
For each index `k` in `idx`, assign values to `dst` according to

    dst[k...] = src[idx[k]...]

# Arguments
- `dst`: the destination where data would be assigned to.
- `src`: the source data to be assigned.
- `idx`: the mapping for assignment from source to destination.
- `dims`: the number of dimensions to be view as an operational unit from the beginning.
If dims=0 is given, operational unit is scalar. Default dims is 1.

Examples for dims are lists here:

- `dims=0`: dst[k...] = src[idx[k]...]
- `dims=1`: dst[:, k...] .= src[:, idx[k]...]
- `dims=2`: dst[:, :, k...] .= src[:, :, idx[k]...]
"""
function gather!(dst::AbstractArray{T,N}, src::AbstractArray{T}, idx::AbstractArray{<:IntOrTuple,N};
                 dims::Integer=1) where {T,N}
    @boundscheck _check_output(idx, dst, src, dims)
    @boundscheck _check_input(idx, src)
    if dims > 0
        gather_vec!(dst, src, idx, Val(dims))
    elseif dims == 0
        gather_scl!(dst, src, idx)
    else
        throw(ArgumentError("dims must be non-negative but got dims=$dims"))
    end
end

@inbounds function gather_scl!(dst::AbstractArray{T}, src::AbstractArray{T}, idx::AbstractArray{<:IntOrTuple}) where {T<:Real}
    for k = CartesianIndices(idx)
        dst[k] = src[idx[k]]
    end
    dst
end

function gather_vec!(dst::AbstractArray{T}, src::AbstractArray{T}, idx::AbstractArray{<:IntOrTuple},
                     dims::Integer) where {T<:Real}
    colons = Base.ntuple(_->Colon(), dims)
    @simd for k = CartesianIndices(idx)
        @inbounds view(dst, colons..., k) .= view(src, colons..., idx[k]...)
    end
    dst
end

"""
    gather(src, idx, dims)

Reverse operation of scatter, which gather data in `src` to destination according to `idx`.
For each index `k` in `idx`, assign values to `dst` according to

    dst[k...] = src[idx[k]...]

where destination `dst` is created according to `idx`.

# Arguments
- `src`: the source data to be assigned.
- `idx`: the mapping for assignment from source to destination.
- `dims`: the number of dimensions to be view as an operational unit from the beginning.
If dims=0 is given, operational unit is scalar. Default dims is 1.

Examples for dims are lists here:

- `dims=0`: dst[k...] = src[idx[k]...]
- `dims=1`: dst[:, k...] .= src[:, idx[k]...]
- `dims=2`: dst[:, :, k...] .= src[:, :, idx[k]...]
"""
function gather(src::AbstractArray{T}, idx::AbstractArray{<:IntOrTuple}; dims::Integer=1) where {T}
    dst = similar(src, axes(idx)...)
    gather!(dst, src, idx; dims=dims)
end
