export gather, gather_indices

"""
    gather!(dst, src, idx)

Reverse operation of scatter, which gather data in `src` to destination according to `idx`.
For each index `k` in `idx`, assign values to `dst` according to

    dst[k...] = src[idx[k]...]

# Arguments
- `dst`: the destination where data would be assigned to.
- `src`: the source data to be assigned.
- `idx`: the mapping for assignment from source to destination.
"""
function gather!(dst::AbstractArray{T,N}, src::AbstractArray{T}, idx::AbstractArray{<:IntOrTuple,N}) where {T,N}
    @assert size(dst) == size(idx) "dst and idx must have the same size."
    for k = CartesianIndices(idx)
        @inbounds dst[k] = src[idx[k]]
    end
    dst
end

"""
    gather(src, idx)

Reverse operation of scatter, which gather data in `src` to destination according to `idx`.
For each index `k` in `idx`, assign values to `dst` according to

    dst[k...] = src[idx[k]...]

where destination `dst` is created according to `idx`.

# Arguments
- `src`: the source data to be assigned.
- `idx`: the mapping for assignment from source to destination.
"""
function gather(src::AbstractArray{T}, idx::AbstractArray{<:IntOrTuple}) where {T}
    dst = similar(src, axes(idx)...)
    gather!(dst, src, idx)
end

function gather_indices(X::Array{T}) where T
    Y = DefaultDict{T,Vector{CartesianIndex}}(CartesianIndex[])
    @inbounds for (ind, val) = pairs(X)
        push!(Y[val], ind)
    end
    Y
end
