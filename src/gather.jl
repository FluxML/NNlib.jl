"""
    gather!(dst, src, idx)

Reverse operation of [`scatter!`](@ref). Gathers data from source `src` 
and writes it in destination `dst` according to the index array `idx`.
For each `k` in `CartesianIndices(idx)`, assign values to `dst` according to

    dst[:, ... , k] .= src[:, ... , idx[k]...]

Notice that if `idx` is a vector containing integers,
and both `dst` and `src` are matrices, previous expression simplifies to

    dst[:, k] .= src[:, idx[k]]

and `k` will run over `1:length(idx)`. 

The elements of `idx` can be integers or integer tuples and may be repeated. 
A single `src` column can end up being copied into zero, one, 
or multiple `dst` columns.

See [`gather`](@ref) for an allocating version.
"""
function gather!(dst::AbstractArray, src::AbstractArray, idx::AbstractArray)
    dims = _check_dims(src, dst, idx)
    colons = ntuple(i -> Colon(), dims)
    for k in CartesianIndices(idx)
        _view(dst, colons, k) .= _view(src, colons, idx[k])
    end
    return dst
end

"""
    gather(src, idx) -> dst

Reverse operation of [`scatter`](@ref). Gathers data from source `src` 
and writes it in a destination `dst` according to the index
array `idx`.
For each `k` in `CartesianIndices(idx)`, assign values to `dst` 
according to

    dst[:, ... , k] .= src[:, ... , idx[k]...]

Notice that if `idx` is a vector containing integers
and `src` is a matrix, previous expression simplifies to

    dst[:, k] .= src[:, idx[k]]

and `k` will run over `1:length(idx)`. 

The elements of `idx` can be integers or integer tuples and may be repeated. 
A single `src` column can end up being copied into zero, one, 
or multiple `dst` columns.

See [`gather!`](@ref) for an in-place version.
"""
function gather(src::AbstractArray{Tsrc, Nsrc}, 
                idx::AbstractArray{Tidx, Nidx}) where 
                    {Tsrc, Nsrc, Nidx, Tidx}

    M = typelength(Tidx) 
    dstsize = (size(src)[1:Nsrc-M]..., size(idx)...)
    dst = similar(src, Tsrc, dstsize)
    return gather!(dst, src, idx)
end

# Simple implementation with getindex for integer array.
# Perf equivalent to the one above (which can also handle the integer case)
# leave it here to show the simple connection with getindex.
function gather(src::AbstractArray{Tsrc, Nsrc}, 
                idx::AbstractArray{<:Integer}) where {Tsrc, Nsrc}
    colons = ntuple(i -> Colon(), Nsrc-1)
    return src[colons..., idx]
end
