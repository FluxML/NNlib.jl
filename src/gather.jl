"""
    NNlib.gather!(dst, src, idx)

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
    dims = scatter_dims(src, dst, idx)
    colons = ntuple(i -> Colon(), dims)
    for k in CartesianIndices(idx)
        _view(dst, colons, k) .= _view(src, colons, idx[k])
    end
    return dst
end

"""
    NNlib.gather(src, idx) -> dst

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

# Examples
```jldoctest
julia> NNlib.gather([1,20,300,4000], [2,4,2])
3-element Vector{Int64}:
   20
 4000
   20

julia> NNlib.gather([1 2 3; 4 5 6], [1,3,1,3,1])
2×5 Matrix{Int64}:
 1  3  1  3  1
 4  6  4  6  4
```
"""
function gather(src::AbstractArray{Tsrc, Nsrc}, 
                idx::AbstractArray{Tidx, Nidx}) where 
                    {Tsrc, Nsrc, Nidx, Tidx}

    M = typelength(Tidx) 
    dstsize = (size(src)[1:Nsrc-M]..., size(idx)...)
    dst = similar(src, Tsrc, dstsize)
    return gather!(dst, src, idx)
end

∇gather_src(Δ, src_size, idx) = scatter!(+, fill!(similar(Δ, eltype(Δ), src_size), 0), Δ, idx)

function rrule(::typeof(gather!), dst::AbstractArray, src::AbstractArray, idx::AbstractArray)
    y = gather!(dst, src, idx)
    src_size = size(src)
    gather!_pullback(Δ) = (NoTangent(), NoTangent(), ∇gather_src(unthunk(Δ), src_size, idx), NoTangent())
    y, gather!_pullback
end
