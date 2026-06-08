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
function gather(
    src::AbstractArray{Tsrc, Nsrc}, idx::AbstractArray{Tidx, Nidx},
) where {Tsrc, Nsrc, Nidx, Tidx}
    M = typelength(Tidx)
    dstsize = (size(src)[1:Nsrc-M]..., size(idx)...)
    dst = dense_like(src, Tsrc, dstsize)
    return gather!(dst, src, idx)
end

"""
    gather(src, IJK...)

Convert the tuple of integer vectors `IJK` to a tuple of `CartesianIndex` and
call `gather` on it: `gather(src, CartesianIndex.(IJK...))`.

# Examples

```jldoctest
julia> src = reshape([1:15;], 3, 5)
3×5 Matrix{Int64}:
 1  4  7  10  13
 2  5  8  11  14
 3  6  9  12  15

julia> NNlib.gather(src, [1, 2], [2, 4])
2-element Vector{Int64}:
  4
 11
```
"""
function gather(
    src::AbstractArray{Tsrc, Nsrc},
    I::AbstractVector{<:Integer}, J::AbstractVector{<:Integer},
    Ks::AbstractVector{<:Integer}...,
) where {Nsrc, Tsrc}
    return gather(src, to_cartesian_index(I, J, Ks...))
end

to_cartesian_index(IJK...) = CartesianIndex.(IJK...)

@non_differentiable to_cartesian_index(::Any...)
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
    gather!(dst, src, IJK...)

Convert the tuple of integer vectors `IJK` to a tuple of `CartesianIndex` and
call `gather!` on it: `gather!(dst, src, CartesianIndex.(IJK...))`.

# Examples

```jldoctest
julia> src = reshape([1:15;], 3, 5)
3×5 Matrix{Int64}:
 1  4  7  10  13
 2  5  8  11  14
 3  6  9  12  15

julia> dst = zeros(Int, 2);

julia> NNlib.gather!(dst, src, [1, 2], [2, 4])
2-element Vector{Int64}:
  4
 11
```
"""
function gather!(
    dst::AbstractArray, src::AbstractArray,
    I::AbstractVector{<:Integer}, J::AbstractVector{<:Integer},
    Ks::AbstractVector{<:Integer}...,
)
    return gather!(dst, src, to_cartesian_index(I, J, Ks...))
end

function gather!(dst::AnyGPUArray, src::AnyGPUArray, idx::AnyGPUArray)
    isempty(dst) && return dst
    n_dims = scatter_dims(src, dst, idx)
    checkbounds_gather(src, idx, n_dims)
    dims = size(src)[1:n_dims]
    max_dims_idx = prod(dims)
    ndrange = max_dims_idx * length(idx)
    _gather!(KernelAbstractions.get_backend(src))(
        dst, src, idx, CartesianIndices(dims), max_dims_idx; ndrange)
    return dst
end

@kernel function _gather!(
    dst, @Const(src), @Const(idx),
    dim_ids::CartesianIndices, max_dims_idx::Int,
)
    i = @index(Global)
    j, k = divrem(i - 1, max_dims_idx)
    @inbounds dst[i] = src[dim_ids[k + 1], Tuple(idx[j + 1])...]
end

@inline _idx_in_bounds(i::Integer, sz::Tuple) = (1 ≤ i) & (i ≤ sz[1])
@inline _idx_in_bounds(i::Union{Tuple, CartesianIndex}, sz::Tuple) =
    all(map((j, s) -> (1 ≤ j) & (j ≤ s), Tuple(i), sz))

# The GPU kernel reads `src` with `@inbounds`, so out-of-range indices would
# silently return garbage. Validate the indices up front to mirror the
# `BoundsError` thrown by the CPU `view`-based `gather!`. See issue #416.
function checkbounds_gather(src::AbstractArray, idx::AbstractArray, n_dims::Int)
    idx_size = size(src)[n_dims + 1:end]
    inbounds = mapreduce(i -> _idx_in_bounds(i, idx_size), &, idx; init=true)
    inbounds || throw(BoundsError(src))
    return nothing
end

∇gather_src(Δ, src_size, idx) = scatter!(+, fill!(similar(Δ, eltype(Δ), src_size), 0), Δ, idx)

function rrule(::typeof(gather!), dst::AbstractArray, src::AbstractArray, idx::AbstractArray)
    y = gather!(dst, src, idx)
    src_size = size(src)
    gather!_pullback(Δ) = (NoTangent(), NoTangent(), ∇gather_src(unthunk(Δ), src_size, idx), NoTangent())
    return y, gather!_pullback
end
