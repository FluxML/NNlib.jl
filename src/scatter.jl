export scatter!, scatter

## Scatter API
#   - Scatter:
#     - scatter(op, src, idx)
#     - scatter!(op, dst, src, idx)
#   - Scatter destination backpropagation
#     - ∇scatter_dst!
#   - Scatter source backpropagation
#     - ∇scatter_src
#     - ∇scatter_src!
#

function _check_dims(Ndst, Nsrc, N, Nidx)
    @assert Ndst - N == Nsrc - Nidx
    dims = Ndst - N
    if dims < 0
        throw(ArgumentError("dims must be non-negative but got dims=$dims"))
    end
    return dims
end

_check_input(idx::AbstractArray{<:Integer}, arr) = checkbounds(arr, minimum(idx):maximum(idx))

function _check_input(idx::AbstractArray{<:Tuple}, arr)
    pairs = map(xs -> Base.OneTo(maximum(xs)), zip(idx...))
    checkbounds(arr, pairs...)
end

function _check_output(idx::AbstractArray{<:IntOrTuple}, arr, src, dims)
    pre_dims = axes(src)[1:dims]
    post_dims = Base.OneTo.(maximum_dims(idx))
    checkbounds(arr, pre_dims..., post_dims...)
end


"""
    scatter!(op, dst, src, idx)

Scatter operation, which scatters data in `src` and assigns to `dst` according to `idx`.
With the data going to the same place, specified operation is applied on to reduce data.
For each index `k` in `idx`, accumulate values in `dst` according to

    dst[idx[k]...] = (op).(dst[idx[k]...], src[k...])

# Arguments
- `op`: operations to be applied on `dst` and `src`, e.g. `+`, `-`, `*`, `/`, `max` and `min`.
- `dst`: the destination for `src` to aggregate to. This argument will be mutated.
- `src`: the source data for aggregating.
- `idx`: the mapping for aggregation from source (index) to destination (value).
The index of `idx` is corresponding to the index of `src` and the value of `idx` is
corresponding to the index of `dst`. The value of `idx` can be `Int` or `Tuple` type.

    dst[:, ..., idx[k]...] = (op).(dst[:, ..., idx[k]...], src[:, ..., k...])
"""
function scatter!(op,
                  dst::AbstractArray{Tdst,Ndst},
                  src::AbstractArray{Tsrc,Nsrc},
                  idx::AbstractArray{<:Integer,Nidx}) where {Tdst,Tsrc,Ndst,Nsrc,Nidx}
    dims = _check_dims(Ndst, Nsrc, 1, Nidx)
    @boundscheck _check_output(idx, dst, src, dims)
    @boundscheck _check_input(idx, src)
    scatter!(op, dst, src, idx, Val(dims))
end

function scatter!(op,
                  dst::AbstractArray{Tdst,Ndst},
                  src::AbstractArray{Tsrc,Nsrc},
                  idx::AbstractArray{NTuple{N,Int},Nidx}) where {Tdst,Tsrc,Ndst,Nsrc,N,Nidx}
    dims = _check_dims(Ndst, Nsrc, N, Nidx)
    @boundscheck _check_output(idx, dst, src, dims)
    @boundscheck _check_input(idx, src)
    scatter!(op, dst, src, idx, Val(dims))
end

function scatter!(op, dst::AbstractArray{Tdst}, src::AbstractArray{Tsrc}, idx::AbstractArray{<:IntOrTuple},
                  dims::Val{N}) where {Tdst,Tsrc,N}
    colons = Base.ntuple(_->Colon(), N)
    for k in CartesianIndices(idx)
        dst_v = view(dst, colons..., idx[k]...)
        src_v = view(src, colons..., k)
        dst_v .= (op).(dst_v, src_v)
    end
    dst
end

"""
    scatter!(mean, dst, src, idx)

Scatter mean operation, which scatters data in `src` and assigns to `dst` according to `idx`.
With the data going to the same place, mean is applied on to reduce data.
For each index `k` in `idx`, accumulate values in `dst` according to

    dst[idx[k]...] = dst[idx[k]...] + mean.(src[k...])

# Arguments
- `dst`: the destination for `src` to aggregate to. This argument will be mutated.
- `src`: the source data for aggregating.
- `idx`: the mapping for aggregation from source (index) to destination (value).
The index of `idx` is corresponding to the index of `src` and the value of `idx` is
corresponding to the index of `dst`. The value of `idx` can be `Int` or `Tuple` type.

    dst[:, ..., idx[k]...] = (op).(dst[:, ..., idx[k]...], src[:, ..., k...])
"""
function scatter!(op::typeof(mean),
                  dst::AbstractArray{Tdst,Ndst},
                  src::AbstractArray{Tsrc,Nsrc},
                  idx::AbstractArray{<:Integer,Nidx}) where {Tdst,Tsrc,Ndst,Nsrc,Nidx}
    Ns = scatter!(+, zero(dst), one.(src), idx)
    dst_ = scatter!(+, zero(dst), src, idx)
    dst .+= safe_div.(dst_, Ns)
    return dst
end

function scatter!(op::typeof(mean),
                  dst::AbstractArray{Tdst,Ndst},
                  src::AbstractArray{Tsrc,Nsrc},
                  idx::AbstractArray{NTuple{N,Int},Nidx}) where {Tdst,Tsrc,Ndst,Nsrc,N,Nidx}
    Ns = scatter!(+, zero(dst), one.(src), idx)
    dst_ = scatter!(+, zero(dst), src, idx)
    dst .+= safe_div.(dst_, Ns)
    return dst
end


"""
    scatter(op, src, idx)

Scatter operation, which applies specified operation on `src` according to `idx`
and gives an new array `dst`.
For each index `k` in `idx`, accumulate values in `dst` according to

    dst[idx[k]...] = (op).(src[k...])

# Arguments
- `op`: operations to be applied on `dst` and `src`, e.g. `+`, `-`, `*`, `/`, `max` and `min`.
- `src`: the source data for aggregating.
- `idx`: the mapping for aggregation from source (index) to destination (value).
The index of `idx` is corresponding to the index of `src` and the value of `idx` is
corresponding to the index of `dst`. The value of `idx` can be `Int` or `Tuple` type.
"""
function scatter end

for op in [+, -]
    @eval function scatter(op::typeof($op),
                           src::AbstractArray{T,Nsrc},
                           idx::AbstractArray{<:IntOrTuple,Nidx}) where {T,Nsrc,Nidx}
        dims = Nsrc - Nidx
        dst = zeros(T, size(src)[1:dims]..., maximum_dims(idx)...)
        scatter!(op, dst, src, idx)
    end
end

for op in [*, /]
    @eval function scatter(op::typeof($op),
                           src::AbstractArray{T,Nsrc},
                           idx::AbstractArray{<:IntOrTuple,Nidx}) where {T,Nsrc,Nidx}
        dims = Nsrc - Nidx
        dst = ones(T, size(src)[1:dims]..., maximum_dims(idx)...)
        scatter!(op, dst, src, idx)
    end
end

function scatter(op::typeof(max),
                 src::AbstractArray{T,Nsrc},
                 idx::AbstractArray{<:IntOrTuple,Nidx}) where {T,Nsrc,Nidx}
    dims = Nsrc - Nidx
    dst = fill(typemin(T), size(src)[1:dims]..., maximum_dims(idx)...)
    scatter!(op, dst, src, idx)
end

function scatter(op::typeof(min),
                 src::AbstractArray{T,Nsrc},
                 idx::AbstractArray{<:IntOrTuple,Nidx}) where {T,Nsrc,Nidx}
    dims = Nsrc - Nidx
    dst = fill(typemax(T), size(src)[1:dims]..., maximum_dims(idx)...)
    scatter!(op, dst, src, idx)
end

function scatter(op::typeof(mean),
                 src::AbstractArray{T,Nsrc},
                 idx::AbstractArray{<:IntOrTuple,Nidx}) where {T,Nsrc,Nidx}
    FT = float(T)
    dims = Nsrc - Nidx
    dst = zeros(FT, size(src)[1:dims]..., maximum_dims(idx)...)
    scatter!(op, dst, FT.(src), idx)
end
