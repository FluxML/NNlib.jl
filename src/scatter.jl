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
    @assert Ndst - N == Nsrc - Nidx "Incompatible input shapes of (dst, src, idx) = ($Ndst, $Nsrc, $Nidx)."
    dims = Ndst - N
    if dims < 0
        throw(ArgumentError("dims must be non-negative but got dims=$dims."))
    end
    return dims
end

typelength(::Type{<:Number}) = 1
typelength(::Type{<:NTuple{M}}) where M = M

"""
    scatter!(op, dst, src, idx)

Scatter operation, which scatters data in `src` and assigns to `dst` according to `idx`.
With the data going to the same place, specified aggregate operation is applied on to reduce
data. For each index `k` in `idx`, accumulate values in `dst` according to

    dst[:, ..., idx[k]...] = (op).(dst[:, ..., idx[k]...], src[:, ..., k...])

# Arguments
- `op`: operations to be applied on `dst` and `src`, e.g. `+`, `-`, `*`, `/`, `max`, `min`
and `mean`.
- `dst`: the destination for `src` to aggregate to. This argument will be mutated.
- `src`: the source data for aggregating.
- `idx`: the mapping for aggregation from source (index) to destination (value).
The index of `idx` is corresponding to the index of `src` and the dimensions of `idx` must
aligned with the last few dimensions of `src`. The value of `idx` is corresponding to the
index of `dst` and the value of `idx` must indicate the last few dimensions of `dst`.
Once the dimensions match, arrays are aligned automatically. The value of `idx` can be
`Int` or `Tuple` type.
"""
function scatter!(op,
                  dst::AbstractArray{Tdst,Ndst},
                  src::AbstractArray{Tsrc,Nsrc},
                  idx::AbstractArray{Tidx,Nidx}) where {Tdst,Tsrc,Tidx<:IntOrIntTuple,Ndst,Nsrc,Nidx}
    M = typelength(Tidx)
    dims = _check_dims(Ndst, Nsrc, M, Nidx)
    scatter!(op, dst, src, idx, Val(dims))
end

function scatter!(op, dst::AbstractArray{Tdst}, src::AbstractArray{Tsrc}, idx::AbstractArray{<:IntOrIntTuple},
                  dims::Val{N}) where {Tdst,Tsrc,N}
    colons = Base.ntuple(_->Colon(), dims)
    for k in CartesianIndices(idx)
        dst_v = view(dst, colons..., idx[k]...)
        src_v = view(src, colons..., k)
        dst_v .= (op).(dst_v, src_v)
    end
    dst
end

function scatter!(op::typeof(mean),
                  dst::AbstractArray{Tdst,Ndst},
                  src::AbstractArray{Tsrc,Nsrc},
                  idx::AbstractArray{<:IntOrIntTuple,Nidx}) where {Tdst,Tsrc,Ndst,Nsrc,Nidx}
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

    dst[:, ..., idx[k]...] = (op).(src[:, ..., k...])

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
                           idx::AbstractArray{<:IntOrIntTuple,Nidx}) where {T,Nsrc,Nidx}
        dims = Nsrc - Nidx
        dstsize = (size(src)[1:dims]..., maximum_dims(idx)...)
        dst = similar(src, T, dstsize)
        fill!(dst, Base.reduce_empty(+, T))
        scatter!(op, dst, src, idx)
    end
end

for op in [*, /]
    @eval function scatter(op::typeof($op),
                           src::AbstractArray{T,Nsrc},
                           idx::AbstractArray{<:IntOrIntTuple,Nidx}) where {T,Nsrc,Nidx}
        dims = Nsrc - Nidx
        dstsize = (size(src)[1:dims]..., maximum_dims(idx)...)
        dst = similar(src, T, dstsize)
        fill!(dst, Base.reduce_empty(*, T))
        scatter!(op, dst, src, idx)
    end
end

function scatter(op::typeof(max),
                 src::AbstractArray{T,Nsrc},
                 idx::AbstractArray{<:IntOrIntTuple,Nidx}) where {T,Nsrc,Nidx}
    dims = Nsrc - Nidx
    dstsize = (size(src)[1:dims]..., maximum_dims(idx)...)
    dst = similar(src, T, dstsize)
    fill!(dst, typemin(T))
    scatter!(op, dst, src, idx)
end

function scatter(op::typeof(min),
                 src::AbstractArray{T,Nsrc},
                 idx::AbstractArray{<:IntOrIntTuple,Nidx}) where {T,Nsrc,Nidx}
    dims = Nsrc - Nidx
    dstsize = (size(src)[1:dims]..., maximum_dims(idx)...)
    dst = similar(src, T, dstsize)
    fill!(dst, typemax(T))
    scatter!(op, dst, src, idx)
end

function scatter(op::typeof(mean),
                 src::AbstractArray{T,Nsrc},
                 idx::AbstractArray{<:IntOrIntTuple,Nidx}) where {T,Nsrc,Nidx}
    FT = float(T)
    dims = Nsrc - Nidx
    dstsize = (size(src)[1:dims]..., maximum_dims(idx)...)
    dst = similar(src, T, dstsize)
    fill!(dst, Base.reduce_empty(+, FT))
    scatter!(op, dst, src, idx)
end
