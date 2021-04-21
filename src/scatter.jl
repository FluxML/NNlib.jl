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

typelength(::Type{<:Number}) = 1
typelength(::Type{<:NTuple{M}}) where M = M
typelength(::Type{CartesianIndex{M}}) where M = M

function _check_dims(X::AbstractArray{Tx,Nx}, 
                     Y::AbstractArray{Ty,Ny},
                     idx::AbstractArray{Tidx,Nidx}) where
                     {Tx,Ty,Tidx<:IntOrIntTuple,Nx,Ny,Nidx}
    M = typelength(Tidx)
    dims = _check_dims(Nx, Ny, M, Nidx)
    size(X)[1:dims] == size(Y)[1:dims] || throw(ArgumentError("Incompatible input shapes."))
    size(Y)[dims+1:end] == size(idx) || throw(ArgumentError("Incompatible input shapes."))
    return dims
end

function _check_dims(X::AbstractArray{Tx,Nx}, 
                     Y::AbstractArray{Ty,Ny},
                     idx::AbstractArray{CartesianIndex{M},Nidx}) where {Tx,Ty,Nx,Ny,M,Nidx}
    dims = _check_dims(Nx, Ny, M, Nidx)
    size(X)[1:dims] == size(Y)[1:dims] || throw(ArgumentError("Incompatible input shapes."))
    size(Y)[dims+1:end] == size(idx) || throw(ArgumentError("Incompatible input shapes."))
    return dims
end

function _check_dims(Nx, Ny, M, Nidx)
    @assert Nx - M == Ny - Nidx "Incompatible input shapes of (dst, src, idx) = ($Nx, $Ny, $Nidx)."
    dims = Nx - M
    dims < 0 && throw(ArgumentError("dims must be non-negative but got dims=$dims."))
    return dims
end

_view(X, colons, k) = view(X, colons..., k...)
_view(X, colons, k::Union{Integer, CartesianIndex}) = view(X, colons..., k)

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
function scatter!(op, dst::AbstractArray, src::AbstractArray, idx::AbstractArray)
    dims = _check_dims(dst, src, idx)
    colons = Base.ntuple(_->Colon(), dims)
    for k in CartesianIndices(idx)
        dst_v = _view(dst, colons, idx[k])
        src_v = _view(src, colons, k)
        dst_v .= (op).(dst_v, src_v)
    end
    dst
end

function scatter!(op::typeof(mean), dst::AbstractArray, src::AbstractArray, idx::AbstractArray)
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
                           src::AbstractArray{Tsrc,Nsrc},
                           idx::AbstractArray{Tidx,Nidx}) where {Tsrc,Tidx,Nsrc,Nidx}
        dims = Nsrc - Nidx
        dstsize = (size(src)[1:dims]..., maximum_dims(idx)...)
        dst = similar(src, Tsrc, dstsize)
        fill!(dst, Base.reduce_empty(+, Tsrc))
        scatter!(op, dst, src, idx)
    end
end

for op in [*, /]
    @eval function scatter(op::typeof($op),
                           src::AbstractArray{Tsrc,Nsrc},
                           idx::AbstractArray{Tidx,Nidx}) where {Tsrc,Tidx,Nsrc,Nidx}
        dims = Nsrc - Nidx
        dstsize = (size(src)[1:dims]..., maximum_dims(idx)...)
        dst = similar(src, Tsrc, dstsize)
        fill!(dst, Base.reduce_empty(*, Tsrc))
        scatter!(op, dst, src, idx)
    end
end

function scatter(op::typeof(max),
                 src::AbstractArray{Tsrc,Nsrc},
                 idx::AbstractArray{Tidx,Nidx}) where {Tsrc,Tidx,Nsrc,Nidx}
    dims = Nsrc - Nidx
    dstsize = (size(src)[1:dims]..., maximum_dims(idx)...)
    dst = similar(src, Tsrc, dstsize)
    fill!(dst, typemin(Tsrc))
    scatter!(op, dst, src, idx)
end

function scatter(op::typeof(min),
                 src::AbstractArray{Tsrc,Nsrc},
                 idx::AbstractArray{Tidx,Nidx}) where {Tsrc,Tidx,Nsrc,Nidx}
    dims = Nsrc - Nidx
    dstsize = (size(src)[1:dims]..., maximum_dims(idx)...)
    dst = similar(src, Tsrc, dstsize)
    fill!(dst, typemax(Tsrc))
    scatter!(op, dst, src, idx)
end

function scatter(op::typeof(mean),
                 src::AbstractArray{Tsrc,Nsrc},
                 idx::AbstractArray{Tidx,Nidx}) where {Tsrc,Tidx,Nsrc,Nidx}
    FT = float(Tsrc)
    dims = Nsrc - Nidx
    dstsize = (size(src)[1:dims]..., maximum_dims(idx)...)
    dst = similar(src, Tsrc, dstsize)
    fill!(dst, Base.reduce_empty(+, FT))
    scatter!(op, dst, src, idx)
end
