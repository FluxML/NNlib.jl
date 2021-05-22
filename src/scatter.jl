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
    scatter(op, src, idx; [init])

Scatter operation allocating a destination array `dst` and 
calling `scatter!(op, dst, src, idx)` on it.

If `init` is provided it is used to initialized the content of `dst`,
otherwise tries to guess it from the reduction operator `op`
(e.g. `init = 0` for `op = +`). 

See [`scatter!`](@ref) for the details.
"""
function scatter(op,
                src::AbstractArray{Tsrc,Nsrc},
                idx::AbstractArray{Tidx,Nidx};
                init = nothing) where {Tsrc,Tidx,Nsrc,Nidx}
    
    dims = Nsrc - Nidx
    dstsize = (size(src)[1:dims]..., maximum_dims(idx)...)
    dst = similar(src, Tsrc, dstsize)
    xinit = isnothing(init) ? scatter_empty(op, Tsrc) : init 
    fill!(dst, xinit)
    scatter!(op, dst, src, idx)
end

scatter_empty(op, T) = Base.reduce_empty(op, T)
scatter_empty(op::typeof(-), T) = zero(T)
scatter_empty(op::typeof(/), T) = one(T)
scatter_empty(op::typeof(min), T) = typemax(T)
scatter_empty(op::typeof(max), T) = typemin(T)
scatter_empty(op::typeof(mean), T) = zero(T)


## Gradients

opname(::typeof(+)) = :add
opname(::typeof(-)) = :sub
opname(::typeof(*)) = :mul
opname(::typeof(/)) = :div


∇scatter_dst!(op, Δ, dst, y) = Δ

# function ∇scatter_dst!(op::Union{typeof(max),typeof(min)}, Δ, dst, y)
#     mask_y = (dst .== op.(dst, y))
#     mask_y .* Δ
# end

modify_src(::typeof(+), X) = X
modify_src(::typeof(-), X) = -X
modify_src(::typeof(*), X, Y) = X
modify_src(::typeof(/), X, Y) = -X ./ Y.^2

∇src_init!(Δ, idx) = gather(Δ, idx)
∇src_init!(Δ, dst, idx) = gather(dst, idx) .* ∇src_init!(Δ, idx)
∇src_init(Δ, idx) = gather(Δ, idx)

∇scatter_src!(op::Union{typeof(+),typeof(-)}, Δ, dst, src, idx) = modify_src(op, ∇src_init!(Δ, idx))
∇scatter_src(op::Union{typeof(+),typeof(-)}, Δ, dst, src, idx) = modify_src(op, ∇src_init(Δ, idx))

function ∇scatter_src!(op::Union{typeof(*),typeof(/)}, Δ, dst,
                       src::AbstractArray{Tsrc,Nsrc}, 
                       idx::AbstractArray{Tidx,Nidx}) where {Tsrc,Tidx,Nsrc,Nidx}
    dims = Nsrc - Nidx
    Δsrc = modify_src(op, ∇src_init!(Δ, dst, idx), src)
    rev_idx = reverse_indices(idx)
    for k = CartesianIndices(idx)
        inds = filter(x -> x != k, rev_idx[idx[k]])
        for i = CartesianIndices(axes(src)[1:dims])
            Δsrc[i, k] *= prod(j -> src[i, j], inds)
        end
    end
    Δsrc
end

function ∇scatter_src(op::Union{typeof(*),typeof(/)}, Δ, dst,
                      src::AbstractArray{Tsrc,Nsrc}, 
                      idx::AbstractArray{Tidx,Nidx}) where {Tsrc,Tidx,Nsrc,Nidx}
    dims = Nsrc - Nidx
    Δsrc = modify_src(op, ∇src_init(Δ, idx), src)
    rev_idx = reverse_indices(idx)
    for k = CartesianIndices(idx)
        inds = filter(x -> x != k, rev_idx[idx[k]])
        for i = CartesianIndices(axes(src)[1:dims])
            Δsrc[i, k] = op(Δsrc[i, k], prod(j -> src[i, j], inds))
        end
    end
    Δsrc
end

# ∇scatter_src!(op::Union{typeof(max),typeof(min)}, Δ, dst, src, idx) = (src .== op.(src, gather(dst, idx))) .* ∇src_init!(Δ, idx)
# ∇scatter_src(op::Union{typeof(max),typeof(min)}, Δ, dst, src, idx) = (src .== op.(src, gather(dst, idx))) .* ∇src_init(Δ, idx)

∇scatter_src!(::typeof(mean), Δ, idx, dims) = divide_by_counts!(∇src_init!(Δ, idx), idx, dims)
∇scatter_src(::typeof(mean), Δ, idx, dims) = divide_by_counts!(∇src_init(Δ, idx), idx, dims)


function rrule(::typeof(scatter!), op, dst::AbstractArray, src::AbstractArray, idx::AbstractArray)
    y = scatter!(op, copy(dst), src, idx)
    scatter!_pullback(Δ) = (NO_FIELDS, NO_FIELDS, ∇scatter_dst!(op, Δ, dst, y), ∇scatter_src!(op, Δ, dst, src, idx), NoTangent())
    y, scatter!_pullback
end

function rrule(::typeof(scatter), op, src::AbstractArray, idx::AbstractArray)
    y = scatter(op, src, idx)
    scatter_pullback(Δ) = (NO_FIELDS, NO_FIELDS, ∇scatter_src(op, Δ, y, src, idx), NoTangent())
    y, scatter_pullback
end

function rrule(::typeof(scatter!), op::typeof(mean), dst::AbstractArray, src::AbstractArray{Tsrc,Nsrc}, idx::AbstractArray{Tidx,Nidx}) where {Tsrc,Tidx,Nsrc,Nidx}
    dims = Nsrc - Nidx
    y = scatter!(op, copy(dst), src, idx)
    scatter!_pullback(Δ) = (NO_FIELDS, NO_FIELDS, ∇scatter_dst!(op, Δ, dst, y), ∇scatter_src!(op, Δ, idx, dims), NoTangent())
    y, scatter!_pullback
end

function rrule(::typeof(scatter), op::typeof(mean), src::AbstractArray{Tsrc,Nsrc}, idx::AbstractArray{Tidx,Nidx}) where {Tsrc,Tidx,Nsrc,Nidx}
    dims = Nsrc - Nidx
    y = scatter(op, src, idx)
    scatter_pullback(Δ) = (NO_FIELDS, NO_FIELDS, ∇scatter_src(op, Δ, idx, dims), NoTangent())
    y, scatter_pullback
end
