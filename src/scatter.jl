## Scatter API
#   - Scatter:
#     - scatter(op, src, idx)
#     - scatter!(op, dst, src, idx)
#   - Scatter destination backpropagation
#     - ∇scatter!_dst
#   - Scatter source backpropagation
#     - ∇scatter_src
#     - ∇scatter!_src
#

typelength(::Type{<:Number}) = 1
typelength(::Type{<:NTuple{M}}) where M = M
typelength(::Type{CartesianIndex{M}}) where M = M

"""
Performs dimensional consistency checks and return the 
dimensionality of the scattered objects.
"""
function scatter_dims(X::AbstractArray{Tx,Nx}, 
                     Y::AbstractArray{Ty,Ny},
                     idx::AbstractArray{Tidx,Nidx}) where {Tx,Ty,Tidx,Nx,Ny,Nidx}
    M = typelength(Tidx)
    dims = scatter_dims(Nx, Ny, M, Nidx)
    size(X)[1:dims] == size(Y)[1:dims] || throw(ArgumentError("Incompatible input shapes."))
    size(Y)[dims+1:end] == size(idx) || throw(ArgumentError("Incompatible input shapes."))
    return dims
end

function scatter_dims(Nx, Ny, M, Nidx)
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
A binary reduction operator `op` is applied during the scatter. 
For each index `k` in `idx`, accumulates values in `dst` according to

    dst[:, ..., idx[k]...] = (op).(dst[:, ..., idx[k]...], src[:, ..., k...])

# Arguments

- `op`: Operations to be applied on `dst` and `src`, e.g. `+`, `-`, `*`, `/`, `max`, `min` and `mean`.
- `dst`: The destination for `src` to aggregate to. This argument will be mutated.
- `src`: The source data for aggregating.
- `idx`: The mapping for aggregation from source (index) to destination (value). 
         The `idx` array can contain either integers or tuples.
"""
function scatter!(op, dst::AbstractArray, src::AbstractArray, idx::AbstractArray)
    dims = scatter_dims(dst, src, idx)
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

If `init` is provided, it is used to initialize the content of `dst`.
Otherwise, the init values is inferred from the reduction operator `op`
for some common operators (e.g. `init = 0` for `op = +`). 

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

∇scatter!_src(op, Δ, dst, src, idx) = ∇scatter_src(op, Δ, dst, src, idx) 
∇scatter!_dst(op, Δ, dst, y) = Δ

∇scatter!_dst(op::Union{typeof(max),typeof(min)}, Δ, dst_old, dst) = 
    (dst_old .== op.(dst_old, dst)) .* Δ

modify_src(::typeof(+), X) = X
modify_src(::typeof(-), X) = -X
modify_src(::typeof(*), X, Y) = X
modify_src(::typeof(/), X, Y) = .-X ./ Y.^2

∇scatter_src(op::Union{typeof(+),typeof(-)}, Δ, dst, src, idx) = modify_src(op, gather(Δ, idx))

∇scatter!_src(op::Union{typeof(*),typeof(/)}, Δ, dst, src, idx) = 
    gather(dst, idx) .* ∇scatter_src(op, Δ, dst, src, idx)

function ∇scatter_src(op::Union{typeof(*),typeof(/)}, Δ, dst,
                      src::AbstractArray{Tsrc,Nsrc}, 
                      idx::AbstractArray{Tidx,Nidx}) where {Tsrc,Tidx,Nsrc,Nidx}
    dims = Nsrc - Nidx
    Δsrc = modify_src(op, gather(Δ, idx), src)
    rev_idx = reverse_indices(idx)
    ax = CartesianIndices(axes(src)[1:dims])
    for k in CartesianIndices(idx)
        inds = filter(x -> x != k, rev_idx[idx[k]])
        for i in ax
            Δsrc[i, k] = op(Δsrc[i, k], prod(j -> src[i, j], inds))
        end
    end
    Δsrc
end

∇scatter_src(::Union{typeof(max),typeof(min)}, Δ, dst, src, idx) = (src .== gather(dst, idx)) .* gather(Δ, idx)

function ∇scatter_src(::typeof(mean), Δ, dst,
                    src::AbstractArray{Tsrc,Nsrc}, 
                    idx::AbstractArray{Tidx,Nidx}) where {Tsrc,Tidx,Nsrc,Nidx}
    dims = Nsrc - Nidx
    divide_by_counts!(gather(Δ, idx), idx, dims)
end

function rrule(::typeof(scatter!), op, dst::AbstractArray, src::AbstractArray, idx::AbstractArray)
    dst_old = copy(dst)
    scatter!(op, dst, src, idx)
    scatter!_pullback(Δ) = (NoTangent(), NoTangent(), ∇scatter!_dst(op, unthunk(Δ), dst_old, dst), ∇scatter!_src(op, unthunk(Δ), dst, src, idx), NoTangent())
    dst, scatter!_pullback
end

function rrule(::typeof(scatter), op, src::AbstractArray, idx::AbstractArray)
    y = scatter(op, src, idx)
    scatter_pullback(Δ) = (NoTangent(), NoTangent(), ∇scatter_src(op, unthunk(Δ), y, src, idx), NoTangent())
    y, scatter_pullback
end
