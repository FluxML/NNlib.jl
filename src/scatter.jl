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
                     idx::AbstractArray{Tidx,Nidx},
                     dims::Union{Nothing, Integer} = nothing) where {Tx,Ty,Tidx,Nx,Ny,Nidx}
    nsrcin = typelength(Tidx)
    ndstin = Nidx
    nbefore, nafter = scatter_dims(Nx, Ny, nsrcin, ndstin, dims)
    size(Y)[1:nbefore] == size(X)[1:nbefore] || throw(ArgumentError("Incompatible input shapes."))
    size(Y)[nbefore+1:nbefore+ndstin] == size(idx) || throw(ArgumentError("Incompatible input shapes."))
    size(Y)[nbefore+ndstin+1:end] == size(X)[nbefore+nsrcin+1:end] || throw(ArgumentError("Incompatible input shapes."))
    return nbefore, nafter
end

function scatter_dims(Nx, Ny, nsrcin, ndstin, dims = nothing)
    @assert Nx - nsrcin == Ny - ndstin "Incompatible input shapes of (dst, src, idx, Tidx) = ($Nx, $Ny, $ndstin, $nsrcin)."
    
    if dims === nothing
        nbefore = Nx - nsrcin
        nbefore < 0 && throw(ArgumentError("nbefore must be non-negative but got $nbefore."))
        nafter = 0
        return nbefore, nafter
    else
        nbefore = dims - 1
        nafter = Ny - ndstin - nbefore
        nbefore < 0 && throw(ArgumentError("nbefore must be non-negative but got $nbefore."))
        nafter < 0 && throw(ArgumentError("nafter must be non-negative but got $nafter."))
        return nbefore, nafter
    end
end

_view(X, colons, k) = view(X, colons..., k...)
_view(X, colons, k::Union{Integer, CartesianIndex}) = view(X, colons..., k)
_view(X, colbefore, k, colafter) = view(X, colbefore..., k..., colafter...)
_view(X, colbefore, k::Union{Integer, CartesianIndex}, colafter) = view(X, colbefore..., k, colafter...)

"""
    NNlib.scatter!(op, dst, src, idx)

Scatter operation, which writes data in `src` into `dst` at locations `idx`.
A binary reduction operator `op` is applied during the scatter. 
For each index `k` in `idx`, accumulates values in `dst` according to

    dst[:, ..., idx[k]...] = (op).(dst[:, ..., idx[k]...], src[:, ..., k...])

See also [`scatter`](@ref), [`gather`](@ref).

# Arguments

- `op`: Operations to be applied on `dst` and `src`, e.g. `+`, `-`, `*`, `/`, `max`, `min` and `mean`.
- `dst`: The destination for `src` to aggregate to. This argument will be mutated.
- `src`: The source data for aggregating.
- `idx`: The mapping for aggregation from source (index) to destination (value). 
         The `idx` array can contain either integers or tuples.

# Examples
```jldoctest
julia> NNlib.scatter!(+, ones(3), [10,100], [1,3])
3-element Vector{Float64}:
  11.0
   1.0
 101.0

julia> NNlib.scatter!(*, fill(0.5, 2, 4), [1 10; 100 1000], [3,2])
2×4 Matrix{Float64}:
 0.5    5.0   0.5  0.5
 0.5  500.0  50.0  0.5
```
"""
function scatter!(op::OP, dst::AbstractArray, src::AbstractArray, idx::AbstractArray) where OP
    dims = scatter_dims(dst, src, idx)
    colons = Base.ntuple(_->Colon(), dims)
    for k in CartesianIndices(idx)
        dst_v = _view(dst, colons, idx[k])
        src_v = _view(src, colons, k)
        dst_v .= (op).(dst_v, src_v)
    end
    return dst
end

function scatter!(op::typeof(mean), dst::AbstractArray, src::AbstractArray, idx::AbstractArray)
    Ns = scatter!(+, zero(dst), one.(src), idx)
    dst_ = scatter!(+, zero(dst), src, idx)
    dst .+= safe_div.(dst_, Ns)
    return dst
end


"""
    NNlib.scatter(op, src, idx; [init, dstsize])

Scatter operation allocating a destination array `dst` and 
calling `scatter!(op, dst, src, idx)` on it.

* If keyword `init` is provided, it is used to initialize the content of `dst`.
  Otherwise, the init values is inferred from the reduction operator `op`
  for some common operators (e.g. `init = 0` for `op = +`). 

* If `dstsize` is provided, it will be used to define the size of
  destination array, otherwise it will be inferred by `src` and `idx`.

See [`scatter!`](@ref) for full details on how `idx` works.

# Examples
```jldoctest
julia> NNlib.scatter(+, [10,100,1000], [3,1,2])
3-element Vector{Int64}:
  100
 1000
   10

julia> NNlib.scatter(+, [1 2 3 4; 5 6 7 8], [2,1,1,5])
2×5 Matrix{Int64}:
  5  1  0  0  4
 13  5  0  0  8

julia> NNlib.scatter(*, [10,200,3000], [1,4,2]; init = 10, dstsize = 6)
6-element Vector{Int64}:
   100
 30000
    10
  2000
    10
    10
```
"""
function scatter(op::OP,
                src::AbstractArray{Tsrc,Nsrc},
                idx::AbstractArray{Tidx,Nidx};
                init = nothing, dstsize = nothing) where {Tsrc,Tidx,Nsrc,Nidx,OP}

    dims = Nsrc - Nidx
    dstsz = isnothing(dstsize) ? (size(src)[1:dims]..., maximum_dims(idx)...) : dstsize 
    dst = similar(src, Tsrc, dstsz)
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
    
    M = typelength(Tidx)
    num = gather(Δ, idx)
    counts = fill!(similar(Δ, Int, size(Δ)[end-M+1:end]), 0)
    scatter!(+, counts, fill!(similar(idx, Int), 1), idx)
    den = gather(counts, idx)
    # make num and den broadcast compatible
    for i in 1:ndims(num)-ndims(den)
        den = unsqueeze(den)
    end
    return safe_div.(num, den)
end

∇scatter_src(op, Δ, dst, src, idx) =
  ∇scatter_src(op, unthunk(Δ), dst, src, idx)

function rrule(::typeof(scatter!), op, dst::AbstractArray, src::AbstractArray, idx::AbstractArray)
    dst_old = copy(dst)
    scatter!(op, dst, src, idx)
    scatter!_pullback(Δ) = (NoTangent(), NoTangent(), ∇scatter!_dst(op, unthunk(Δ), dst_old, dst), ∇scatter!_src(op, unthunk(Δ), dst, src, idx), NoTangent())
    dst, scatter!_pullback
end

function rrule(::typeof(scatter), op, src::AbstractArray, idx::AbstractArray; kws...)
    y = scatter(op, src, idx; kws...)
    scatter_pullback(Δ) = (NoTangent(), NoTangent(), ∇scatter_src(op, unthunk(Δ), y, src, idx), NoTangent())
    y, scatter_pullback
end
