export scatter!, scatter, ∇scatter_dst!, ∇scatter_src!, ∇scatter_src

## Scatter API
#   - Scatter:
#     - scatter(op, src, idx, dims)
#     - scatter!(op, dst, src, idx, dims)
#   - Scatter destination backpropagation
#     - ∇scatter_dst!
#   - Scatter source backpropagation
#     - ∇scatter_src
#     - ∇scatter_src!
#

"""
    scatter!(op, dst, src, idx, dims)

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
- `dims`: the number of dimensions to be view as an operational unit from the beginning.
If dims=0 is given, operational unit is scalar. Default dims is 1.

Examples for dims are lists here:

- `dims=0`: dst[idx[k]...] = op(dst[idx[k]...], src[k...])
- `dims=1`: dst[:, idx[k]...] = (op).(dst[:, idx[k]...], src[:, k...])
- `dims=2`: dst[:, :, idx[k]...] = (op).(dst[:, :, idx[k]...], src[:, :, k...])
"""
function scatter!(op, dst::AbstractArray{T}, src::AbstractArray{T}, idx::AbstractArray{<:IntOrTuple};
                  dims::Integer=1) where {T<:Real}
    # @boundscheck _check_output(idx, dst)
    @boundscheck _check_input(idx, src)
    if dims > 0
        scatter_vec!(op, dst, src, idx, Val(dims))
    elseif dims == 0
        scatter_scl!(op, dst, src, idx)
    else
        throw(ArgumentError("dims must be non-negative but got dims=$dims"))
    end
end

@inbounds function scatter_scl!(op, dst::AbstractArray{T}, src::AbstractArray{T}, idx::AbstractArray{<:IntOrTuple}) where {T<:Real}
    for k in CartesianIndices(idx)
        dst[idx[k]...] = op(dst[idx[k]...], src[k])
    end
    dst
end

function scatter_vec!(op, dst::AbstractArray{T}, src::AbstractArray{T}, idx::AbstractArray{<:IntOrTuple},
                      dims::Integer) where {T<:Real}
    colons = Base.ntuple(_->Colon(), dims)
    @simd for k in CartesianIndices(idx)
        dst_v = view(dst, colons..., idx[k]...)
        src_v = view(src, k)
        @inbounds dst_v .= (op).(dst_v, src_v)
    end
    dst
end

"""
    scatter!(mean, dst, src, idx, dims)

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
- `dims`: the number of dimensions to be view as an operational unit from the beginning.
If dims=0 is given, operational unit is scalar. Default dims is 1.

Examples for dims are lists here:

- `dims=0`: dst[idx[k]...] = op(dst[idx[k]...], src[k...])
- `dims=1`: dst[:, idx[k]...] = (op).(dst[:, idx[k]...], src[:, k...])
- `dims=2`: dst[:, :, idx[k]...] = (op).(dst[:, :, idx[k]...], src[:, :, k...])
"""
function scatter!(op::typeof(mean), dst::AbstractArray{T}, src::AbstractArray{T}, idx::AbstractArray{<:IntOrTuple};
                  dims::Integer=1) where {T<:Real}
    Ns = zero(dst)
    dst_ = zero(dst)
    scatter!(+, Ns, one.(src), idx; dims=dims)
    scatter!(+, dst_, src, idx; dims=dims)
    dst .+= safe_div.(dst_, Ns)
    return dst
end

function scatter!(op, dst::AbstractArray{T}, src::AbstractArray{S}, idx::AbstractArray{<:IntOrTuple};
                  dims::Integer=1) where {T<:Real,S<:Real}
    PT = promote_type(T, S)
    scatter!(op, PT.(dst), PT.(src), idx; dims=dims)
end


"""
    scatter(op, src, idx, dims)

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
- `dims`: the number of dimensions to be view as an operational unit from the beginning.
If dims=0 is given, operational unit is scalar. Default dims is 1.
"""
function scatter end

for op in [+, -]
    @eval function scatter(op::typeof($op), src::AbstractArray{T}, idx::AbstractArray{<:IntOrTuple};
                           dims::Integer=1) where {T<:Real}
        dst = zeros(T, size(src)[1:dims]..., least_dims(idx)...)
        scatter!(op, dst, src, idx, dims=dims)
    end
end

for op in [*, /]
    @eval function scatter(op::typeof($op), src::AbstractArray{T}, idx::AbstractArray{<:IntOrTuple};
                           dims::Integer=1) where {T<:Real}
        dst = ones(T, size(src)[1:dims]..., least_dims(idx)...)
        scatter!(op, dst, src, idx, dims=dims)
    end
end

function scatter(op::typeof(max), src::AbstractArray{T}, idx::AbstractArray{<:IntOrTuple};
                 dims::Integer=1) where {T<:Real}
    dst = fill(typemin(T), size(src)[1:dims]..., least_dims(idx)...)
    scatter!(max, dst, src, idx, dims=dims)
end

function scatter(op::typeof(min), src::AbstractArray{T}, idx::AbstractArray{<:IntOrTuple};
                 dims::Integer=1) where {T<:Real}
    dst = fill(typemax(T), size(src)[1:dims]..., least_dims(idx)...)
    scatter!(min, dst, src, idx, dims=dims)
end

function scatter(op::typeof(mean), src::AbstractArray{T}, idx::AbstractArray{<:IntOrTuple};
                 dims::Integer=1) where {T<:Real}
    FT = float(T)
    dst = zeros(FT, size(src)[1:dims]..., least_dims(idx)...)
    scatter!(mean, dst, FT.(src), idx, dims=dims)
end


## Gradients

opname(::typeof(+)) = :add
opname(::typeof(-)) = :sub
opname(::typeof(*)) = :mul
opname(::typeof(/)) = :div


∇scatter_dst!(op, Δ) = Δ
∇scatter_dst!(::Union{typeof(max),typeof(min)}, Δ, X) = X .* Δ

modify_src(::typeof(+), X) = X
modify_src(::typeof(-), X) = -X
modify_src(::typeof(*), X, Y) = X
modify_src(::typeof(/), X, Y) = -X ./ Y.^2

∇src_init!(Δ, idx) = gather(Δ, idx)
∇src_init!(Δ, dst, idx) = gather(dst, idx) .* ∇src_init!(Δ, idx)
∇src_init(Δ, idx) = gather(zero(Δ)+Δ, idx)

∇scatter_src!(op::Union{typeof(+),typeof(-)}, Δ, idx) = modify_src(op, ∇src_init!(Δ, idx))
∇scatter_src(op::Union{typeof(+),typeof(-)}, Δ, idx) = modify_src(op, ∇src_init!(Δ, idx))

function ∇scatter_src!(op::Union{typeof(*),typeof(/)}, Δ, dst, src, idx)
    Δsrc = modify_src(op, ∇src_init!(Δ, dst, idx), src)
    rev_idx = reverse_indices(idx)
    @inbounds for ind = CartesianIndices(idx)
        inds = filter(x -> x != ind, rev_idx[idx[ind]])
        for i = 1:size(src, 1)
            Δsrc[i, ind] *= prod(j -> src[i, j], inds)
        end
    end
    Δsrc
end

function ∇scatter_src(op::Union{typeof(*),typeof(/)}, Δ, src, idx)
    Δsrc = modify_src(op, ∇src_init(Δ, idx), src)
    rev_idx = reverse_indices(idx)
    @inbounds for ind = CartesianIndices(idx)
        inds = filter(x -> x != ind, rev_idx[idx[ind]])
        for i = 1:size(src, 1)
            Δsrc[i, ind] = op(Δsrc[i, ind], prod(j -> src[i, j], inds))
        end
    end
    Δsrc
end

∇scatter_src!(::Union{typeof(max),typeof(min)}, Δ, X) = X .* ∇src_init!(Δ, idx)
∇scatter_src(::Union{typeof(max),typeof(min)}, Δ, X) = X .* ∇src_init!(Δ, idx)

function ∇scatter_src!(::typeof(mean), Δ, dst, idx)
    divide_by_counts!(∇src_init!(Δ, idx), idx, size(dst, 2))
end

function ∇scatter_src(::typeof(mean), Δ, dst, idx)
    divide_by_counts!(∇src_init!(Δ, idx), idx, size(dst, 2))
end



for op in [+, -]
    pullback = Symbol(:scatter!_, opname(op), :_pullback)
    @eval function ChainRulesCore.rrule(::typeof(scatter!), op::typeof($op), dst::AbstractArray, src::AbstractArray, idx::AbstractArray)
        $pullback(Δ) = (NO_FIELDS, NO_FIELDS, ∇scatter_dst!(op, Δ), ∇scatter_src!(op, Δ, idx), DoesNotExist())
        scatter!(op, copy(dst), src, idx), $pullback
    end

    pullback = Symbol(:scatter_, opname(op), :_pullback)
    @eval function ChainRulesCore.rrule(::typeof(scatter), op::typeof($op), src::AbstractArray, idx::AbstractArray)
        $pullback(Δ) = (NO_FIELDS, NO_FIELDS, ∇scatter_src(op, Δ, idx), DoesNotExist())
        scatter(op, src, idx), $pullback
    end
end

for op in [*, /]
    pullback = Symbol(:scatter!_, opname(op), :_pullback)
    @eval function ChainRulesCore.rrule(::typeof(scatter!), op::typeof($op), dst::AbstractArray, src::AbstractArray, idx::AbstractArray)
        $pullback(Δ) = (NO_FIELDS, NO_FIELDS, ∇scatter_dst!(op, Δ), ∇scatter_src!(op, Δ, dst, src, idx), DoesNotExist())
        scatter!(op, copy(dst), src, idx), $pullback
    end

    pullback = Symbol(:scatter_, opname(op), :_pullback)
    @eval function ChainRulesCore.rrule(::typeof(scatter), op::typeof($op), src::AbstractArray, idx::AbstractArray)
        $pullback(Δ) = (NO_FIELDS, NO_FIELDS, ∇scatter_src(op, Δ, src, idx), DoesNotExist())
        scatter(op, src, idx), $pullback
    end
end

for op in [max, min]
    pullback = Symbol(:scatter!_, Symbol(op), :_pullback)
    @eval function ChainRulesCore.rrule(::typeof(scatter!), op::typeof($op), dst::AbstractArray, src::AbstractArray, idx::AbstractArray)
        m = scatter!(op, copy(dst), src, idx)
        Y = (dst .== m)
        X = (src .== gather(dst, idx))
        $pullback(Δ) = (NO_FIELDS, NO_FIELDS, ∇scatter_dst!(op, Δ, Y), ∇scatter_src!(op, Δ, X), DoesNotExist())
        m, $pullback
    end

    pullback = Symbol(:scatter_, Symbol(op), :_pullback)
    @eval function ChainRulesCore.rrule(::typeof(scatter), op::typeof($op), src::AbstractArray, idx::AbstractArray)
        X = (src .== gather(dst, idx))
        $pullback(Δ) = (NO_FIELDS, NO_FIELDS, ∇scatter_src(op, Δ, X), DoesNotExist())
        scatter(op, src, idx), $pullback
    end
end

function ChainRulesCore.rrule(::typeof(scatter!), op::typeof(mean), dst::AbstractArray, src::AbstractArray, idx::AbstractArray)
    scatter!_mean_pullback(Δ) = (NO_FIELDS, NO_FIELDS, ∇scatter_dst!(op, Δ), ∇scatter_src!(op, Δ, dst, idx), DoesNotExist())
    scatter!(op, copy(dst), src, idx), scatter!_mean_pullback
end

function ChainRulesCore.rrule(::typeof(scatter), op::typeof(mean), src::AbstractArray, idx::AbstractArray)
    scatter_mean_pullback(Δ) = (NO_FIELDS, NO_FIELDS, ∇scatter_src(op, Δ, dst, idx), DoesNotExist())
    scatter(op, src, idx), scatter_mean_pullback
end
