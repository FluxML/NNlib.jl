export scatter!

const IntOrTuple = Union{Integer,Tuple}

"""
    scatter!(op, ys, us, xs)

Scatter operation. For each index `k` in `xs`, accumulate values in `ys` according to

    ys[xs[k]...] = (op).(ys[xs[k]...], us[k...])

# Arguments
- `op`: operations to be applied on `ys` and `us`, e.g. `+`, `-`, `*`, `/`, `max` and `min`.
- `ys`: the destination for `us` to aggregate to. This argument will be mutated.
- `us`: the source data for aggregating.
- `xs`: the mapping for aggregation from source (index) to destination (value).
The index of `xs` is corresponding to the index of `us` and the value of `xs` is
corresponding to the index of `ys`. The value of `xs` can be `Int` or `Tuple` type.

The dimension of `us` must equal to dimension of `xs`. `ys`, `us` and `xs` must be
supported array type and be the same type.`Array`, `StaticArray` and `CuArray`
are currently supported.
"""
function scatter!(op, ys::AbstractArray{T}, us::AbstractArray{T}, xs::AbstractArray{<:IntOrTuple}) where {T<:Real}
    @simd for k in CartesianIndices(xs)
        k = CartesianIndices(xs)[k]
        ys_v = view(ys, xs[k]...)
        us_v = view(us, k)
        @inbounds ys_v .= (op).(ys_v, us_v)
    end
    ys
end

"""
    scatter!(mean, ys, us, xs)

Scatter mean operation. For each index `k` in `xs`, accumulate values in `ys` according to

    ys[xs[k]...] = ys[xs[k]...] + mean.(us[k...])

# Arguments
- `ys`: the destination for `us` to aggregate to. This argument will be mutated.
- `us`: the source data for aggregating.
- `xs`: the mapping for aggregation from source (index) to destination (value).
The index of `xs` is corresponding to the index of `us` and the value of `xs` is
corresponding to the index of `ys`. The value of `xs` can be `Int` or `Tuple` type.

The dimension of `us` must equal to dimension of `xs`. `ys`, `us` and `xs` must be
supported array type and be the same type.`Array`, `StaticArray` and `CuArray`
are currently supported.
"""
function scatter!(op::typeof(mean), ys::AbstractArray{T}, us::AbstractArray{T}, xs::AbstractArray{<:IntOrTuple}) where {T<:Real}
    Ns = zero(ys)
    ys_ = zero(ys)
    scatter!(+, Ns, one.(us), xs)
    scatter!(+, ys_, us, xs)
    ys .+= safe_div.(ys_, Ns)
    return ys
end

function scatter!(op, ys::AbstractArray{T}, us::AbstractArray{S}, xs::AbstractArray{<:IntOrTuple}) where {T<:Real,S<:Real}
    PT = promote_type(T, S)
    scatter!(op, PT.(ys), PT.(us), xs)
end
