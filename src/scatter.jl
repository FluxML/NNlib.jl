export scatter!

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
- `dims=1`: dst[:, idx[k]...] = (op).(dst[:, idx[k]...], src[k...])
- `dims=2`: dst[:, :, idx[k]...] = (op).(dst[:, :, idx[k]...], src[k...])
"""
function scatter!(op, dst::AbstractArray{T}, src::AbstractArray{T}, idx::AbstractArray{<:IntOrTuple},
                  dims::Integer=1) where {T<:Real}
    if dims > 0
        scatter_vec!(op, dst, src, idx, dims)
    elseif dims == 0
        scatter_scl!(op, dst, src, idx)
    else
        throw(ArgumentError("dims must be non-negative but got dims=$dims"))
    end
end

function scatter_scl!(op, dst::AbstractArray{T}, src::AbstractArray{T}, idx::AbstractArray{<:IntOrTuple}) where {T<:Real}
    @inbounds for k in CartesianIndices(idx)
        dst[idx[k]...] = op(dst[idx[k]...], src[k])
    end
    dst
end

function scatter_vec!(op, dst::AbstractArray{T}, src::AbstractArray{T}, idx::AbstractArray{<:IntOrTuple},
                      dims::Integer) where {T<:Real}
    @simd for k in CartesianIndices(idx)
        dst_v = view(dst, colons(dims)..., idx[k]...)
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
- `dims=1`: dst[:, idx[k]...] = (op).(dst[:, idx[k]...], src[k...])
- `dims=2`: dst[:, :, idx[k]...] = (op).(dst[:, :, idx[k]...], src[k...])
"""
function scatter!(op::typeof(mean), dst::AbstractArray{T}, src::AbstractArray{T}, idx::AbstractArray{<:IntOrTuple},
                  dims::Integer=1) where {T<:Real}
    Ns = zero(dst)
    dst_ = zero(dst)
    scatter!(+, Ns, one.(src), idx, dims)
    scatter!(+, dst_, src, idx, dims)
    dst .+= safe_div.(dst_, Ns)
    return dst
end

function scatter!(op, dst::AbstractArray{T}, src::AbstractArray{S}, idx::AbstractArray{<:IntOrTuple},
                  dims::Integer=1) where {T<:Real,S<:Real}
    PT = promote_type(T, S)
    scatter!(op, PT.(dst), PT.(src), idx, dims)
end
