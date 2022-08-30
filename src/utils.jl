"""
    within_gradient(x) --> Bool

Returns `false` except when used inside a `gradient` call, when it returns `true`.
Useful for Flux regularisation layers which behave differently during training and inference.

Works with any ChainRules-based differentiation package, in which case `x` is ignored.
But Tracker.jl overloads `with_gradient(x::TrackedArray)`, thus for widest use you should
pass it an array whose gradient is of interest.
"""
within_gradient(x) = false

ChainRulesCore.rrule(::typeof(within_gradient), x) = true, _ -> (NoTangent(), NoTangent())


"""
    safe_div(x, y)

Safely divide `x` by `y`. If `y` is zero, return `x` directly.
"""
safe_div(x, y) = ifelse(iszero(y), x, x/y)

"""
    maximum_dims(dims)

Return the maximum value for each dimension. An array of dimensions `dims` is accepted.
The maximum of each dimension in the element is computed.
"""
maximum_dims(dims::AbstractArray{<:Integer}) = (maximum(dims), )
maximum_dims(dims::AbstractArray{NTuple{N, T}}) where {N,T} = ntuple(i -> maximum(x->x[i], dims), N)
maximum_dims(dims::AbstractArray{CartesianIndex{N}}) where {N} = ntuple(i -> maximum(x->x[i], dims), N)

function reverse_indices!(rev::AbstractArray, idx::AbstractArray{<:Tuple})
    for (ind, val) in pairs(Array(idx))
        push!(rev[val...], ind)
    end
    # if CUDA supports `unique`, a more efficient version:
    # cidx in CartesianIndices(idx)
    # for i = unique(idx)
    #     rev[i] = cidx[idx .== i]
    # end
    rev
end

function reverse_indices!(rev::AbstractArray, idx::AbstractArray)
    for (ind, val) in pairs(Array(idx))
        push!(rev[val], ind)
    end
    rev
end

"""
    reverse_indices(idx)

Return the reverse indices of `idx`. The indices of `idx` will be values, and values of `idx` will be index.

# Arguments

- `idx`: The indices to be reversed. Accepts array or cuarray of integer, tuple or `CartesianIndex`.
"""
function reverse_indices(idx::AbstractArray{<:Any,N}) where N
    max_dims = maximum_dims(idx)
    T = CartesianIndex{N}
    rev = Array{Vector{T}}(undef, max_dims...)
    for i in eachindex(rev)
        rev[i] = T[]
    end
    return reverse_indices!(rev, idx)
end

unsqueeze(x) = reshape(x, 1, size(x)...) 
