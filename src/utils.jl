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

function reverse_indices!(rev::AbstractArray, idx::AbstractArray)
    for (ind, val) in pairs(Array(idx))
        push!(rev[val], ind)
    end
    # if CUDA supports `unique`, a more efficient version:
    # cidx in CartesianIndices(idx)
    # for i = unique(idx)
    #     rev[i] = cidx[idx .== i]
    # end
    rev
end

function reverse_indices(idx::AbstractArray)
    rev = Array{Vector{CartesianIndex}}(undef, maximum_dims(idx)...)
    for i in eachindex(rev)
        rev[i] = CartesianIndex[]
    end
    return reverse_indices!(rev, idx)
end

function count_indices(idx::AbstractArray)
    counts = zero.(idx)
    for i in unique(idx)
        counts += sum(idx .== i) * (idx .== i)
    end
    return counts
end

function divide_by_counts!(xs, idx::AbstractArray, dims)
    colons = Base.ntuple(_->Colon(), dims)
    counts = count_indices(idx)
    for i in CartesianIndices(counts)
        view(xs, colons..., i) ./= counts[i]
    end
    return xs
end
