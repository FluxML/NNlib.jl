"""
    safe_div(x, y)

Safely divide `x` by `y`. If `y` is zero, return `x` directly.
"""
safe_div(x, y) = ifelse(iszero(y), x, x/y)

"""
    least_dims(idxs)

Compute the least dimensions, of which array can be accessed by the indices `idxs`.
"""
least_dims(idxs::AbstractArray{<:Integer}) = (maximum(idxs), )

function least_dims(idxs::AbstractArray{<:Tuple})
    Tuple(maximum(xs) for xs in zip(idxs...))
end

function reverse_indices(X::Array{T}) where T
    Y = Dict{T,Vector{CartesianIndex}}()
    @inbounds for (ind, val) = pairs(X)
        Y[val] = get(Y, val, CartesianIndex[])
        push!(Y[val], ind)
    end
    Y
end

function count_indices(idx::AbstractArray, N)
    counts = zero.(idx)
    @inbounds for i = 1:N
        counts += sum(idx.==i) * (idx.==i)
    end
    counts
end

function divide_by_counts!(xs, idx::AbstractArray, N)
    counts = count_indices(idx, N)
    @inbounds for ind = CartesianIndices(counts)
        view(xs, :, ind) ./= counts[ind]
    end
end
