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

function maximum_dims(dims::AbstractArray{<:Tuple})
    Tuple(maximum(xs) for xs in zip(dims...))
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
