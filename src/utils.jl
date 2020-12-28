"""
    safe_div(x, y)

Safely divide `x` by `y`. If `y` is zero, return `x` directly.
"""
safe_div(x, y) = ifelse(iszero(y), x, x/y)

"""
    least_dims(idxs)

Compute the least dimensions, of which array can be accessed by the indecies `idxs`.
"""
least_dims(idxs::AbstractArray{<:Integer}) = (maximum(idxs), )

function least_dims(idxs::AbstractArray{<:Tuple})
    Tuple(maximum(xs) for xs in zip(idxs...))
end
