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

_check_input(idx::AbstractArray{<:Integer}, arr) = checkbounds(arr, minimum(idx):maximum(idx))

function _check_input(idx::AbstractArray{<:Tuple}, arr)
    pairs = map(xs -> Base.OneTo(maximum(xs)), zip(idx...))
    checkbounds(arr, pairs...)
end

function _check_output(idx::AbstractArray{<:IntOrTuple}, dst, src, dims)
    idx_dims = size(idx)
    dst_dims = size(dst)
    src_dims = size(src)
    dst_dims[begin:dims] == src_dims[begin:dims] || throw(ArgumentError("dst and src must have the same dimensions in the first $(dims) dimensions"))
    dst_dims[dims+1:end] == idx_dims || throw(ArgumentError("dst must have the same dimensions with idx from $(dims+1)-th"))
end
