export gather, gather!

"""
    gather!(dst, src, idx)

Reverse operation of scatter. Gathers data from source `src` 
and writes it in destination `dst` according to the index
array `idx`.
For each `k` in `CartesianIndices(idx)`, assign values to `dst` according to

    dst[:, ... , k] .= src[:, ... , idx[k]...]

Notice that if `idx` is a vector containing integers,
and both `dst` and `src` are matrices, previous 
expression simplifies to

    dst[:, k] .= src[:, idx[k]]

and `k` will range over `1:length(idx)`. 

The elements of `idx` may be repeated. A single `src` column
can end up being copied into zero, one, or multiple `dst` columns.

# Arguments

- `dst`: the destination where data would be assigned to.
- `src`: the source of the data to be assigned.
- `idx`: the mapping from source to destination.
"""
function gather!(dst::AbstractArray{Tdst,Ndst}, 
                 src::AbstractArray{Tsrc,Nsrc}, 
                 idx::AbstractArray{Tidx, Nidx}) where 
                    {Tdst, Tsrc, Ndst, Nsrc, Nidx, Tidx <: IntOrIntTuple}

    M = typelength(Tidx)
    Ndst - Nidx == Nsrc - M || throw(ArgumentError("Incompatible input shapes."))
    size(dst)[1:Ndst-Nidx] == size(src)[1:Ndst-Nidx] || throw(ArgumentError("Incompatible input shapes."))
    size(dst)[Ndst-Nidx+1:end] == size(idx) || throw(ArgumentError("Incompatible input shapes."))

    coldst = ntuple(i -> Colon(), Ndst - Nidx)
    colsrc = ntuple(i -> Colon(), Nsrc - M)
    for k in CartesianIndices(idx)
        view(dst, coldst..., k) .= view(src, colsrc..., idx[k]...)
    end
    return dst
end

"""
    gather(src, idx)

Non-mutating version of [`gather!`](@ref).
"""
function gather(src::AbstractArray{Tsrc, Nsrc}, 
                idx::AbstractArray{Tidx, Nidx}) where 
                    {Tsrc, Nsrc, Nidx, Tidx<:IntOrIntTuple}

    M = typelength(Tidx) 
    dstsize = (size(src)[1:Nsrc-M]..., size(idx)...)
    dst = similar(src, Tsrc, dstsize)
    return gather!(dst, src, idx)
end

typelength(::Type{<:Number}) = 1
typelength(::Type{<:NTuple{M}}) where M = M
