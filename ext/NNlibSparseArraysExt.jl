module NNlibSparseArraysExt

using SparseArrays: AbstractSparseArray
using NNlib: NNlib

# `gather` on a sparse source should return a dense array, not a sparse one.
# `similar(src, ...)` on a sparse array yields a sparse array, so allocate the
# destination explicitly as a dense `Array`. See issue #625.
function NNlib.gather(
    src::AbstractSparseArray{Tsrc, <:Any, Nsrc}, idx::AbstractArray{Tidx},
) where {Tsrc, Nsrc, Tidx}
    M = NNlib.typelength(Tidx)
    dstsize = (size(src)[1:Nsrc - M]..., size(idx)...)
    dst = Array{Tsrc}(undef, dstsize)
    return NNlib.gather!(dst, src, idx)
end

end # module
