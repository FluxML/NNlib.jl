module NNlibSparseArraysExt

using SparseArrays: AbstractSparseArray, nonzeros
using NNlib: NNlib

# A sparse array stores its nonzero values in a dense buffer that already lives
# on the right device (a `Vector` for `SparseMatrixCSC`, a `CuVector` for a
# `CuSparseMatrix`, ...). `similar` of that buffer therefore gives a dense array
# of the correct type and backend. This makes `NNlib.gather` on a sparse source
# return a dense array instead of a sparse one. See issue #625.
NNlib.dense_like(x::AbstractSparseArray, ::Type{T}, sz) where {T} =
    similar(nonzeros(x), T, sz)

end # module
