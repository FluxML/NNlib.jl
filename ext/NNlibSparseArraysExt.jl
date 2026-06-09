module NNlibSparseArraysExt

using SparseArrays: AbstractSparseArray
using NNlib: NNlib

# `gather` does not support sparse sources (matching PyTorch and PyTorch
# Geometric, where the gather source is always dense). Without this method a
# sparse `src` would silently return a sparse result via `similar`. Throw a
# clear error instead and let the user densify explicitly. See issue #625.
NNlib.gather(::AbstractSparseArray, ::AbstractArray) = _gather_sparse_error()
NNlib.gather(::AbstractSparseArray, ::AbstractVector{<:Integer},
    ::AbstractVector{<:Integer}...) = _gather_sparse_error()

_gather_sparse_error() = throw(ArgumentError(
    "`gather` does not support sparse sources; convert `src` to a dense array \
    first, e.g. `gather(collect(src), idx)`."))

end # module
