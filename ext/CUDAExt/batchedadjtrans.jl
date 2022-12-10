using NNlib: BatchedAdjoint, BatchedTranspose, BatchedAdjOrTrans
using Adapt
using Adapt: WrappedArray

const CuBatchedAdjoint{T} = BatchedAdjoint{T, <: CuArray{T}}
const CuBatchedTranspose{T} = BatchedTranspose{T, <: CuArray{T}}
const CuBatchedAdjOrTrans{T} = Union{CuBatchedAdjoint{T}, CuBatchedTranspose{T}}
const WrappedCuBatchedAdjOrTrans{T, N} = WrappedArray{T, N, CuBatchedAdjOrTrans{T}, CuBatchedAdjOrTrans{T}}


Base.print_array(io::IO, b::Union{CuBatchedAdjOrTrans, WrappedCuBatchedAdjOrTrans}) = Base.print_array(io, adapt(Array, b))
Base._show_nonempty(io::IO, b::Union{CuBatchedAdjOrTrans, WrappedCuBatchedAdjOrTrans}, prefix::String) = Base._show_nonempty(io, adapt(Array, b), prefix)
Base.show_vector(io::IO, b::Union{CuBatchedAdjOrTrans, WrappedCuBatchedAdjOrTrans}, opn, cls) = Base.show_vector(io, adapt(Array, b), opn, cls)

Base.convert(::Type{T}, b::Union{CuBatchedAdjOrTrans, WrappedCuBatchedAdjOrTrans}) where {T<:Array} = Base.convert(T, adapt(Array, b))
Base.Array{T, N}(b::Union{CuBatchedAdjOrTrans, WrappedCuBatchedAdjOrTrans}) where {T, N} = Array{T, N}(adapt(Array, b))
Base.collect(b::Union{CuBatchedAdjOrTrans, WrappedCuBatchedAdjOrTrans}) = collect(adapt(Array, b))
