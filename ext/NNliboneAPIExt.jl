module NNliboneAPIExt

using NNlib
using oneAPI

function NNlib._batched_gemm!(::Type{<:oneArray}, transA::Char, transB::Char, α::Number, A, B, β::Number, C)
    oneAPI.oneMKL.gemm_strided_batched!(transA, transB, α, A, B, β, C)
end

using NNlib: BatchedAdjoint, BatchedTranspose, BatchedAdjOrTrans
using Adapt
using Adapt: WrappedArray

const oneAPIBatchedAdjoint{T} = BatchedAdjoint{T, <: oneArray{T}}
const oneAPIBatchedTranspose{T} = BatchedTranspose{T, <: oneArray{T}}
const oneAPIBatchedAdjOrTrans{T} = Union{oneAPIBatchedAdjoint{T}, oneAPIBatchedTranspose{T}}
const WrappedoneAPIBatchedAdjOrTrans{T, N} = WrappedArray{T, N, oneAPIBatchedAdjOrTrans{T}, oneAPIBatchedAdjOrTrans{T}}

Base.print_array(io::IO, b::Union{oneAPIBatchedAdjOrTrans, WrappedoneAPIBatchedAdjOrTrans}) = Base.print_array(io, adapt(Array, b))
Base._show_nonempty(io::IO, b::Union{oneAPIBatchedAdjOrTrans, WrappedoneAPIBatchedAdjOrTrans}, prefix::String) = Base._show_nonempty(io, adapt(Array, b), prefix)
Base.show_vector(io::IO, b::Union{oneAPIBatchedAdjOrTrans, WrappedoneAPIBatchedAdjOrTrans}, opn, cls) = Base.show_vector(io, adapt(Array, b), opn, cls)

Base.convert(::Type{T}, b::Union{oneAPIBatchedAdjOrTrans, WrappedoneAPIBatchedAdjOrTrans}) where {T<:Array} = Base.convert(T, adapt(Array, b))
Base.Array{T, N}(b::Union{oneAPIBatchedAdjOrTrans, WrappedoneAPIBatchedAdjOrTrans}) where {T, N} = Array{T, N}(adapt(Array, b))
Base.collect(b::Union{oneAPIBatchedAdjOrTrans, WrappedoneAPIBatchedAdjOrTrans}) = collect(adapt(Array, b))


end  # module NNliboneAPIExt