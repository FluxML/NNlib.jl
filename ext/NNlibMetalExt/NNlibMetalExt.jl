module NNlibMetalExt

using Metal, NNlib
using NNlib: AbstractRNG  # === Random.AbstractRNG

# Random
NNlib._rng_from_array(::MtlArray) = Metal.MPS.default_rng()

NNlib._rng_compat_array(rng::Metal.MPS.RNG, A::MtlArray) = nothing
NNlib._rng_compat_array(rng::AbstractRNG, A::MtlArray) = throw(ArgumentError(
    "cannot use rng::$(typeof(rng)) with array::MtlArray, only Metal's own RNG type works"))

# Batched matrix multiplication
function NNlib._batched_gemm!(::Type{<:MtlArray}, transA::Char, transB::Char, α::Number, A, B, β::Number, C)
    eltype(C) <: Complex && @warn "don't trust this on complex arrays!" transA transB
    Metal.MPS.matmul!(C, A, B, α, β, transA != 'N', transB != 'N') # transA, transB, α, A, B, β, C)
end

#=

help?> Metal.MPS.matmul!
  matMulMPS(a::MtlMatrix, b::MtlMatrix, c::MtlMatrix, alpha=1, beta=1,
            transpose_left=false, transpose_right=false)

  A MPSMatrixMultiplication kernel thay computes: c = alpha * op(a) * beta * op(b) + beta * C

  This function should not typically be used. Rather, use the normal LinearAlgebra interface with
  any MtlArray and it should be accelerated using Metal Performance Shaders.

=#

using NNlib: BatchedAdjoint, BatchedTranspose, BatchedAdjOrTrans
using Adapt
using Adapt: WrappedArray

const MetalBatchedAdjoint{T} = BatchedAdjoint{T, <: MtlArray{T}}
const MetalBatchedTranspose{T} = BatchedTranspose{T, <: MtlArray{T}}
const MetalBatchedAdjOrTrans{T} = Union{MetalBatchedAdjoint{T}, MetalBatchedTranspose{T}}
const WrappedMetalBatchedAdjOrTrans{T, N} = WrappedArray{T, N, MetalBatchedAdjOrTrans{T}, MetalBatchedAdjOrTrans{T}}

Base.print_array(io::IO, b::Union{MetalBatchedAdjOrTrans, WrappedMetalBatchedAdjOrTrans}) = Base.print_array(io, adapt(Array, b))
Base._show_nonempty(io::IO, b::Union{MetalBatchedAdjOrTrans, WrappedMetalBatchedAdjOrTrans}, prefix::String) = Base._show_nonempty(io, adapt(Array, b), prefix)
Base.show_vector(io::IO, b::Union{MetalBatchedAdjOrTrans, WrappedMetalBatchedAdjOrTrans}, opn, cls) = Base.show_vector(io, adapt(Array, b), opn, cls)

Base.convert(::Type{T}, b::Union{MetalBatchedAdjOrTrans, WrappedMetalBatchedAdjOrTrans}) where {T<:Array} = Base.convert(T, adapt(Array, b))
Base.Array{T, N}(b::Union{MetalBatchedAdjOrTrans, WrappedMetalBatchedAdjOrTrans}) where {T, N} = Array{T, N}(adapt(Array, b))
Base.collect(b::Union{MetalBatchedAdjOrTrans, WrappedMetalBatchedAdjOrTrans}) = collect(adapt(Array, b))


end  # module NNlibMetalExt
