# batch-wise matrix multiplication,
# including a wrapper for batched_gemm!

export batched_mul, batched_transpose, batched_adjoint

using LinearAlgebra: BlasFloat

include("./batchedadjtrans.jl")

"""
    batched_mul(A, B) -> C

Batched matrix multiplication. Result has `C[:,:,k] == A[:,:,k] * B[:,:,k]` for all `k`.

Using `batched_transpose(A)` or `PermutedDimsArray(A, (2,1,3))` will transpose each `A[:,:,k]`,
and similarly `batched_adjoint(B)` will use `adjoint(B[:,:,k])`.
"""
function batched_mul(A::AbstractArray{T1, 3}, B::AbstractArray{T2, 3}) where {T1, T2}
    axes(A, 3) == axes(B, 3) || throw(DimensionMismatch("batch size mismatch"))
    T = promote_type(T1, T2)
    C = similar(A, T, (axes(A, 1), axes(B, 2), axes(A, 3)))
    batched_mul!(C, A, B)
end

"""
    batched_mul!(C, A, B) -> C

In-place batched matrix multiplication,
equivalent to `mul!(C[:,:,k], A[:,:,k], B[:,:,k])` for all `k`.
"""
function batched_mul! end

_unbatch(A) = A
_unbatch(A::BatchedAdjOrTrans) = A.parent

# batched_gemm!

_BATCHED_GEMM_LIST = [
    (:(StridedArray{T, 3}), 'N'),
    (:(BatchedTranspose{T, <:StridedArray{T, 3}}), 'T'),
    (:(BatchedAdjoint{T, <:StridedArray{T, 3}}), 'C')
]

for (TA, transA) in _BATCHED_GEMM_LIST, (TB, transB) in _BATCHED_GEMM_LIST
    @eval function batched_mul!(C::Array{T, 3}, A::$TA, B::$TB) where {T<:BlasFloat}
        batched_gemm!($transA, $transB, one(T), _unbatch(A), _unbatch(B), zero(T), C)
        C
    end
end

# fallback

_BATCHED_LIST = [
    (:(AbstractArray{<:Any, 3}), :identity),
    (:BatchedTranspose, :transpose),
    (:BatchedAdjoint, :adjoint),
]
for (TA, fA) in _BATCHED_LIST, (TB, fB) in _BATCHED_LIST
    @eval function batched_mul!(C::AbstractArray{<:Any, 3}, A::$TA, B::$TB)
        axes(A, 3) == axes(B, 3) == axes(C, 3) || throw(DimensionMismatch("batch size mismatch"))
        if A isa PermutedDimsArray{<:BlasFloat,3,(2,1,3)}
            return batched_mul!(C, batched_transpose(parent(A)), B)
        elseif B isa PermutedDimsArray{<:BlasFloat,3,(2,1,3)}
            return batched_mul!(C, A, batched_transpose(parent(B)))
        end
        @debug "calling fallback method for batched_mul!" typeof(A) typeof(B) typeof(C)
        A′, B′ = _unbatch(A), _unbatch(B)
        @inbounds for k in axes(C, 3)
            @views mul!(C[:,:,k], $fA(A′[:,:,k]), $fB(B′[:,:,k]))
        end
        C
    end
end
