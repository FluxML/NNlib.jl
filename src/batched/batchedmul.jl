# batch-wise matrix multiplication
# wrapper for batched_gemm!
export batched_mul, batched_transpose, batched_adjoint

include("./batchedadjtrans.jl")

"""
    batched_mul(A, B) -> C

Batched matrix multiplication. Result has `C[:,:,k] == A[:,:,k] * B[:,:,k]` for all `k`.
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

const _GemmFloat = Union{Float64, Float32, ComplexF64, ComplexF32}

_BATCHED_GEMM_LIST = [
    (:(StridedArray{T, 3}), 'N'),
    (:(BatchedTranspose{T, <:StridedArray{T, 3}}), 'T'),
    (:(BatchedAdjoint{T, <:StridedArray{T, 3}}), 'C')
]

for (TA, transA) in _BATCHED_GEMM_LIST, (TB, transB) in _BATCHED_GEMM_LIST
    @eval function batched_mul!(C::StridedArray{T, 3}, A::$TA, B::$TB) where {T<:_GemmFloat}
        batched_gemm!($transA, $transB, one(T), _unbatch(A), _unbatch(B), zero(T), C)
        C
    end
end

# fallback

_BATCHED_LIST = [
    (:(AbstractArray{<:Any, 3}), :identity),
    (:(BatchedTranspose{<:Any, <:AbstractArray{<:Any, 3}}), :transpose),
    (:(BatchedAdjoint{<:Any, <:AbstractArray{<:Any, 3}}), :adjoint)
]
for (TA, fA) in _BATCHED_LIST, (TB, fB) in _BATCHED_LIST
    @eval function batched_mul!(C::AbstractArray{<:Any, 3}, A::$TA, B::$TB)
        axes(A, 3) == axes(B, 3) == axes(C, 3) || throw(DimensionMismatch("batch size mismatch"))
        @debug "calling fallback method for batched_mul!" typeof(A) typeof(B) typeof(C)
        A′, B′ = _unbatch(A), _unbatch(B)
        @inbounds for k in axes(C, 3)
            @views mul!(C[:,:,k], $fA(A′[:,:,k]), $fB(B′[:,:,k]))
        end
        C
    end
end
