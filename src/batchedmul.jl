# batch-wise matrix multiplication
# wrapper for batched_gemm!

include("./batchedadjtrans.jl")

function batched_mul(A::AbstractArray{T, 3}, B::AbstractArray{T, 3}) where T
    size(A, 3) == size(B, 3) || throw(DimensionMismatch("batch size mismatch"))
    batched_mul!(similar(A, (size(A, 1), size(B, 2), size(A, 3))), A, B)
end

"""
    batched_mul!(C, A, B) -> C
batched `mul!`.
"""
function batched_mul! end

_unbatch(A) = A
_unbatch(A::BatchedAdjOrTrans) = A.parent

# bmm
const _BATCHED_MATRIX_LIST = [
        (:(AbstractArray{T, 3}), 'N'),
        (:(BatchedTranspose{T, <:AbstractArray{T, 3}}), 'T'),
        (:(BatchedAdjoint{T, <:AbstractArray{T, 3}}), 'C')
]

for (TA, transA) in _BATCHED_MATRIX_LIST, (TB, transB) in _BATCHED_MATRIX_LIST
    @eval begin
        function batched_mul!(C::AbstractArray{T, 3}, A::$TA, B::$TB) where T
            batched_gemm!($transA, $transB, one(T), _unbatch(A), _unbatch(B), zero(T), C)
            C
        end


    end
end

function ∇batched_mul(Δ::AbstractArray{T, 3}, A::AbstractArray{T, 3}, B::AbstractArray{T, 3}) where T
    (batched_mul(Δ, batched_transpose(B)), batched_mul(batched_transpose(A), Δ))
end


function ∇batched_mul(Δ::AbstractArray{T, 3}, A::BatchedTranspose{T, <: AbstractArray{T, 3}}, B::AbstractArray{T, 3}) where T
    (batched_mul(Δ, batched_transpose(B)), batched_mul(A, Δ))
end

function ∇batched_mul(Δ::AbstractArray{T, 3}, A::AbstractArray{T, 3}, B::BatchedTranspose{T, <: AbstractArray{T, 3}}) where T
    (batched_mul(Δ, B), batched_mul(batched_transpose(A), Δ))
end

function ∇batched_mul(Δ::AbstractArray{T, 3}, A::BatchedTranspose{T, <: AbstractArray{T, 3}}, B::BatchedTranspose{T, <: AbstractArray{T, 3}}) where T
    (batched_mul(batched_transpose(Δ), batched_transpose(B)), batched_mul(batched_transpose(A), batched_transpose(Δ)))
end
