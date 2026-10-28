# Batched matrix multiplication
# 1st argument is produced by NNlib.storage_type(A)
NNlib._batched_gemm!(::Type{<:CuArray}, transA::Char, transB::Char, α::Number, A, B, β::Number, C) =
     CUBLAS.gemm_strided_batched!(transA, transB, α, A, B, β, C)

const CUDABatchedHalf = Union{Float16,BFloat16}

function NNlib._batched_mul(::Type{<:CuArray{T}}, A, B) where {T<:CUDABatchedHalf}
    C = similar(A, T, (size(A, 1), size(B, 2), max(size(A, 3), size(B, 3))))
    return NNlib.batched_mul!(C, NNlib._copy_if_faster(A), NNlib._copy_if_faster(B))
end

function NNlib._batched_mul!(::Type{DT}, C, A, B, α::Number, β::Number) where {T<:CUDABatchedHalf,DT<:CuArray{T}}
    C isa StridedCuArray && stride(C, 1) == 1 || return NNlib.batched_mul_generic!(C, A, B, α, β)
    return NNlib._batched_try_gemm!(DT, C, A, B, α, β, Float32)
end

function NNlib._batched_gemm!(::Type{<:CuArray{T}}, transA::Char, transB::Char,
                              α::Number, A, B, β::Number, C) where {T<:CUDABatchedHalf}
    @static if pkgversion(CUDA) >= v"6.5"
        return CUBLAS.gemm_strided_batched!(transA, transB, α, A, B, β, C; compute_type=Float32)
    else
        # The old wrapper selects Float16 accumulation for Float16 output.
        return batched_gemm_half!(transA, transB, α, A, B, β, C)
    end
end

@static if pkgversion(CUDA) < v"6.5"
    function batched_gemm_half!(transA, transB, α, A, B, β, C)
        m, k = transA == 'N' ? size(A)[1:2] : reverse(size(A)[1:2])
        kb, n = transB == 'N' ? size(B)[1:2] : reverse(size(B)[1:2])
        (m, n) == size(C)[1:2] && k == kb || throw(DimensionMismatch("batched_mul matrix sizes do not match"))
        size(A, 3) in (1, size(C, 3)) && size(B, 3) in (1, size(C, 3)) ||
            throw(DimensionMismatch("batched_mul batch sizes do not match"))
        strideA = size(A, 3) == 1 ? 0 : stride(A, 3)
        strideB = size(B, 3) == 1 ? 0 : stride(B, 3)
        compute = CUDA.math_mode() == CUDA.PEDANTIC_MATH ?
                  CUBLAS.CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS.CUBLAS_COMPUTE_32F
        CUBLAS.cublasGemmStridedBatchedEx(CUBLAS.handle(), transA, transB, m, n, k,
            Ref(Float32(α)), A, eltype(A), max(1, stride(A, 2)), strideA,
            B, eltype(B), max(1, stride(B, 2)), strideB,
            Ref(Float32(β)), C, eltype(C), max(1, stride(C, 2)), stride(C, 3),
            size(C, 3), compute, CUBLAS.CUBLAS_GEMM_DEFAULT)
        return C
    end
end

Base.unsafe_convert(::Type{CuPtr{T}}, A::NNlib.BatchedAdjOrTrans{T}) where {T} =
    Base.unsafe_convert(CuPtr{T}, parent(A))
