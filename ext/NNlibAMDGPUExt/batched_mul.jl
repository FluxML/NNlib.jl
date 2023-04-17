function _blas_at(x)
    Base.stride(x, 1) == 1 && return x, 'N'
    Base.stride(x, 2) == 1 && return batched_transpose(x), 'T'
    throw(ArgumentError("""
    Unsupported array layout for batched mul.
    - Size: $(size(x))
    - Strides: $(strides(x))
    """))
end

function NNlib._batched_mul!(
    ::Type{AT}, C, A, B, α::Float16, β::Float16,
) where AT <: ROCArray{Float16}
    blasA, transA = _blas_at(A)
    blasB, transB = _blas_at(B)
    NNlib._batched_gemm!(AT, transA, transB, α, blasA, blasB, β, C)
    C
end

function NNlib._batched_gemm!(
    ::Type{<:ROCArray{T}}, transA::Char, transB::Char, α::T, A, B, β::T, C,
) where T <: Union{MIOPENFloat, Float64}
    AMDGPU.rocBLAS.gemm_batched!(transA, transB, α, A, B, β, C)
end
