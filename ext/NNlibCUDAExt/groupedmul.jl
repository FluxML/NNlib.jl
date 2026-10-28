grouped_batch(A) = reshape(A isa DenseCuArray ? A : copy(A), size(A)..., 1)
grouped_batch(A::LinearAlgebra.Transpose) = batched_transpose(grouped_batch(parent(A)))
grouped_batch(A::LinearAlgebra.Adjoint) = batched_adjoint(grouped_batch(parent(A)))

function NNlib._grouped_product!(C::StridedCuMatrix{T}, A, B, α, β) where {T<:CUDABatchedHalf}
    isempty(C) && return C
    size(A, 2) == 0 && return NNlib.scale!(C, β)
    out = C isa DenseCuArray ? C : iszero(β) ? similar(C) : copy(C)
    NNlib.batched_mul!(grouped_batch(out), grouped_batch(A), grouped_batch(B), α, β)
    out === C || copyto!(C, out)
    return C
end

@static if pkgversion(CUDA) >= v"6.5"
    const CUDAGroupedFloat = Union{Float16,BFloat16,Float32,Float64}

    function grouped_blas_matrix(A)
        if A isa LinearAlgebra.Transpose || A isa LinearAlgebra.Adjoint
            p = parent(A)
            p isa StridedCuMatrix && stride(p, 1) == 1 && stride(p, 2) > 0 || return nothing
            return (p, 'T')
        end
        A isa StridedCuMatrix && stride(A, 1) == 1 && stride(A, 2) > 0 || return nothing
        return (A, 'N')
    end

    function NNlib._grouped_mul!(::Type{<:CuArray{T}}, C, A, B, ends, α, β) where {T<:CUDAGroupedFloat}
        minimum_version = T <: CUDABatchedHalf ? v"12.5" : v"12.4"
        CUBLAS.version() >= minimum_version || return NNlib._grouped_mul_generic!(C, A, B, ends, α, β)
        isempty(ends) && return C
        products = [NNlib._grouped_views(C, A, B, g, NNlib._grouped_range(ends, g)) for g in eachindex(ends)]
        active = filter(p -> !isempty(p[1]) && size(p[2], 2) > 0, products)
        if !isempty(active)
            as = map(p -> grouped_blas_matrix(p[2]), active)
            bs = map(p -> grouped_blas_matrix(p[3]), active)
            if any(isnothing, as) || any(isnothing, bs) ||
               any(p -> !(p[1] isa StridedCuMatrix) || stride(p[1], 1) != 1 || stride(p[1], 2) <= 0, active)
                return NNlib._grouped_mul_generic!(C, A, B, ends, α, β)
            end
            CUBLAS.gemm_grouped_batched!(last.(as), last.(bs), α, first.(as), first.(bs), β,
                                         first.(active))
        end
        for (c, a, _) in products
            size(a, 2) == 0 && NNlib.scale!(c, β)
        end
        return C
    end
end
