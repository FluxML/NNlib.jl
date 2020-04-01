# batch-wise matrix multiplication,
# including a wrapper for batched_gemm!

export batched_mul, batched_transpose, batched_adjoint

using LinearAlgebra: BlasFloat

include("./batchedadjtrans.jl")

"""
    batched_mul(A, B) -> C

Batched matrix multiplication. Result has `C[:,:,k] == A[:,:,k] * B[:,:,k]` for all `k`.

Using `batched_transpose(A)` will transpose each `A[:,:,k]`,
and similarly `batched_adjoint(B)` will use `adjoint(B[:,:,k])`.

It will also accept `A` or `B` which are `PermutedDimsArray{T,3}`.
On the CPU, these will still be handled by `BLAS.gemm!` provided `T <: LinearAlgebra.BlasFloat`
and they can be permuted to be column-major. For `T <: Real`, this allows any permutations
so long as `Base.stride(A,3) != 1` and `Base.stride(B,3) != 1`.
(For `T <: Complex`, instead you must have `Base.stride(A,1) == 1 == Base.stride(B,1)`.)

Other cases will fall back to `batched_mul_generic!`, which logs a message via `@debug`.
```
julia> A = PermutedDimsArray(rand(5,4,10), (2,1,3)); size(A)
(4, 5, 10)

julia> strides(A)  # this will be absorbed by transposing
(5, 1, 20)

julia> B = PermutedDimsArray(rand(5,10,6), (1,3,2)); size(B)
(5, 6, 10)

julia> strides(B)  # this is fine as it is
(1, 50, 5)

julia> ENV["JULIA_DEBUG"] = NNlib;

julia> C = batched_mul(A, B); size(C)  # done by batched_gemm!
(4, 6, 10)

julia> A2 = PermutedDimsArray(rand(10,5,4), (3,2,1)); size(A2)
(4, 5, 10)

julia> strides(A2)  # this can't be fixed
(50, 10, 1)

julia> C2 = batched_mul(A2, B); size(C2)
┌ Debug: couldn't re-arrange strides for batched_gemm!
│   strides(A) = (50, 10, 1)
│   strides(B) = (1, 50, 5)
│   strides(C) = (1, 4, 24)
└ @ NNlib ~/.julia/dev/NNlib/src/batched/batchedmul.jl:112
┌ Debug: calling fallback method for batched_mul!
│   typeof(A) = PermutedDimsArray{Float64,3,(3, 2, 1),(3, 2, 1),Array{Float64,3}}
│   typeof(B) = PermutedDimsArray{Float64,3,(1, 3, 2),(1, 3, 2),Array{Float64,3}}
│   typeof(C) = Array{Float64,3}
└ @ NNlib ~/.julia/dev/NNlib/src/batched/batchedmul.jl:133
(4, 6, 10)
```
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
batched_mul!(C::AbstractArray{<:Any,3}, A::AbstractArray{<:Any,3}, B::AbstractArray{<:Any,3}) =
    batched_mul_cpu!(C, A, B)
# CuArrays can have a more specific method for batched_mul!, which looks innside for storage.
# If rejected, it can call batched_mul_cpu! to continue:

function batched_mul_cpu!(C::AbstractArray{<:Any,3}, A::AbstractArray{<:Any,3}, B::AbstractArray{<:Any,3})
    if eltype(A) <: BlasFloat &&
        eltype(A) == eltype(B) == eltype(C) &&
        is_strided(A) && is_strided(B) && is_strided(C)
        # Now it's safe to call strides(A), and there's a chance batched_gemm! may be legal.
        batched_try_gemm!(C, A, B)
    else
        batched_mul_generic!(C, A, B)
    end
end

_unbatch(A) = A
_unbatch(A::BatchedAdjOrTrans) = A.parent

_perm12(A::AbstractArray) = PermutedDimsArray(A, (2,1,3))
_perm12(A::PermutedDimsArray{<:Any,3,(2,1,3)}) = parent(A)
_perm12(A::PermutedDimsArray{T,3,P}) where {T,P} = PermutedDimsArray(parent(A), (P[2], P[1], P[3]))

_BATCHED_GEMM_LIST = [
    (:(AbstractArray{T, 3}), 'N', :identity),
    (:(BatchedTranspose{T}), 'T', :batched_transpose),
    (:(BatchedAdjoint{T}),   'C', :batched_adjoint)
]

# batched_gemm! is happy with PermutedDimsArray, which is not StridedArray, but needs:
# (1) all same eltype <: BlasFloat, and
# (2) all with Base.stride(X,1) == 1 where A,B might be batched_adjoint(X) etc.

for (TA, transA, fA) in _BATCHED_GEMM_LIST, (TB, transB, fB) in _BATCHED_GEMM_LIST

    @eval function batched_try_gemm!(C::AbstractArray{T, 3}, A::$TA, B::$TB) where {T<:BlasFloat}
        Abase, Bbase = _unbatch(A), _unbatch(B)

        # Best case, we can call batched_gemm! immediately:
        if Base.stride(Abase,1) == Base.stride(Bbase,1) == Base.stride(C,1) == 1
            batched_gemm!($transA, $transB, one(T), _unbatch(A), _unbatch(B), zero(T), C)

        # Second-best, can we fix it by Perm.ing the base, and adjusing 'T' label?
        # But only if we won't produce BatchedTranspose(BatchedAdjoint(complex array)).
        elseif Base.stride(Abase,2) == 1 && !(T<:Complex && $TA<:BatchedAdjoint)
            newAbase = batched_transpose(_perm12(Abase))
            return batched_try_gemm!(C, $fA(newAbase), B)

        elseif Base.stride(Bbase,2) == 1 && !(T<:Complex && $TB<:BatchedAdjoint)
            newBbase = batched_transpose(_perm12(Bbase))
            return batched_try_gemm!(C, A, $fB(newBbase))

        # Fallback, e.g when Base.stride(A,3)==1
        else
            @debug "couldn't re-arrange strides for batched_gemm!" strides(A) strides(B) strides(C)
            batched_mul_generic!(C, A, B)
        end
        C
    end

end

# fallback

_BATCHED_LIST = [
    (:(AbstractArray{<:Any, 3}), :identity),
    (:BatchedTranspose,          :transpose),
    (:BatchedAdjoint,            :adjoint),
]
for (TA, fA) in _BATCHED_LIST, (TB, fB) in _BATCHED_LIST

    @eval function batched_mul_generic!(C::AbstractArray{<:Any, 3}, A::$TA, B::$TB)
        axes(A, 3) == axes(B, 3) == axes(C, 3) || throw(DimensionMismatch("batch size mismatch"))
        @debug "calling fallback method for batched_mul!" typeof(A) typeof(B) typeof(C)
        Abase, Bbase = _unbatch(A), _unbatch(B)
        @inbounds for k in axes(C, 3)
            @views mul!(C[:,:,k], $fA(Abase[:,:,k]), $fB(Bbase[:,:,k]))
        end
        C
    end

end


"""
    is_strided(A::AbstractArray)

This generalises `A isa StridedArray` to treat wrappers like `A::PermutedDimsArray`,
for which it returns `is_strided(parent(A))`.

Other wrappers (defined outside Base, LinearAlgebra) are assumed not to break
strided-ness, and hence also return `is_strided(parent(A))`.
This correctly handles things like `NamedDimsArray` wihch don't alter indexing.
However, it's a little pessimistic in that e.g. a `view` of such a container will return
`false`, even in cases where the same `view` of `parent(A)` would be a `StridedArray`.

`A::Transpose` doesn't currently define `strides`, so for now returns `false`.
(I guess `Adjoint(A)` should only unwrapped when its elements are real numbers.)

"""
is_strided(A::StridedArray) = true
is_strided(A) = false
function is_strided(A::AbstractArray)
    M = parentmodule(typeof(A))
    if parent(A) === A # SparseMatrix, StaticArray, etc
        false
    elseif M === Base || M === Core || M ===LinearAlgebra
        # bad reshapes, etc, plus Diagonal, UpperTriangular, etc.
        false
    else
        is_strided(parent(A)) # PermutedDimsArray, NamedDimsArray
    end
end

if hasmethod(Base.strides, Tuple{LinearAlgebra.Transpose})
    # https://github.com/JuliaLang/julia/pull/29135
    is_strided(A::LinearAlgebra.Transpose) = is_strided(parent(A))
    is_strided(A::LinearAlgebra.Adjoint) = eltype(A) <: Real && is_strided(parent(A))
else
    is_strided(A::LinearAlgebra.Transpose) = false
    is_strided(A::LinearAlgebra.Adjoint) = false
end
