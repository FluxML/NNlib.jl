
export batched_mul, batched_transpose, batched_adjoint

using LinearAlgebra: BlasFloat, BlasReal

using ArrayLayouts: MemoryLayout, FirstMajor, SecondMajor, ConjLayout, StridedLayout, UnknownLayout

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
    batched_mul!(C, A, B, α=1, β=0) -> C
    batched_mul_generic!(C, A, B, α=1, β=0)

In-place batched matrix multiplication,
equivalent to `mul!(C[:,:,k], A[:,:,k], B[:,:,k], α, β)` for all `k`.

The fallback implementation of this literally calls `mul!`,
and hence can only accept `α!=1` or `β!=0` on Julia >= 1.3.
"""
function batched_mul!(C::AbstractArray{T,3}, A::AbstractArray{<:Any,3}, B::AbstractArray{<:Any,3}, α::Number=one(T), β::Number=zero(T)) where {T}
    type = promote_type(storage_type(C), storage_type(A), storage_type(B))
    _batched_mul!(type, memory_layout(C), C, memory_layout(A), A, memory_layout(B), B, α, β)
    C
end

# Dispatch on storage type: CuArrays can define _batched_mul!(::CuArray, ...)
# Dispatch on ArrayLayouts traits: decide where you need 'T' etc.

_BATCHED_GEMM_LIST = [
    (:FirstUnion, 'N', :identity, :FirstUnion),
    (:SecondMajor, 'T', :batched_transpose, :SecondMajor),
    (:(ConjLayout{SecondMajor}), 'C', :batched_adjoint, :Nothing)
]
for (TA, tA, fA, revTA) in _BATCHED_GEMM_LIST, (TB, tB, fB, revTB) in _BATCHED_GEMM_LIST

    # Path 1, e.g. C isa Array, batched_transpose(A) or PermutedDimsArray(B, (3,1,2)) both need 'T'
    @eval function _batched_mul!(::Type{<:Array{T}}, ::FirstUnion, C, ::$TA, A, ::$TB, B, α::Number, β::Number) where {T<:BlasFloat}
        batched_gemm!($tA, $tB, convert(T,α), $fA(A), $fB(B), convert(T,β), C)
    end

    # Path 2, C = batched_transpose(Array), so transpose the entire equation
    @eval function _batched_mul!(::Type{<:Array{T}}, ::SecondMajor, C, ::$revTA, A, ::$revTB, B, α::Number, β::Number) where {T<:BlasFloat}
        @debug "transposing C, and thus A, B to compensate..." size(A) size(B) size(C) strides(A) strides(B) strides(C)
        batched_gemm!($tB, $tA, convert(T,α), $fB(B), $fA(A), convert(T,β), batched_transpose(C))
    end


end

# Path 3, use runtime strides... this ignores complex for now!
function _batched_mul!(::Type{<:Array{T}}, ::AbstractStridedLayout, C, ::AbstractStridedLayout, A, ::AbstractStridedLayout, B, α::Number, β::Number) where {T<:BlasFloat}
    @debug "using runtime strides" strides(A) strides(B) strides(C)

    if Base.stride(C,1) != 1
        if Base.stride(C,2) == 1 && T <: Real
            return batched_mul!(batched_transpose(C), batched_transpose(B), batched_transpose(A), α, β)
        else
            return batched_mul_generic!(C,B,A,α,β)
        end
    end

    tA, fA = if Base.stride(A,1) == 1
        'N', identity
    elseif Base.stride(A,2) == 1 && T <: Real
        'T', batched_transpose
    else
        return batched_mul_generic!(C,B,A,α,β)
    end

    tB, fB = if Base.stride(B,1) == 1
        'N', identity
    elseif Base.stride(B,2) == 1 && T <: Real
        'T', batched_transpose
    else
        return batched_mul_generic!(C,B,A,α,β)
    end

    batched_gemm!(tA, tB, convert(T,α), fA(A), fB(B), convert(T,β), C)
end

# Path 4, anything else goes directly to the fallback
function _batched_mul!(::Type{<:AbstractArray}, ::MemoryLayout, C, ::MemoryLayout, A, ::MemoryLayout, B, α::Number, β::Number)
    batched_mul_generic!(C, A, B, α, β)
end


# Fallback: only here do we look directly at BatchedTranspose etc.

_BATCHED_LIST = [
    (:(AbstractArray{<:Any, 3}), :identity),
    (:BatchedTranspose,          :transpose),
    (:BatchedAdjoint,            :adjoint),
]
for (TA, fA) in _BATCHED_LIST, (TB, fB) in _BATCHED_LIST

    @eval function batched_mul_generic!(C::AbstractArray{T, 3}, A::$TA, B::$TB, α::Number=one(T), β::Number=zero(T)) where {T}
        axes(A, 3) == axes(B, 3) == axes(C, 3) || throw(DimensionMismatch("batch size mismatch"))
        @debug "calling fallback method for batched_mul!" typeof(A) typeof(B) typeof(C)
        Abase, Bbase = _unbatch(A), _unbatch(B)
        if VERSION >= v"1.3"
            @inbounds for k in axes(C, 3)
                @views mul!(C[:,:,k], $fA(Abase[:,:,k]), $fB(Bbase[:,:,k]), convert(T,α), convert(T,β))
            end
        else
            α==1 && β==0 || throw(ArgumentError("5-arg batched_mul_generic! does not work on Julia < 1.3"))
            @inbounds for k in axes(C, 3)
                @views mul!(C[:,:,k], $fA(Abase[:,:,k]), $fB(Bbase[:,:,k]))
            end
        end
        C
    end

    # if VERSION >= v"1.3"

    #     @eval function batched_mul_generic!(C::AbstractArray{T, 3}, A::$TA, B::$TB, α::Number=one(T), β::Number=zero(T)) where {T}
    #         axes(A, 3) == axes(B, 3) == axes(C, 3) || throw(DimensionMismatch("batch size mismatch"))
    #         @debug "calling fallback method for batched_mul!" typeof(A) typeof(B) typeof(C)
    #         Abase, Bbase = _unbatch(A), _unbatch(B)
    #         @inbounds for k in axes(C, 3)
    #             @views mul!(C[:,:,k], $fA(Abase[:,:,k]), $fB(Bbase[:,:,k]), convert(T,α), convert(T,β))
    #         end
    #         C
    #     end

    # else

    #     @eval function batched_mul_generic!(C::AbstractArray{<:Any, 3}, A::$TA, B::$TB, α=1, β=0)
    #         α==1 && β==0 || throw(ArgumentError("5-arg batched_mul_generic! does not work on Julia < 1.3"))
    #         axes(A, 3) == axes(B, 3) == axes(C, 3) || throw(DimensionMismatch("batch size mismatch"))
    #         @debug "calling Julia < 1.3 fallback method for batched_mul!" typeof(A) typeof(B) typeof(C)
    #         Abase, Bbase = _unbatch(A), _unbatch(B)
    #         @inbounds for k in axes(C, 3)
    #             @views mul!(C[:,:,k], $fA(Abase[:,:,k]), $fB(Bbase[:,:,k]))
    #         end
    #         C
    #     end

    # end
end

_unbatch(A) = A
_unbatch(A::BatchedAdjOrTrans) = A.parent

"""
    storage_type(A)

Unwraps all wrappers to return the `Array` or `CuArray` () type within.
```
julia> view(reshape(ones(10)',2,5),:, 3:4) |> storage_type
Array{Float64,1}
```
"""
function storage_type(A::AbstractArray)
    P = parent(A)
    typeof(A) === typeof(P) ? typeof(A) : storage_type(P)
end
storage_type(A) = typeof(A)

"""
    memory_layout(A)

This is usually `ArrayLayouts.MemoryLayout(A)`.

The exception is that, for wrapper types which that package does not know about,
and for which `parent(A)` has any `AbstractStridedLayout`, it returns `StridedLayout()`.
"""
memory_layout(A)  = _memory_layout(MemoryLayout(A), A)

_memory_layout(::AbstractStridedLayout, A) = MemoryLayout(A)

function _memory_layout(::MemoryLayout, A)
    P = parent(A)
    if typeof(A) === typeof(P)
        UnknownLayout()
    elseif MemoryLayout(P) isa AbstractStridedLayout
        StridedLayout()
    else
        UnknownLayout()
    end
end

#=

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
        elseif Base.stride(C,2) == 1
            newAbase = _perm12(Abase)
            newBbase = _perm12(Bbase)
            newC = _perm12(C)
            return batched_try_gemm!(newC, $fB(newBbase), $fA(newAbase))

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
=#

