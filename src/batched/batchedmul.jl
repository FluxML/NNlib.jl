
export batched_mul, batched_transpose, batched_adjoint

include("./batchedadjtrans.jl")

using LinearAlgebra: BlasFloat, Transpose, Adjoint

_unbatch(A) = A
_unbatch(A::BatchedAdjOrTrans) = parent(A)

"""
    batched_mul(A, B) -> C

Batched matrix multiplication. Result has `C[:,:,k] == A[:,:,k] * B[:,:,k]` for all `k`.
If `size(B,3) == 1` then instead `C[:,:,k] == A[:,:,k] * B[:,:,1]`, and similarly for `A`.

To transpose each matrix apply `batched_transpose` to the array,
and similarly `batched_adjoint`. Other permutations are also handled efficiently,
provided that the batch index `k` is not the first dimension of the underlying array.
Thus `PermutedDimsArray(::Array, (1,3,2))` and `PermutedDimsArray(::Array, (3,1,2))` are fine,
but `PermutedDimsArray(::Array, (3,2,1))` will use the fallback `batched_mul_generic!`.

There is an `@debug` message produced by `batched_mul_generic!`,
setting for instance `ENV["JULIA_DEBUG"] = NNlib` will display this.
"""
function batched_mul(A::AbstractArray{T1, 3}, B::AbstractArray{T2, 3}) where {T1, T2}
    size(A, 3) == size(B, 3) || size(A, 3) == 1 || size(B, 3) == 1 ||
        throw(DimensionMismatch("batch size mismatch: A != B"))
    T = promote_type(T1, T2)
    C = similar(A, T, (size(A, 1), size(B, 2), max(size(A, 3), size(B, 3))))
    batched_mul!(C, A, B)
end

"""
    batched_mul!(C, A, B) -> C
    batched_mul!(C, A, B, α=1, β=0)

In-place batched matrix multiplication, equivalent to
`mul!(C[:,:,k], A[:,:,k], B[:,:,k], α, β)` for all `k`.
If `size(B,3) == 1` then every batch uses `B[:,:,1]` instead.

This will call `batched_gemm!` whenever possible. For real arrays this means that,
for `X ∈ [A,B,C]`, either `strides(X,1)==1` or `strides(X,2)==1`, the latter may
be caused by `batched_transpose` or by for instance `PermutedDimsArray(::Array, (3,1,2))`.

For complex arrays, the wrapper made by `batched_adjoint` must be outermost to be seen,
and in this case `stride(A::BatchedAdjoint,2) == 1` is not optional.

The fallback method calls 5-argument `mul!` on Julia 1.3 and later.
On earlier verions it will thrown an error if `α!=1` or `β!=0`.
"""
function batched_mul!(C::AbstractArray{T,3}, A::AbstractArray{<:Any,3}, B::AbstractArray{<:Any,3},
        α::Number=one(T), β::Number=zero(T)) where {T}
    _batched_mul!(storage_type(C,A,B), C, A, B, α, β)
    C
end

_batched_mul!(::Type, C, A, B, α::Number, β::Number) = batched_mul_generic!(C, A, B, α, β)

function _batched_mul!(CT::Type{<:DenseArray{T}}, C, A, B, α::Number, β::Number) where {T<:BlasFloat}

    are_strided(C, _unbatch(A), _unbatch(B)) || return batched_mul_generic!(C, A, B, α, β)

    if Base.stride(C,1) == 1
    elseif Base.stride(C,2) == 1
        @debug "transposing C = A * B into Cᵀ = Bᵀ * Aᵀ" size(C) strides(C)
        return batched_mul!(batched_transpose(C), batched_transpose(B), batched_transpose(A), α, β)
    else
        return batched_mul_generic!(C, A, B, α, β)
    end

    blasA, transA = if A isa BatchedAdjoint
        Base.stride(parent(A),1) == 1 || return batched_mul_generic!(C, A, B, α, β)
        parent(A), 'C'
    elseif Base.stride(A,1) == 1
        A, 'N'
    elseif Base.stride(A,2) == 1
        batched_transpose(A), 'T'
    else
        return batched_mul_generic!(C, A, B, α, β)
    end

    blasB, transB = if B isa BatchedAdjoint
        Base.stride(parent(B),1) == 1 || return batched_mul_generic!(C, A, B, α, β)
        parent(B), 'C'
    elseif Base.stride(B,1) == 1
        B, 'N'
    elseif Base.stride(B,2) == 1
        batched_transpose(B), 'T'
    else
        return batched_mul_generic!(C, A, B, α, β)
    end

    _batched_gemm!(CT, transA, transB, convert(T,α), blasA, blasB, convert(T,β), C)
    C
end

_batched_gemm!(::Type{<:Array}, transA::Char, transB::Char, α::Number, A, B, β::Number, C) =
    batched_gemm!(transA, transB, α, A, B, β, C)

_BATCHED_LIST = [
    (:(AbstractArray{<:Any, 3}), :identity),
    (:BatchedTranspose,          :transpose),
    (:BatchedAdjoint,            :adjoint),
]
for (TA, fA) in _BATCHED_LIST, (TB, fB) in _BATCHED_LIST

    @eval function batched_mul_generic!(C::AbstractArray{T, 3}, A::$TA, B::$TB,
            α::Number=one(T), β::Number=zero(T)) where {T}
        size(A, 3) == size(C, 3) || size(A, 3) == 1 || throw(DimensionMismatch("batch size mismatch: A != C"))
        size(B, 3) == size(C, 3) || size(B, 3) == 1 || throw(DimensionMismatch("batch size mismatch: B != C"))
        @debug "calling fallback method for batched_mul!" typeof(A) typeof(B) typeof(C)
        Abase, Bbase = _unbatch(A), _unbatch(B)
        sA, oA = size(A,3) == 1 ? (0,1) : (1,0)
        sB, oB = size(B,3) == 1 ? (0,1) : (1,0)

        if VERSION >= v"1.3"
            @inbounds for k in 1:size(C,3)
                @views mul!(C[:,:,k], $fA(Abase[:,:,k*sA+oA]), $fB(Bbase[:,:,k*sB+oB]), convert(T,α), convert(T,β))
            end
        else
            α==1 && β==0 || throw(ArgumentError("5-arg batched_mul_generic! does not work on Julia < 1.3"))
            @inbounds for k in 1:size(C,3)
                @views mul!(C[:,:,k], $fA(Abase[:,:,k*sA+oA]), $fB(Bbase[:,:,k*sB+oB]))
            end
        end

        C
    end

end

"""
    storage_type(A) -> Type

Removes all wrappers to return the `Array` or `CuArray` (or whatever) type within.
```
julia> view(reshape(ones(10)',2,5),:, 3:4) |> storage_type
Array{Float64,1}

julia> reshape(sparse(rand(10)), 5,2) |> storage_type
SparseVector{Float64,Int64}
```
"""
function storage_type(A::AbstractArray)
    P = parent(A)
    typeof(A) === typeof(P) ? typeof(A) : storage_type(P)
end
storage_type(A) = typeof(A)

"""
    storage_type(A, B, C, ...) -> Type

Reduces with `Base.promote_typejoin`, in order that this conveys useful information
for dispatching to BLAS, rather than information about the storage to allocate:
```
julia> storage_type(rand(2), rand(Float32, 2))
Array{T,1} where T

julia> eltype(ans) <: LinearAlgebra.BlasFloat
false

julia> storage_type(rand(2), rand(2,3), rand(2,3,4))
Array{Float64,N} where N
```
"""
storage_type(A, Bs...) = Base.promote_typejoin(storage_type(A), storage_type(Bs...))


"""
    is_strided(A::AbstractArray) -> Bool

This generalises `A isa StridedArray` to treat wrappers like `A::PermutedDimsArray`,
for which it returns `is_strided(parent(A))`.

Other wrappers (defined outside Base, LinearAlgebra) are assumed not to break
strided-ness, and hence also return `is_strided(parent(A))`.
This correctly handles things like `NamedDimsArray` wihch don't alter indexing.
However, it's a little pessimistic in that e.g. a `view` of such a container will return
`false`, even in cases where the same `view` of `parent(A)` would be a `StridedArray`.

`A::Transpose` doesn't currently define `strides`, until that's fixed this returns `false`.
The PR to fix that only defines `strides(::Adjoint{T})` for `T<:Real`, so this will follow.
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

is_strided(A::BatchedAdjoint) = eltype(A) <: Real && is_strided(parent(A))
is_strided(A::BatchedTranspose) = is_strided(parent(A))

if hasmethod(Base.strides, Tuple{LinearAlgebra.Transpose})
    # https://github.com/JuliaLang/julia/pull/29135
    is_strided(A::LinearAlgebra.Transpose) = is_strided(parent(A))
    is_strided(A::LinearAlgebra.Adjoint) = eltype(A) <: Real && is_strided(parent(A))
else
    is_strided(A::LinearAlgebra.Transpose) = false
    is_strided(A::LinearAlgebra.Adjoint) = false
end

are_strided(As...) = mapfoldl(is_strided, &, As; init=true)
