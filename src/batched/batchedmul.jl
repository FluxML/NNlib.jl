
export batched_mul, batched_transpose, batched_adjoint

using LinearAlgebra: BlasFloat, BlasReal

using Base: promote_typejoin

using ArrayLayouts: MemoryLayout, UnitStride, AbstractColumnMajor, ConjLayout, StridedLayout, UnknownLayout, AbstractStridedLayout

const UnitStrideFirst = Union{UnitStride{1}, AbstractColumnMajor}
const MaybeConjStrided = Union{AbstractStridedLayout, ConjLayout{<:AbstractStridedLayout}}

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
function batched_mul!(C::AbstractArray{T,3}, A::AbstractArray{<:Any,3}, B::AbstractArray{<:Any,3},
        α::Number=one(T), β::Number=zero(T)) where {T}
    # Use promote_typejoin here to ensure Float64 * Int doesn't go to gemm!
    type = promote_typejoin(storage_type(C), promote_typejoin(storage_type(A), storage_type(B)))
    _batched_mul!(type, C, memory_layout(C), A, memory_layout(A), B, memory_layout(B), α, β)
    C
end

_BATCHED_GEMM_LIST = [
    (:UnitStrideFirst,             'N', :identity),
    (:(UnitStride{2}),             'T', :batched_transpose),
    (:(ConjLayout{UnitStride{2}}), 'C', :batched_adjoint)
]
for (MA, tA, fA) in _BATCHED_GEMM_LIST, (MB, tB, fB) in _BATCHED_GEMM_LIST

    @eval function _batched_mul!(::Type{<:Array{T}}, C, ::UnitStrideFirst, A, ::$MA, B, ::$MB,
            α::Number, β::Number) where {T<:BlasFloat}
        batched_gemm!($tA, $tB, convert(T,α), $fA(A), $fB(B), convert(T,β), C)
    end

end

function _batched_mul!(::Type{<:AbstractArray{T}}, C, ::UnitStride{2},
        A, ::MaybeConjStrided, B, ::MaybeConjStrided, α::Number, β::Number) where {T<:BlasFloat}
    batched_mul!(batched_transpose(C), batched_transpose(B), batched_transpose(A), α, β)
end

function _batched_mul!(::Type{<:AbstractArray}, C, ::MemoryLayout, A, ::MemoryLayout, B, ::MemoryLayout,
        α::Number, β::Number)
    batched_mul_generic!(C, A, B, α, β)
end

# Fallback: only here do we look directly at types BatchedTranspose etc.

_unbatch(A) = A
_unbatch(A::BatchedAdjOrTrans) = A.parent

_BATCHED_LIST = [
    (:(AbstractArray{<:Any, 3}), :identity),
    (:BatchedTranspose,          :transpose),
    (:BatchedAdjoint,            :adjoint),
]
for (TA, fA) in _BATCHED_LIST, (TB, fB) in _BATCHED_LIST

    @eval function batched_mul_generic!(C::AbstractArray{T, 3}, A::$TA, B::$TB,
            α::Number=one(T), β::Number=zero(T)) where {T}
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

end


"""
    storage_type(A)

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
    memory_layout(A)

This is usually `ArrayLayouts.MemoryLayout(A)`.

The exception is that, for wrapper types which that package does not know about,
and for which `parent(A)` has any `AbstractStridedLayout`,
it will use `strides(A)` to return `UnitStride{1}()`, `UnitStride{2}()`, or `StridedLayout()`.
(And if parent(A) is conjugated, then `ConjLayout{UnitStride{1}}()` etc.)
"""
memory_layout(A)  = _memory_layout(A, MemoryLayout(A))

_memory_layout(A, M::AbstractStridedLayout) = M
_memory_layout(A, M::ConjLayout{<:AbstractStridedLayout}) = M

function _memory_layout(A, ::MemoryLayout)
    P = parent(A)
    typeof(A) === typeof(P) && return UnknownLayout()
    # Now it's a wrapper. If it contains something strided,
    # then we go by the strides of A, since those of P may be re-ordered.
    if MemoryLayout(P) isa AbstractStridedLayout
        @debug "using runtime strides" typeof(A) strides(A)
        return _find_unit_stride(A)
    elseif MemoryLayout(P) isa ConjLayout{<:AbstractStridedLayout}
        @debug "using runtime strides, parent is conjugated" typeof(A) strides(A)
        return ArrayLayouts.conjlayout(eltype(A), _find_unit_stride(A))
    else
        return UnknownLayout()
    end
end

function _find_unit_stride(A)
    s = Base.strides(A)
    if s[1] == 1
        return UnitStride{1}()
    elseif ndims(A) >= 2 && s[2] == 1
        return UnitStride{2}()
    else
        return StridedLayout()
    end
end
