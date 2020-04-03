
export batched_mul, batched_transpose, batched_adjoint

using LinearAlgebra: BlasFloat, BlasReal

using Base: promote_typejoin

using ArrayLayouts: MemoryLayout, UnitStride, AbstractColumnMajor, ConjLayout, StridedLayout, UnknownLayout
const UnitStrideFirst = Union{UnitStride{1}, AbstractColumnMajor}

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
    # Use promote_typejoin here to ensure Float64 * Int doesn't go to gemm!
    type = promote_typejoin(storage_type(C), promote_typejoin(storage_type(A), storage_type(B)))
    _batched_mul!(type, memory_layout(C), C, memory_layout(A), A, memory_layout(B), B, α, β)
    C
end

# Dispatch on storage type: CuArrays can define _batched_mul!(::CuArray, ...)
# Dispatch on ArrayLayouts traits: decide where you need 'T' etc.
# If the list of permutations handled by CuArrays is the same, then some duplication,
# which you could avoid by dispatching on storage type later than on layout.

_BATCHED_GEMM_LIST = [
    (:UnitStrideFirst, 'N', :identity, :(UnitStride{2})),
    (:(UnitStride{2}), 'T', :batched_transpose, :UnitStrideFirst),
    (:(ConjLayout{UnitStride{2}}), 'C', :batched_adjoint, :Nothing)
]
for (TA, tA, fA, revTA) in _BATCHED_GEMM_LIST, (TB, tB, fB, revTB) in _BATCHED_GEMM_LIST

    # Path 1, e.g. C isa Array, batched_transpose(A) or PermutedDimsArray(B, (3,1,2)) both need 'T'
    @eval function _batched_mul!(::Type{<:Array{T}}, ::UnitStrideFirst, C, ::$TA, A, ::$TB, B, α::Number, β::Number) where {T<:BlasFloat}
        batched_gemm!($tA, $tB, convert(T,α), $fA(A), $fB(B), convert(T,β), C)
    end

    # Path 2, C = batched_transpose(Array), so transpose the entire equation
    if tA != 'C' && tB != 'C'
    not_tA = tA == 'T' ? 'N' : 'T'
    not_tB = tB == 'T' ? 'N' : 'T'
    @eval function _batched_mul!(::Type{<:Array{T}}, ::UnitStride{2}, C, ::$TA, A, ::$TB, B, α::Number, β::Number) where {T<:BlasFloat}
        @warn "this is broken!"
        @debug "transposing C, and thus A, B to compensate..." size(A) size(B) size(C) strides(A) strides(B) strides(C)
        batched_gemm!($not_tB, $not_tA, convert(T,α), $fB(B), $fA(A), convert(T,β), batched_transpose(C))
    end
end
end

# Path 3, use runtime strides. Does not catch ConjLayout{StridedLayout}()
function _batched_mul!(TC::Type{<:Array{T}}, ::AbstractStridedLayout, C, ::AbstractStridedLayout, A, ::AbstractStridedLayout, B, α::Number, β::Number) where {T<:BlasFloat}
    @debug "using runtime strides" strides(A) strides(B) strides(C)

    MA = Base.stride(A,1) == 1 ? UnitStride{1}() :
        Base.stride(A,2) == 1 ? UnitStride{2}() :
        return batched_mul_generic!(C,A,B,α,β)

    MB = Base.stride(B,1) == 1 ? UnitStride{1}() :
        Base.stride(B,2) == 1 ? UnitStride{2}() :
        return batched_mul_generic!(C,A,B,α,β)

    MC = Base.stride(C,1) == 1 ? UnitStride{1}() :
        Base.stride(C,2) == 1 ? UnitStride{2}() :
        return batched_mul_generic!(C,A,B,α,β)

    # Useless, as batched_transpose would make ConjLayout{StridedLayout}()
    # MA = A isa BatchedAdjoint ? ArrayLayouts.conjlayout(T, MA) : MA
    # MB = B isa BatchedAdjoint ? ArrayLayouts.conjlayout(T, MB) : MB

    _batched_mul!(TC, MC, C, MA, A, MB, B, α, β)
end

# Path 4, anything else goes directly to the fallback
function _batched_mul!(::Type{<:AbstractArray}, ::MemoryLayout, C, ::MemoryLayout, A, ::MemoryLayout, B, α::Number, β::Number)
    batched_mul_generic!(C, A, B, α, β)
end

# Fallback: only here do we look directly at BatchedTranspose etc.

_unbatch(A) = A
_unbatch(A::BatchedAdjOrTrans) = A.parent

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
and for which `parent(A)` has any `AbstractStridedLayout`, it returns `StridedLayout()`.
(And if parent(A) is conjugated, then `ConjLayout{StridedLayout}()`.)
"""
memory_layout(A)  = _memory_layout(A, MemoryLayout(A))

_memory_layout(A, M::AbstractStridedLayout) = M
_memory_layout(A, M::ConjLayout{<:AbstractStridedLayout}) = M

function _memory_layout(A, ::MemoryLayout)
    P = parent(A)
    if typeof(A) === typeof(P)
        UnknownLayout()
    elseif MemoryLayout(P) isa AbstractStridedLayout
        StridedLayout()
    elseif MemoryLayout(P) isa ConjLayout{<:AbstractStridedLayout}
        ConjLayout{StridedLayout}()
    else
        UnknownLayout()
    end
end

