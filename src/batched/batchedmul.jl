_unbatch(A) = A
_unbatch(A::BatchedAdjOrTrans) = parent(A)

"""
    batched_mul(A, B) -> C
    A ⊠ B  # \\boxtimes

Batched matrix multiplication. Result has `C[:,:,k...] == A[:,:,k...] * B[:,:,k...]` where `k...` represent 
any indices in the last dimensions.

If `ndims(A) == ndims(B) == 3` and `size(B,3) == 1` then instead `C[:,:,k] == A[:,:,k] * B[:,:,1]`, and similarly for `A`.

To transpose each matrix, apply `batched_transpose` to the array,
or `batched_adjoint` for conjugate-transpose:

```jldoctest
julia> A, B = randn(2,5,17), randn(5,9,17);

julia> A ⊠ B |> size
(2, 9, 17)

julia> batched_adjoint(A) |> size
(5, 2, 17)

julia> batched_mul(A, batched_adjoint(randn(9,5,17))) |> size
(2, 9, 17)

julia> A ⊠ randn(5,9,1) |> size
(2, 9, 17)

julia> batched_transpose(A) == PermutedDimsArray(A, (2,1,3))
true
```

The equivalent `PermutedDimsArray` may be used in place of `batched_transpose`.
Other permutations are also handled by BLAS,
provided that the batch index `k` is not the first dimension of the underlying array.
Thus `PermutedDimsArray(::Array, (1,3,2))` and `PermutedDimsArray(::Array, (3,1,2))` are fine.

However, `A = PermutedDimsArray(::Array, (3,2,1))` is not acceptable to BLAS,
since the batch dimension is the contiguous one: `stride(A,3) == 1`.
This will be copied, as doing so is faster than `batched_mul_generic!`.

Both this `copy` and `batched_mul_generic!` produce `@debug` messages,
and setting for instance `ENV["JULIA_DEBUG"] = NNlib` will display them.
"""
function batched_mul(x::AbstractArray{T1,N}, y::AbstractArray{T2,N}) where {T1,T2,N}
    batch_size = size(x)[3:end]
    @assert batch_size == size(y)[3:end] "batch size has to be the same for the two arrays."
    x2 = reshape(x, size(x, 1), size(x, 2), :)
    y2 = reshape(y, size(y, 1), size(y, 2), :)
    z = batched_mul(x2, y2)
    return reshape(z, size(z, 1), size(z, 2), batch_size...)
  end

function batched_mul(A::AbstractArray{T1, 3}, B::AbstractArray{T2, 3}) where {T1, T2}
    size(A, 3) == size(B, 3) || size(A, 3) == 1 || size(B, 3) == 1 ||
        throw(DimensionMismatch("batch size mismatch: A != B"))
    _batched_mul(storage_typejoin(A, B), A, B)
end

const ⊠ = batched_mul

function _batched_mul(::Type, A, B)
    T = promote_type(eltype(A), eltype(B))
    C = similar(A, T, (size(A, 1), size(B, 2), max(size(A, 3), size(B, 3))))
    batched_mul!(C, A, B)
    C
end
function _batched_mul(::Type{<:DenseArray{T}}, A, B) where {T<:BlasFloat}
    C = similar(A, T, (size(A, 1), size(B, 2), max(size(A, 3), size(B, 3))))
    batched_mul!(C, _copy_if_faster(A), _copy_if_faster(B))
    C
end

function _copy_if_faster(X::AbstractArray{<:Number, 3})
    is_strided(X) || return X
    if Base.stride(X, 3) == 1 && Base.stride(X, 1) != 1
        @debug "copying to avoid batched_mul_generic!" typeof(X) size(X) strides(X)
        return copy(X)
    end
    X
end
function _copy_if_faster(X::BatchedAdjoint{<:Complex})
    Xbase = _unbatch(X)
    is_strided(Xbase) || return X
    if Base.stride(Xbase, 1) != 1
        @debug "copying to avoid batched_mul_generic!" typeof(X) size(X) strides(_unbatch(X))
        return copy(X) # or batched_adjoint(copy(Xbase)), may be better on GPU?
    end
    X
end

# Gradient, allowing that size(A,3)==1 means it's "broadcasted" out to size(B,3)

function rrule(::typeof(batched_mul), A::AbstractArray{<:Any,3}, B::AbstractArray{<:Any,3})
    function batched_mul_pullback(_Δ)
        Δ = unthunk(_Δ)
        Athunk = @thunk begin
            tmp = batched_mul(Δ, batched_adjoint(B))
            size(A,3) == 1 ? sum(tmp, dims=3) : tmp
        end
        Bthunk = @thunk begin
            tmp = batched_mul(batched_adjoint(A), Δ)
            size(B,3) == 1 ? sum(tmp, dims=3) : tmp
        end
        return (NoTangent(), Athunk, Bthunk)
    end
    batched_mul(A, B), batched_mul_pullback
end

"""
    batched_mul(A::Array{T,3}, B::Matrix)
    batched_mul(A::Matrix, B::Array{T,3})
    A ⊠ B

This is always matrix-matrix multiplication, but
either `A` or `B` may lack a batch index.

* When `B` is a matrix, result has `C[:,:,k] == A[:,:,k] * B[:,:]` for all `k`.

* When `A` is a matrix, then `C[:,:,k] == A[:,:] * B[:,:,k]`.
  This can also be done by reshaping and calling `*`,
  for instance `A ⊡ B` using TensorCore.jl, but is implemented here using
  `batched_gemm` instead of `gemm`.

```jldoctest
julia> randn(16,8,32) ⊠ randn(8,4) |> size
(16, 4, 32)

julia> randn(16,8,32) ⊠ randn(8,4,1) |> size  # equivalent
(16, 4, 32)

julia> randn(16,8) ⊠ randn(8,4,32) |> size
(16, 4, 32)
```

See also `batched_vec` to regard `B` as a batch of vectors, `A[:,:,k] * B[:,k]`.
"""
batched_mul(A::AbstractArray{T,3} where T, B::AbstractMatrix) = _semi_batched_mul(A,B)

# Simplify signature of batched_mul by hiding dispatch on Adjoint etc:

_semi_batched_mul(A::AbstractArray{<:Any,3}, B::AbstractMatrix) =
    batched_mul(A, reshape(B, size(B)..., 1))

_semi_batched_mul(A::AbstractArray{<:Any,3}, B::Adjoint{<:Number,<:AbstractMatrix}) =
    batched_mul(A, batched_adjoint(reshape(parent(B), size(parent(B))..., 1)))

_semi_batched_mul(A::AbstractArray{<:Any,3}, B::Transpose{<:Number,<:AbstractMatrix}) =
    batched_mul(A, batched_transpose(reshape(parent(B), size(parent(B))..., 1)))

batched_mul(A::AbstractMatrix, B::AbstractArray{T,3} where T) = _semi_batched_mul(A,B)

_semi_batched_mul(A::AbstractMatrix, B::AbstractArray{<:Any,3}) =
    batched_mul(reshape(A, size(A)..., 1), B)

_semi_batched_mul(A::Adjoint{<:Number,<:AbstractMatrix}, B::AbstractArray{<:Any,3}) =
    batched_mul(batched_adjoint(reshape(parent(A), size(parent(A))..., 1)), B)

_semi_batched_mul(A::Transpose{<:Number,<:AbstractMatrix}, B::AbstractArray{<:Any,3}) =
    batched_mul(batched_transpose(reshape(parent(A), size(parent(A))..., 1)), B)

"""
    batched_vec(A::AbstractArray{T,3}, B::AbstractMatrix)
    batched_vec(A::AbstractArray{T,3}, b::AbstractVector)
    batched_vec(A::AbstractArray, B::AbstractArray)

Batched matrix-vector multiplication. For the 3D case:
the result has `C[:,:,k] == A[:,:,k] * B[:,k]` for all `k`,
or else `C[:,:,k] == A[:,:,k] * b` for `b::Vector`.

For the general N-D case where `ndims(A) == ndims(B) + 1`:
the result has `C[:,k...] == A[:,:,k...] * B[:,k...]` for all batch indices `k...`.
The batch dimensions must match: `size(A)[3:end] == size(B)[2:end]`.

With the same argument types, `batched_mul(A, B)` would regard `B` as
a fixed matrix, not a batch of vectors. Both reshape and then
call `batched_mul(::Array{T,3}, ::Array{T,3})`.

```jldoctest
julia> A, B, b = randn(16,8,32), randn(8,32), randn(8);

julia> batched_vec(A,B) |> size
(16, 32)

julia> batched_vec(A,b) |> size
(16, 32)

julia> A4d, B3d = randn(16,8,10,32), randn(8,10,32);  # 4D and 3D arrays

julia> batched_vec(A4d, B3d) |> size
(16, 10, 32)
```
"""
function batched_vec(A::AbstractArray, B::AbstractArray)
    ndims(A) == ndims(B) + 1 || throw(DimensionMismatch(
        "batched_vec requires ndims(A) == ndims(B) + 1, got ndims(A)=$(ndims(A)) and ndims(B)=$(ndims(B))"))
    size(A)[3:end] == size(B)[2:end] || throw(DimensionMismatch(
        "batch dimensions must match: size(A)[3:end]=$(size(A)[3:end]) != size(B)[2:end]=$(size(B)[2:end])"))
    
    # Reshape B to add a singleton dimension for matrix multiplication
    B_reshaped = reshape(B, size(B, 1), 1, size(B)[2:end]...)
    # Perform batched multiplication
    C = batched_mul(A, B_reshaped)
    # Remove the singleton dimension
    return dropdims(C, dims=2)
end

batched_vec(A::AbstractArray{T,3} where T, B::AbstractMatrix) =
    reshape(batched_mul(A, reshape(B, size(B,1), 1, size(B,2))), size(A,1), size(A,3))

# If B is transposed, then stride=1 is the batch dim, so we will end up copying anyway:
batched_vec(A::AbstractArray{T,3} where T, B::AdjOrTransAbsMat{<:BlasFloat, <:StridedMatrix}) =
    batched_vec(A, copy(B))

batched_vec(A::AbstractArray{T,3} where T, b::AbstractVector) =
    reshape(batched_mul(A, reshape(b, length(b), 1, 1)), size(A,1), size(A,3))


"""
    batched_mul!(C, A, B) -> C
    batched_mul!(C, A, B, α=1, β=0)

In-place batched matrix multiplication, equivalent to
`mul!(C[:,:,k], A[:,:,k], B[:,:,k], α, β)` for all `k`.
If `size(B,3) == 1` then every batch uses `B[:,:,1]` instead.

This will call `batched_gemm!` whenever possible. For real arrays this means that,
for `X ∈ [A,B,C]`, either `stride(X,1)==1` or `stride(X,2)==1`, the latter may
be caused by `batched_transpose` or by for instance `PermutedDimsArray(::Array, (3,1,2))`.
Unlike `batched_mul` this will never make a copy.

For complex arrays, the wrapper made by `batched_adjoint` must be outermost to be seen.
In this case the strided accepted by BLAS are more restricted, if `stride(C,1)==1` then
only `stride(AorB::BatchedAdjoint,2) == 1` is accepted.
"""
function batched_mul!(C::AbstractArray{T,3}, A::AbstractArray{<:Any,3}, B::AbstractArray{<:Any,3},
        α::Number=one(T), β::Number=zero(T)) where {T}
    _batched_mul!(storage_typejoin(C,A,B), C, A, B, α, β)
    C
end

_batched_mul!(::Type, C, A, B, α::Number, β::Number) = batched_mul_generic!(C, A, B, α, β)

_batched_mul!(::Type{DT}, C, A, B, α::Number, β::Number) where {DT<:DenseArray{T}} where {T<:BlasFloat} =
    _batched_try_gemm!(DT, C, A, B, α, β)

function _batched_try_gemm!(::Type{DT}, C, A, B, α::Number, β::Number) where {DT<:DenseArray{T}} where {T<:BlasFloat}
    alpha, beta = promote(α, β, zero(T))
    alpha isa T && beta isa T || return batched_mul_generic!(C, A, B, α, β)

    are_strided(_unbatch(A), _unbatch(B)) || return batched_mul_generic!(C, A, B, α, β)
    C isa StridedArray || return batched_mul_generic!(C, A, B, α, β)

    blasA, transA = if A isa BatchedAdjoint && T <: Complex
        Base.stride(parent(A),1) == 1 || return batched_mul_generic!(C, A, B, α, β)
        parent(A), 'C'
    elseif Base.stride(A,2) == 1 && size(A,1) > 1
        batched_transpose(A), 'T'
    elseif Base.stride(A,1) == 1
        A, 'N'
    elseif Base.stride(A,2) == 1  # This is awful, but exhaustively tested. Issues 268, 282.
        batched_transpose(A), 'T'
    else
        return batched_mul_generic!(C, A, B, α, β)
    end

    blasB, transB = if B isa BatchedAdjoint && T <: Complex
        Base.stride(parent(B),1) == 1 || return batched_mul_generic!(C, A, B, α, β)
        parent(B), 'C'
    elseif Base.stride(B,2) == 1 && size(B,1) > 1
        batched_transpose(B), 'T'
    elseif Base.stride(B,1) == 1
        B, 'N'
    elseif Base.stride(B,2) == 1
        batched_transpose(B), 'T'
    else
        return batched_mul_generic!(C, A, B, α, β)
    end

    _batched_gemm!(DT, transA, transB, alpha, blasA, blasB, beta, C)
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
        @debug "calling fallback method for batched_mul!" typeof(A) size(A) typeof(B) size(B) typeof(C)

        Abase, Bbase = _unbatch(A), _unbatch(B)
        sA, oA = size(A,3) == 1 ? (0,1) : (1,0)
        sB, oB = size(B,3) == 1 ? (0,1) : (1,0)

        @inbounds for k in 1:size(C,3)
            @views mul!(C[:,:,k], $fA(Abase[:,:,k*sA+oA]), $fB(Bbase[:,:,k*sB+oB]), α, β)
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
    storage_typejoin(A, B, C, ...) -> Type

Reduces with `Base.promote_typejoin`, in order that this conveys useful information
for dispatching to BLAS. It does not tell you what container to allocate:
```
julia> storage_typejoin(rand(2), rand(Float32, 2))
Array{T,1} where T

julia> eltype(ans) <: LinearAlgebra.BlasFloat
false

julia> storage_typejoin(rand(2), rand(2,3), rand(2,3,4))
Array{Float64,N} where N
```
"""
storage_typejoin(A, Bs...) = Base.promote_typejoin(storage_type(A), storage_typejoin(Bs...))
storage_typejoin(A) = storage_type(A)

"""
    is_strided(A::AbstractArray) -> Bool

This generalises `A isa StridedArray` to treat wrappers like `A::PermutedDimsArray`,
for which it returns `is_strided(parent(A))`.

It returns `true` for `CuArray`s, and `PermutedDimsArray`s of those.

Other wrappers (defined outside Base, LinearAlgebra) are assumed not to break
strided-ness, and hence also return `is_strided(parent(A))`.
This correctly handles things like `NamedDimsArray` wihch don't alter indexing.
However, it's a little pessimistic in that e.g. a `view` of such a container will return
`false`, even in cases where the same `view` of `parent(A)` would be a `StridedArray`.
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

is_strided(A::LinearAlgebra.Transpose) = is_strided(parent(A))
is_strided(A::LinearAlgebra.Adjoint) = eltype(A) <: Real && is_strided(parent(A))
# This needs Compat 3.14, for any Julia < 1.6

are_strided(As...) = mapfoldl(is_strided, &, As; init=true)
