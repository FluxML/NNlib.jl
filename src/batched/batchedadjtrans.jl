using LinearAlgebra

import Base: -

_batched_doc = """
    batched_transpose(A::AbstractArray{T,3})
    batched_adjoint(A)

Equivalent to applying `transpose` or `adjoint` to each matrix `A[:,:,k]`.

These exist to control how `batched_mul` behaves,
as it operates on such matrix slices of an array with `ndims(A)==3`.

`PermutedDimsArray(A, (2,1,3))` is equivalent to `batched_transpose(A)`,
and is also understood by `batched_mul` (and more widely supported elsewhere).

    BatchedTranspose{T, S} <: AbstractBatchedMatrix{T, 3}
    BatchedAdjoint{T, S}

Lazy wrappers analogous to `Transpose` and `Adjoint`, returned by `batched_transpose` etc.
"""

@doc _batched_doc
struct BatchedTranspose{T, S} <: AbstractArray{T, 3}
    parent::S
    BatchedTranspose{T, S}(X::S) where {T, S} = new{T, S}(X)
end

@doc _batched_doc
batched_transpose(A::AbstractArray{T}) where T = BatchedTranspose(A)
batched_transpose(A::BatchedTranspose) = A.parent

@doc _batched_doc
struct BatchedAdjoint{T, S} <: AbstractArray{T, 3}
    parent::S
    BatchedAdjoint{T, S}(X::S) where {T, S} = new{T, S}(X)
end

@doc _batched_doc
batched_adjoint(A::AbstractArray{T, 3}) where T = BatchedAdjoint(A)
batched_adjoint(A::BatchedAdjoint) = A.parent

batched_adjoint(A::BatchedTranspose{<:Real}) = A.parent
batched_transpose(A::BatchedAdjoint{<:Real}) = A.parent
batched_adjoint(A::PermutedDimsArray{<:Real,3,(2,1,3)}) = A.parent
batched_transpose(A::PermutedDimsArray{<:Number,3,(2,1,3)}) = A.parent
# if you can't unwrap, put BatchedAdjoint outside (for dispatch):
batched_transpose(A::BatchedAdjoint{<:Complex}) = BatchedAdjoint(BatchedTranspose(A.parent))

BatchedAdjoint(A) = BatchedAdjoint{Base.promote_op(adjoint,eltype(A)),typeof(A)}(A)
BatchedTranspose(A) = BatchedTranspose{Base.promote_op(transpose,eltype(A)),typeof(A)}(A)


const BatchedAdjOrTrans{T, S} = Union{BatchedTranspose{T, S}, BatchedAdjoint{T, S}}

LinearAlgebra.wrapperop(A::BatchedAdjoint) = batched_adjoint
LinearAlgebra.wrapperop(B::BatchedTranspose) = batched_transpose

# AbstractArray Interface
Base.length(A::BatchedAdjOrTrans) = length(A.parent)
Base.size(m::BatchedAdjOrTrans) = (size(m.parent, 2), size(m.parent, 1), size(m.parent, 3))
Base.axes(m::BatchedAdjOrTrans) = (axes(m.parent, 2), axes(m.parent, 1), axes(m.parent, 3))

Base.IndexStyle(::Type{<:BatchedAdjOrTrans}) = IndexCartesian()
Base.@propagate_inbounds Base.getindex(m::BatchedTranspose, i::Int, j::Int, k::Int) = getindex(m.parent, j, i, k)
Base.@propagate_inbounds Base.getindex(m::BatchedAdjoint, i::Int, j::Int, k::Int) = adjoint(getindex(m.parent, j, i, k))
Base.@propagate_inbounds Base.setindex!(m::BatchedTranspose, v, i::Int, j::Int, k::Int) = setindex!(m.parent, v, j, i, k)
Base.@propagate_inbounds Base.setindex!(m::BatchedAdjoint, v, i::Int, j::Int, k::Int) = setindex!(m.parent, adjoint(v), j, i, k)

Base.similar(A::BatchedAdjOrTrans, T::Type, dims::Dims) = similar(A.parent, T, dims)
Base.similar(A::BatchedAdjOrTrans, dims::Dims) = similar(A.parent, dims)
Base.similar(A::BatchedAdjOrTrans, T::Type) = similar(A.parent, T, size(A))
Base.similar(A::BatchedAdjOrTrans) = similar(A.parent, size(A))

Base.parent(A::BatchedAdjOrTrans) = A.parent

(-)(A::BatchedAdjoint)   = BatchedAdjoint(  -A.parent)
(-)(A::BatchedTranspose) = BatchedTranspose(-A.parent)

# C interface
function Base.strides(A::Union{BatchedTranspose, BatchedAdjoint{<:Real}})
    sp = strides(A.parent)
    (sp[2], sp[1], sp[3])
end

function Base.stride(A::Union{BatchedTranspose, BatchedAdjoint{<:Real}}, d::Integer)
    d == 1 && return Base.stride(A.parent, 2)
    d == 2 && return Base.stride(A.parent, 1)
    Base.stride(A.parent, d)
end

Base.unsafe_convert(::Type{Ptr{T}}, A::BatchedAdjOrTrans{T}) where {T} =
    Base.unsafe_convert(Ptr{T}, parent(A))

# Gradients
function rrule(::typeof(batched_transpose), A::AbstractArray{<:Any,3})
    b_transpose_back(Δ) = (NO_FIELDS, batched_transpose(Δ))
    batched_transpose(A), b_transpose_back
end
function rrule(::typeof(batched_adjoint), A::AbstractArray{<:Any,3})
    b_adjoint_back(Δ) = (NO_FIELDS, batched_adjoint(Δ))
    batched_adjoint(A), b_adjoint_back
end
