using LinearAlgebra
import Base: -

"""
    BatchedTranspose{T, N, S} <: AbstractBatchedMatrix{T, N}
Batched transpose. Transpose a batch of matrix.
"""
struct BatchedTranspose{T, S} <: AbstractArray{T, 3}
    parent::S
    BatchedTranspose{T, S}(X::S) where {T, S} = new{T, S}(X)
end

"""
    batched_transpose(A)
Lazy batched transpose.
"""
batched_transpose(A::AbstractArray{T}) where T = BatchedTranspose(A)


"""
    BatchedAdjoint{T, N, S} <: AbstractBatchedMatrix{T, N}
Batched ajoint. Transpose a batch of matrix.
"""
struct BatchedAdjoint{T, S} <: AbstractArray{T, 3}
    parent::S
    BatchedAdjoint{T, S}(X::S) where {T, S} = new{T, S}(X)
end

"""
    batched_adjoint(A)
Lazy batched adjoint.
"""
batched_adjoint(A::AbstractArray{T, 3}) where T = BatchedAdjoint(A)


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
Base.@propagate_inbounds Base.setindex!(m::BatchedAdjOrTrans, v, i::Int, j::Int, k::Int) = setindex!(m.parent, v, j, i, k)

Base.similar(A::BatchedAdjOrTrans, T::Type, dims::Dims) = similar(A.parent, T, dims)
Base.similar(A::BatchedAdjOrTrans, dims::Dims) = similar(A.parent, dims)
Base.similar(A::BatchedAdjOrTrans, T::Type) = similar(A.parent, T, size(A))
Base.similar(A::BatchedAdjOrTrans) = similar(A.parent, size(A))

Base.parent(A::BatchedAdjOrTrans) = A.parent

(-)(A::BatchedAdjoint)   = BatchedAdjoint(  -A.parent)
(-)(A::BatchedTranspose) = BatchedTranspose(-A.parent)

Base.copy(A::BatchedTranspose) = BatchedTranspose(copy(A.parent))
Base.copy(A::BatchedAdjoint) = BatchedAdjoint(copy(A.parent))

