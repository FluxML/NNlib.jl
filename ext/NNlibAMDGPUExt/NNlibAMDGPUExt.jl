module NNlibAMDGPUExt

using Adapt
using AMDGPU
using AMDGPU.MIOpen
using ChainRulesCore
using NNlib
using NNlib: BatchedAdjoint, BatchedTranspose, BatchedAdjOrTrans
using NNlib: DenseConvDims, PoolDims

const MIOPENFloat = Union{Float16, Float32}

const ROCBatchedAdjoint{T} = BatchedAdjoint{T, <: ROCArray{T}}
const ROCBatchedTranspose{T} = BatchedTranspose{T, <: ROCArray{T}}
const ROCBatchedAdjOrTrans{T} = Union{ROCBatchedAdjoint{T}, ROCBatchedTranspose{T}}
const WrappedROCBatchedAdjOrTrans{T, N} = Adapt.WrappedArray{T, N, ROCBatchedAdjOrTrans{T}, ROCBatchedAdjOrTrans{T}}
const AnyROCBatchedAdjOrTrans = Union{ROCBatchedAdjOrTrans, WrappedROCBatchedAdjOrTrans}

function Base.convert(::Type{T}, b::AnyROCBatchedAdjOrTrans) where {T <: Array}
    Base.convert(T, adapt(Array, b))
end

function Base.Array{T, N}(b::AnyROCBatchedAdjOrTrans) where {T, N}
    Array{T, N}(adapt(Array, b))
end

Base.collect(b::AnyROCBatchedAdjOrTrans) = collect(adapt(Array, b))

function Base.show(
    io::IO, mime::MIME{Symbol("text/plain")}, x::AnyROCBatchedAdjOrTrans,
)
    show(io, mime, adapt(Array, x))
end

Base.show(io::IO, x::AnyROCBatchedAdjOrTrans) = show(io, adapt(Array, x))

Base.display(x::AnyROCBatchedAdjOrTrans) = display(adapt(Array, x))

function NNlib._batched_gemm!(
    ::Type{<: ROCArray}, transA::Char, transB::Char, α, A, B, β, C,
)
    AMDGPU.rocBLAS.gemm_batched!(transA, transB, α, A, B, β, C)
end

function nnlib_padding(dims)
    pd = NNlib.padding(dims)
    if !all(pd[1:2:end] .== pd[2:2:end])
        @warn """
        MIOpen does not support asymmetric padding, defaulting to symmetric choice:
        $pd -> $(pd[1:2:end]).
        """ maxlog=1
    end
    pd[1:2:end]
end

include("conv.jl")
include("pool.jl")
include("softmax.jl")
include("activations.jl")

end
