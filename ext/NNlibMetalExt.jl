module NNlibMetalExt


using Metal: Metal, MtlArray, MtlDeviceArray, method_table, @device_override
using NNlib: NNlib, BatchedAdjoint, BatchedTranspose, BatchedAdjOrTrans,
             batched_transpose, batched_adjoint, batched_mul_generic!
using Adapt: adapt, WrappedArray

@device_override NNlib.tanh_fast(x) = Base.FastMath.tanh_fast(x)

# Atomix's Metal atomics only support a subset of reduction ops: they raise a
# compile-time error for float `max`/`min` and silently no-op for `*`/`/`. Route
# scatter atomics through `Metal.@atomic` instead, which falls back to a generic
# compare-and-swap loop for ops without a native atomic.
# See https://github.com/FluxML/NNlib.jl/issues/534.
@inline function NNlib._atomic_scatter!(dst::MtlDeviceArray, idx, op::OP, val) where OP
    @inbounds Metal.@atomic dst[idx...] = op(dst[idx...], val)
    return nothing
end

# Batched matrix multiplication, see https://github.com/FluxML/NNlib.jl/issues/581.
#
# The first argument is produced by `NNlib.storage_type(A)`. By the time
# `_batched_gemm!` is reached, NNlib's stride analysis has reduced each strided
# operand to a plain `MtlArray` plus a BLAS-style transpose flag (`transA`/`transB`).
#
# Metal Performance Shaders' `matmul!` natively handles 3D (batched) inputs and
# transposition, but only for real `Float16`/`Float32`, and only when the batch
# dimensions match (it cannot broadcast a size-1 batch). Everything else — complex
# eltypes, batch broadcasting, or operands that did not reduce to a plain `MtlArray`
# — is routed through NNlib's generic per-slice `mul!`, which stays correct.

# Re-apply a BLAS transpose flag as the lazy batched wrapper understood by
# `batched_mul_generic!`.
_rewrap_batched(t::Char, X) = t == 'N' ? X : t == 'T' ? batched_transpose(X) : batched_adjoint(X)

function NNlib._batched_gemm!(::Type{<:MtlArray}, transA::Char, transB::Char,
                              α::Number, A, B, β::Number, C)
    if eltype(C) <: Union{Float16, Float32} && A isa MtlArray && B isa MtlArray &&
            size(A, 3) == size(C, 3) && size(B, 3) == size(C, 3)
        Metal.MPS.matmul!(C, A, B, α, β, transA != 'N', transB != 'N')
    else
        batched_mul_generic!(C, _rewrap_batched(transA, A), _rewrap_batched(transB, B), α, β)
    end
    return C
end

# Pretty-printing and conversion of a batched adjoint/transpose of an `MtlArray`,
# mirroring `NNlibCUDAExt`: route through the CPU to avoid scalar indexing.
const MtlBatchedAdjoint{T} = BatchedAdjoint{T, <:MtlArray{T}}
const MtlBatchedTranspose{T} = BatchedTranspose{T, <:MtlArray{T}}
const MtlBatchedAdjOrTrans{T} = Union{MtlBatchedAdjoint{T}, MtlBatchedTranspose{T}}
const WrappedMtlBatchedAdjOrTrans{T, N} = WrappedArray{T, N, MtlBatchedAdjOrTrans{T}, MtlBatchedAdjOrTrans{T}}

Base.print_array(io::IO, b::Union{MtlBatchedAdjOrTrans, WrappedMtlBatchedAdjOrTrans}) =
    Base.print_array(io, adapt(Array, b))
Base._show_nonempty(io::IO, b::Union{MtlBatchedAdjOrTrans, WrappedMtlBatchedAdjOrTrans}, prefix::String) =
    Base._show_nonempty(io, adapt(Array, b), prefix)
Base.show_vector(io::IO, b::Union{MtlBatchedAdjOrTrans, WrappedMtlBatchedAdjOrTrans}, opn, cls) =
    Base.show_vector(io, adapt(Array, b), opn, cls)

Base.convert(::Type{T}, b::Union{MtlBatchedAdjOrTrans, WrappedMtlBatchedAdjOrTrans}) where {T<:Array} =
    Base.convert(T, adapt(Array, b))
Base.Array{T, N}(b::Union{MtlBatchedAdjOrTrans, WrappedMtlBatchedAdjOrTrans}) where {T, N} =
    Array{T, N}(adapt(Array, b))
Base.collect(b::Union{MtlBatchedAdjOrTrans, WrappedMtlBatchedAdjOrTrans}) =
    collect(adapt(Array, b))

const MtlFloatArray{T, N} = MtlArray{T, N} where {T<:Union{Float16, Float32}, N}

function supports_mpsgraph_softmax(x::MtlArray, dims)
    return dims isa Integer && 1 <= dims <= ndims(x)
end

function NNlib.softmax!(y::MtlFloatArray{T, N}, x::MtlFloatArray{T, N};
                        dims = 1) where {T, N}
    supports_mpsgraph_softmax(x, dims) || return NNlib._softmax!(y, x; dims)
    return Metal.MPSGraphs.graph_softmax!(y, x; dims)
end

function NNlib.logsoftmax!(y::MtlFloatArray{T, N}, x::MtlFloatArray{T, N};
                           dims = 1) where {T, N}
    supports_mpsgraph_softmax(x, dims) || return NNlib._logsoftmax!(y, x; dims)
    return Metal.MPSGraphs.graph_logsoftmax!(y, x; dims)
end

function NNlib.∇softmax!(dx::MtlFloatArray{T, N}, dy::MtlFloatArray{T, N},
                         y::MtlFloatArray{T, N}; dims = 1) where {T, N}
    supports_mpsgraph_softmax(y, dims) || return NNlib._∇softmax!(dx, dy, y; dims)
    return Metal.MPSGraphs.graph_softmax_grad!(dx, dy, y; dims)
end

function NNlib.∇logsoftmax!(dx::MtlFloatArray{T, N}, dy::MtlFloatArray{T, N},
                            y::MtlFloatArray{T, N}; dims = 1) where {T, N}
    supports_mpsgraph_softmax(y, dims) || return NNlib._∇logsoftmax!(dx, dy, y; dims)
    return Metal.MPSGraphs.graph_logsoftmax_grad!(dx, dy, y; dims)
end

function supports_mpsgraph_conv(cdims)
    return cdims isa NNlib.DenseConvDims{2} && NNlib.groupcount(cdims) == 1
end

function NNlib.conv!(y::MtlFloatArray{T, 4}, x::MtlFloatArray{T, 4},
                     w::MtlFloatArray{T, 4}, cdims::NNlib.DenseConvDims{2};
                     kwargs...) where {T}
    if supports_mpsgraph_conv(cdims) && isempty(kwargs)
        return Metal.MPSGraphs.graph_conv!(y, x, w; stride=NNlib.stride(cdims),
                                           padding=NNlib.padding(cdims),
                                           dilation=NNlib.dilation(cdims),
                                           groups=NNlib.groupcount(cdims),
                                           flipkernel=NNlib.flipkernel(cdims))
    else
        return NNlib.conv_im2col!(y, x, w, cdims; kwargs...)
    end
end

function supports_mpsgraph_pool(pdims)
    return pdims isa NNlib.PoolDims{2}
end

function NNlib.maxpool!(y::MtlFloatArray{T, 4}, x::MtlFloatArray{T, 4},
                        pdims::NNlib.PoolDims{2}; kwargs...) where {T}
    if supports_mpsgraph_pool(pdims) && isempty(kwargs)
        return Metal.MPSGraphs.graph_maxpool!(y, x; kernel=NNlib.kernel_size(pdims),
                                              stride=NNlib.stride(pdims),
                                              padding=NNlib.padding(pdims),
                                              dilation=NNlib.dilation(pdims))
    else
        return NNlib.maxpool_direct!(y, x, pdims; kwargs...)
    end
end

function NNlib.meanpool!(y::MtlFloatArray{T, 4}, x::MtlFloatArray{T, 4},
                         pdims::NNlib.PoolDims{2}; count_include_pad = true,
                         kwargs...) where {T}
    if supports_mpsgraph_pool(pdims) && isempty(kwargs)
        return Metal.MPSGraphs.graph_meanpool!(y, x; kernel=NNlib.kernel_size(pdims),
                                               stride=NNlib.stride(pdims),
                                               padding=NNlib.padding(pdims),
                                               dilation=NNlib.dilation(pdims),
                                               count_include_pad)
    else
        return NNlib.meanpool_direct!(y, x, pdims; count_include_pad, kwargs...)
    end
end

end
