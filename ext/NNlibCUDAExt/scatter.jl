# supported op: +, -, *, /, max, min, &, |, mean

## TODO support sparse dst/src/idx
## See issue https://github.com/FluxML/NNlib.jl/issues/647
# import CUDA.CUSPARSE: AbstractCuSparseMatrix, CuSparseMatrixCSC, CuSparseMatrixCSR, CuSparseMatrixCOO, AnyCuSparseVector
# const AnyCuSparseMatrix{Tv,Ti} = Union{
#     AbstractCuSparseMatrix{Tv,Ti},
#     CUDA.CuSparseMatrixCSC{Tv,Ti}, # these types do not inherit from AbstractCuSparseMatrix
#     CUDA.CuSparseMatrixCSR{Tv,Ti}, # but from GPUArrays.AbstractGPUSparseMatrixXXX
#     CUDA.CuSparseMatrixCOO{Tv,Ti},
#     }
# const AnyCuSparseArray{Tv,Ti} = Union{AnyCuSparseVector{Tv,Ti},AnyCuSparseMatrix{Tv,Ti}}

# `dst[I...] = op(dst[I...], val)`, atomically
@inline function atomic_update!(op::OP, dst, val, I...) where OP
    CUDA.@atomic dst[I...] = op(dst[I...], val)
    return nothing
end

const HalfFloat = Union{Float16, BFloat16}

@inline atomic_update!(::typeof(+), dst::CuDeviceArray{T}, val, I...) where T<:HalfFloat =
    atomic_add_paired!(dst, Base._to_linear_index(dst, I...), convert(T, val))
@inline atomic_update!(::typeof(-), dst::CuDeviceArray{T}, val, I...) where T<:HalfFloat =
    atomic_add_paired!(dst, Base._to_linear_index(dst, I...), -convert(T, val))

# 16-bit atomic adds serialize when many threads add to the same element, while the
# paired f16x2/bf16x2 ones are as fast as Float32 (#854). As PyTorch does, add `val`
# and a zero to the aligned 32-bit word that holds `dst[i]`. The zero is -0.0, which
# leaves every neighbour unchanged (+0.0 would turn a -0.0 into +0.0).
@inline function atomic_add_paired!(dst::CuDeviceArray{T}, i::Integer, val::T) where T
    p = reinterpret(UInt64, pointer(dst, i))
    low = iszero(p % 4)
    if has_paired_add(T) && (low ? i < length(dst) : i > 1)
        bits = UInt32(reinterpret(UInt16, val))
        word = low ? bits | 0x8000_0000 : bits << 16 | 0x0000_8000
        red_add_x2!(T, low ? p : p - 2, word)
    else
        CUDA.@atomic dst[i] += val
    end
    return nothing
end

@inline has_paired_add(::Type{Float16}) = CUDA.compute_capability().major >= 6
@inline has_paired_add(::Type{BFloat16}) = CUDA.compute_capability().major >= 9

for (T, ptx_type) in ((Float16, "f16x2"), (BFloat16, "bf16x2"))
    asm = "red.global.add.noftz.$ptx_type [\$0], \$1;"
    @eval @inline red_add_x2!(::Type{$T}, p::UInt64, word::UInt32) = Base.llvmcall(
        $("""call void asm sideeffect "$asm", "l,r,~{memory}"(i64 %0, i32 %1)
             ret void"""), Cvoid, Tuple{UInt64, UInt32}, p, word)
end

function scatter_kernel!(op::OP, dst, src, idx) where OP
    index = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    @inbounds if index <= length(idx)
        atomic_update!(op, dst, src[index], idx[index]...)
    end
    return nothing
end

function scatter_kernel!(op::OP, dst, src, idx::CUDA.CuDeviceArray{<:CartesianIndex}) where OP
    index = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    @inbounds if index <= length(idx)
        li = Base._to_linear_index(dst, Tuple(idx[index])...)
        atomic_update!(op, dst, src[index], li)
    end
    return nothing
end

function scatter_kernel!(op::OP, dst, src, idx, max_idx, max_dims_idx, dims_size) where OP
    index = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    @inbounds if index <= max_idx
        j, k = divrem(index-1, max_dims_idx)
        dims_i = CartesianIndices(dims_size)[k+1]
        atomic_update!(op, dst, src[index], Tuple(dims_i)..., idx[j+1]...)
    end
    return nothing
end

function scatter_kernel!(op::OP, dst, src, idx::CUDA.CuDeviceArray{<:CartesianIndex},
            max_idx, max_dims_idx, dims_size) where OP
    index = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    @inbounds if index <= max_idx
        j, k = divrem(index-1, max_dims_idx)
        dims_i = CartesianIndices(dims_size)[k+1]
        li = Base._to_linear_index(dst, Tuple(dims_i)..., Tuple(idx[j+1])...)
        atomic_update!(op, dst, src[index], li)
    end
    return nothing
end


function NNlib.scatter!(op::OP, dst::AnyCuArray,
        src::AnyCuArray,
        idx::AnyCuArray) where OP
    isempty(idx) && return dst
    dims = NNlib.scatter_dims(dst, src, idx)
    NNlib.checkbounds_idx(idx, size(dst)[dims+1:end])  # the kernels below are @inbounds (#416)
    args = if dims == 0
        max_idx = length(idx)
        op, dst, src, idx
    else
        dims_size = size(dst)[1:dims]
        max_dims_idx = prod(dims_size)
        max_idx = max_dims_idx * length(idx)
        op, dst, src, idx, max_idx, max_dims_idx, dims_size
    end

    kernel = @cuda launch=false scatter_kernel!(args...)
    config = launch_configuration(kernel.fun; max_threads=256)
    threads = min(max_idx, config.threads)
    blocks = cld(max_idx, threads)
    kernel(args...; threads=threads, blocks=blocks)
    return dst
end

function NNlib.scatter!(op::typeof(mean), dst::AnyCuArray,
        src::AnyCuArray,
        idx::AnyCuArray)
    Ns = NNlib.scatter!(+, zero(dst), one.(src), idx)
    dst_ = NNlib.scatter!(+, zero(dst), src, idx)
    dst .+= NNlib.safe_div.(dst_, Ns)
    return dst
end


## Gradients

function ∇scatter_src_kernel!(op::OP, Δsrc, src, idx,
    rev_idx, max_idx, T::Type{TT}) where {OP,TT}
    index = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    @inbounds if index <= max_idx
        cart_j = CartesianIndices(idx)[index]
        # get aggregating indeices, which is to be aggregated together, and itself index
        inds = rev_idx[idx[cart_j]...]
        # multiply all values to be aggregated but not itself
        x = one(T)
        for k in inds
            x *= src[k]
        end
        x /= src[cart_j]
        # apply `op` on `Δsrc[i, k]` and `x`
        Δsrc[cart_j] = op(Δsrc[cart_j], x)
    end
    return nothing
end

function ∇scatter_src_kernel!(op::OP, Δsrc, src, idx::CUDA.CuDeviceArray{<:CartesianIndex},
            rev_idx, max_idx, T::Type{TT}) where {OP,TT}
    index = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    @inbounds if index <= max_idx
        cart_j = CartesianIndices(idx)[index]
        # get aggregating indeices, which is to be aggregated together, and itself index
        inds = rev_idx[Tuple(idx[cart_j])...]
        # multiply all values to be aggregated but not itself
        x = one(T)
        for k in inds
            x *= src[k]
        end
        x /= src[cart_j]
        # apply `op` on `Δsrc[i, k]` and `x`
        Δsrc[cart_j] = op(Δsrc[cart_j], x)
    end
    return nothing
end

function ∇scatter_src_kernel!(op::OP, Δsrc, src, idx,
    rev_idx, pre_cart_idx, max_dims_idx, max_idx, T::Type{TT}) where {OP,TT}
    index = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    @inbounds if index <= max_idx
        i, j = fldmod1(index, max_dims_idx)
        cart_i = CartesianIndices(idx)[i]
        cart_j = pre_cart_idx[j]
        # get aggregating indeices, which is to be aggregated together, and itself index
        inds = rev_idx[idx[cart_i]...]
        # multiply all values to be aggregated but not itself
        x = one(T)
        for k in inds
            jk = Base._to_linear_index(src, Tuple(cart_j)..., Tuple(k)...)
            x *= src[jk]
        end
        x /= src[index]
        # apply `op` on `Δsrc[i, k]` and `x`
        Δsrc[index] = op(Δsrc[index], x)
    end
    return nothing
end

function ∇scatter_src_kernel!(op::OP, Δsrc, src, idx::CUDA.CuDeviceArray{<:CartesianIndex},
                rev_idx, pre_cart_idx, max_dims_idx, max_idx, T::Type{TT}) where {OP,TT}
    index = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    @inbounds if index <= max_idx
        i, j = fldmod1(index, max_dims_idx)
        cart_i = CartesianIndices(idx)[i]
        cart_j = pre_cart_idx[j]
        # get aggregating indeices, which is to be aggregated together, and itself index
        inds = rev_idx[Tuple(idx[cart_i])...]
        # multiply all values to be aggregated but not itself
        x = one(T)
        for k in inds
            jk = Base._to_linear_index(src, Tuple(cart_j)..., Tuple(k)...)
            x *= src[jk]
        end
        x /= src[index]
        # apply `op` on `Δsrc[i, k]` and `x`
        Δsrc[index] = op(Δsrc[index], x)
    end
    return nothing
end

function NNlib.∇scatter_src(op::Union{typeof(*),typeof(/)}, Δ, dst,
    src::AnyCuArray,
    idx::AnyCuArray)
    dims = ndims(src) - ndims(idx)
    Δsrc = NNlib.modify_src(op, NNlib.gather(Δ, idx), src)
    rev_idx = NNlib.reverse_indices(idx)
    rev_idx = CuArray(map(CUDA.cudaconvert, rev_idx))

    if dims == 0
        max_idx = length(idx)
        args = op, Δsrc, src, idx, rev_idx, max_idx, eltype(src)
    else
        pre_cart_idx = CartesianIndices(axes(src)[1:dims])
        max_dims_idx = length(pre_cart_idx)
        max_idx = max_dims_idx * length(idx)
        args = op, Δsrc, src, idx, rev_idx, pre_cart_idx, max_dims_idx, max_idx, eltype(src)
    end

    kernel = @cuda launch=false ∇scatter_src_kernel!(args...)
    config = launch_configuration(kernel.fun; max_threads=256)
    threads = min(max_idx, config.threads)
    blocks = cld(max_idx, threads)
    kernel(args...; threads=threads, blocks=blocks)

    CUDA.unsafe_free!(rev_idx)
    return Δsrc
end
