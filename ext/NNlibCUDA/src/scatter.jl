# supported op: +, -, *, /, max, min, &, |, mean

function scatter_kernel!(op::OP, dst, src, idx) where OP
    index = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    @inbounds if index <= length(idx)
        CUDA.@atomic dst[idx[index]...] = op(dst[idx[index]...], src[index])
    end
    return nothing
end

function scatter_kernel!(op::OP, dst, src, idx::CUDA.CuDeviceArray{<:CartesianIndex}) where OP
    index = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    @inbounds if index <= length(idx)
        li = Base._to_linear_index(dst, Tuple(idx[index])...)
        CUDA.@atomic dst[li] = op(dst[li], src[index])
    end
    return nothing
end

function scatter_kernel!(op::OP, dst, src, idx, max_idx, max_dims_idx, dims_size) where OP
    index = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    @inbounds if index <= max_idx
        j, k = divrem(index-1, max_dims_idx)
        dims_i = CartesianIndices(dims_size)[k+1]
        CUDA.@atomic dst[Tuple(dims_i)..., idx[j+1]...] = op(dst[Tuple(dims_i)..., idx[j+1]...], src[index])
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
        CUDA.@atomic dst[li] = op(dst[li], src[index])
    end
    return nothing
end

function NNlib.scatter!(op::OP, dst::AnyCuArray, src::AnyCuArray, idx::AnyCuArray) where OP
    dims = NNlib.scatter_dims(dst, src, idx)
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

function NNlib.scatter!(op::typeof(mean), dst::AnyCuArray, src::AnyCuArray, idx::AnyCuArray)
    Ns = NNlib.scatter!(+, zero(dst), one.(src), idx)
    dst_ = NNlib.scatter!(+, zero(dst), src, idx)
    dst .+= NNlib.safe_div.(dst_, Ns)
    return dst
end


## Gradients

function ∇scatter_src_kernel!(op::OP, Δsrc, src, idx, 
                rev_idx, max_idx, T::Type{TT})  where {OP,TT}
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
                            src::AnyCuArray{Tsrc,Nsrc}, 
                            idx::AnyCuArray{Tidx,Nidx}) where {Tsrc,Tidx,Nsrc,Nidx}
    dims = Nsrc - Nidx
    Δsrc = NNlib.modify_src(op, NNlib.gather(Δ, idx), src)
    rev_idx = NNlib.reverse_indices(idx)
    rev_idx = CuArray(map(CUDA.cudaconvert, rev_idx))
    
    if dims == 0
        max_idx = length(idx)
        args = op, Δsrc, src, idx, rev_idx, max_idx, Tsrc
    else
        pre_cart_idx = CartesianIndices(axes(src)[1:dims])
        max_dims_idx = length(pre_cart_idx)
        max_idx = max_dims_idx * length(idx)
        args = op, Δsrc, src, idx, rev_idx, pre_cart_idx, max_dims_idx, max_idx, Tsrc
    end

    kernel = @cuda launch=false ∇scatter_src_kernel!(args...)
    config = launch_configuration(kernel.fun; max_threads=256)
    threads = min(max_idx, config.threads)
    blocks = cld(max_idx, threads)
    kernel(args...; threads=threads, blocks=blocks)

    CUDA.unsafe_free!(rev_idx)
    return Δsrc
end
