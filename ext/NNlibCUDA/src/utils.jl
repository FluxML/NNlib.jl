function NNlib.count_indices(idx::AnyCuArray)
    dst_counts = length.(NNlib.reverse_indices(idx))
    src_counts = NNlib.gather(cu(dst_counts), idx)
    return src_counts
end

function divide_kernel!(xs, ys, max_idx)
    index = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    @inbounds if index <= max_idx
        xs[index] = xs[index] / ys[index]
    end
    return nothing
end

function divide_kernel!(xs, counts, max_idx, max_dims_idx, dims_size)
    index = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    @inbounds if index <= max_idx
        j, k = divrem(index-1, max_dims_idx)
        dims_i = Tuple(CartesianIndices(dims_size)[k+1])
        CUDA.@atomic xs[dims_i..., j+1] = xs[dims_i..., j+1] / counts[j+1]
    end
    return nothing
end

function NNlib.divide_by_counts!(xs::AnyCuArray{T}, idx::AnyCuArray, dims) where {T}
    counts = CuArray{T}(NNlib.count_indices(idx))
    args = if dims == 0
        max_idx = length(idx)
        xs, counts, max_idx
    else
        dims_size = size(xs)[1:dims]
        max_dims_idx = prod(dims_size)
        max_idx = prod(size(xs))
        xs, counts, max_idx, max_dims_idx, dims_size
    end

    kernel = @cuda launch=false divide_kernel!(args...)
    config = launch_configuration(kernel.fun; max_threads=256)
    threads = min(max_idx, config.threads)
    blocks = cld(max_idx, threads)
    kernel(args...; threads=threads, blocks=blocks)
    return xs
end

function NNlib.reverse_indices(idx::AnyCuArray{<:Any,N}) where N
    max_dims = NNlib.maximum_dims(idx)
    T = CartesianIndex{N}
    rev = Array{Vector{T}}(undef, max_dims...)
    for i in eachindex(rev)
        rev[i] = T[]
    end
    NNlib.reverse_indices!(rev, idx)
    return map(cu, rev)
end
