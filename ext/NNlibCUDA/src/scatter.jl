# supported op: +, -, *, /, max, min, &, |, mean

function scatter_kernel!(op, dst, src, idx)
    index = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    @inbounds if index <= length(idx)
        CUDA.@atomic dst[idx[index]...] = op(dst[idx[index]...], src[index])
    end
    return nothing
end

function scatter_kernel!(op, dst, src, idx, max_idx, max_dims_idx, dims_size)
    index = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    @inbounds if index <= max_idx
        j, k = divrem(index-1, max_dims_idx)
        dims_i = CartesianIndices(dims_size)[k+1]
        CUDA.@atomic dst[Tuple(dims_i)..., idx[j+1]...] = op(dst[Tuple(dims_i)..., idx[j+1]...], src[index])
    end
    return nothing
end

function NNlib.scatter!(op, dst::AnyCuArray, src::AnyCuArray, idx::AnyCuArray)
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
