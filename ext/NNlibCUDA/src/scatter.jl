for op in [+, -, *, /, max, min, &, |]
    @eval function scatter_kernel!(op::typeof($(op)), dst, src, idx)
        index = threadIdx().x + (blockIdx().x - 1) * blockDim().x

        @inbounds if index <= length(idx)
            @atomic dst[idx[index]...] = $(op)(dst[idx[index]...], src[index])
        end
        return nothing
    end

    @eval function scatter_kernel!(op::typeof($(op)), dst, src, idx, dims::Val{N}, max_idx, max_dims_idx, dims_size) where {N}
        index = threadIdx().x + (blockIdx().x - 1) * blockDim().x

        @inbounds if index <= max_idx
            j, k = divrem(index-1, max_dims_idx)
            dims_i = CartesianIndices(dims_size)[k+1]
            @atomic dst[Tuple(dims_i)..., idx[j+1]...] = $(op)(dst[Tuple(dims_i)..., idx[j+1]...], src[index])
        end
        return nothing
    end

    @eval function NNlib.scatter!(op::typeof($(op)), dst::AnyCuArray{Tdst}, src::AnyCuArray{Tsrc}, idx::AnyCuArray{<:IntOrIntTuple}, dims::Val{N}) where {Tdst,Tsrc,N}
        args = if N == 0
            max_idx = length(idx)
            op, dst, src, idx
        else
            dims_size = size(dst)[1:N]
            max_dims_idx = prod(dims_size)
            max_idx = max_dims_idx * length(idx)
            op, dst, src, idx, dims, max_idx, max_dims_idx, dims_size
        end

        kernel = @cuda launch=false scatter_kernel!(args...)
        config = launch_configuration(kernel.fun; max_threads=256)
        threads = min(max_idx, config.threads)
        blocks = ceil(Int, max_idx / threads)
        kernel(args...; threads=threads, blocks=blocks)
        return dst
    end
end
